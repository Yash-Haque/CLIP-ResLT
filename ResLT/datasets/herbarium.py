# File: h.py
import json
import os
from collections import OrderedDict
from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import mkdir_if_missing
import pandas as pd
import numpy as np
from PIL import Image
import torch
from clip import clip
from tqdm import tqdm
from typing import List
from .oxford_pets import OxfordPets
import pickle
# Load CLIP preprocessing
# _, preprocess = clip.load("ViT-B/32", device="cuda" if torch.cuda.is_available() else "cpu")


@DATASET_REGISTRY.register()
class HERB(DatasetBase):
 

    def __init__(self, cfg):
        # Relevant variables
        self.root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.train_dir = os.path.join(self.root,"Samples/")
        self.train_meta_path = os.path.abspath(os.path.expanduser(cfg.DATASET.HERB.TRAIN_META))
        # self.test_meta_path = os.path.abspath(os.path.expanduser(cfg.DATASET.HERB.TEST_META))
        self.preprocessed = os.path.join(self.root,"preprocessed.pkl")
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")

        # Irrelevant Variable
        self.ratio = 0.4  # Ratio for dataset splitting


        mkdir_if_missing(self.split_fewshot_dir)
        if os.path.exists(self.preprocessed):
            with open(self.preprocessed, "rb") as f:
                preprocessed = pickle.load(f)
                train = preprocessed["train"]
                test = preprocessed["test"]
        else:
            # Load train metadata
            print(f'Preprocessed File "{self.preprocessed}" Does NOT Exist.')
            if not os.path.exists(self.meta_path):
                raise FileNotFoundError(f"Metadata file not found at {self.meta_path}")
            with open(self.train_meta_path, 'r') as f:
                self.train_meta = json.load(f)

            print("Extracting Train Samples")
            train_items, test_items = self.read_dataset()

            # Not Relevant
            # print("Creating Dataset Split")
            # self.split_dataset(train_items)
            # Not Relevant

            train = self.train_items
            test = self.test_items

            preprocessed = {'train': train, 'test': test}
            with open(self.preprocessed, "wb") as f:
                pickle.dump(preprocessed, f, protocol=pickle.HIGHEST_PROTOCOL) 

        num_shots = cfg.DATASET.NUM_SHOTS
        if num_shots >= 1:
            seed = cfg.SEED
            preprocessed = os.path.join(self.split_fewshot_dir, f"shot_{num_shots}-seed_{seed}.pkl")
            
            if os.path.exists(preprocessed):
                print(f"Loading preprocessed few-shot data from {preprocessed}")
                with open(preprocessed, "rb") as file:
                    data = pickle.load(file)
                    print(data)
                    train, test = data["train"], data["test"]
            else:
                train = self.generate_fewshot_dataset(train, num_shots=num_shots)
                test = self.generate_fewshot_dataset(test, num_shots=min(num_shots, 4))
                data = {'train': train, 'test': test}
                print(f"Saving preprocessed few-shot data to {preprocessed}")
                with open(preprocessed, "wb") as file:
                    pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)
            
        subsample = cfg.DATASET.SUBSAMPLE_CLASSES
        train, test = OxfordPets.subsample_classes(train, test, subsample=subsample)
        # print(f"Train: {test}")
        super().__init__(train_x=train, val=test, test=test)


    def read_dataset(self):
        # Extract columns from metadata
        image_ids = [image["image_id"] for image in self.train_meta["images"]]
        image_dirs = [os.path.join(self.train_dir, image["file_name"]) for image in self.train_meta["images"]]
        category_ids = [annot["category_id"] for annot in self.train_meta["annotations"]]
        genus_ids = [annot["genus_id"] for annot in self.train_meta["annotations"]]
        
        # Create the main training DataFrame
        self.train = pd.DataFrame(
            data=np.array([image_ids, image_dirs, genus_ids, category_ids]).T,
            columns=["image_id", "directory", "genus_id", "category"]
        )
        
        # Merge genus names into the DataFrame
        genera = pd.DataFrame(self.train_meta['genera'])
        self.train["genus_id"] = pd.to_numeric(self.train["genus_id"], errors='coerce')
        genera["genus_id"] = pd.to_numeric(genera["genus_id"], errors='coerce')
        self.train = self.train.merge(genera, on='genus_id')

        # Extract unique rows and split the DataFrame
        self.unique_samples = self.extract_unique_rows(["category"])
        sample_count = int(len(self.unique_samples) * self.ratio)
        self.unique_samples = self.unique_samples.iloc[:sample_count]

        items = []
        for _, row in tqdm(self.unique_samples.iterrows(), total=len(self.unique_samples), desc="Processing rows"):
            item = Datum(
                impath=row["directory"],
                label=int(row["genus_id"]),
                classname=row["genus"]
            )
            items.append(item)

        return items

    def split_dataset(self, items):
        total = len(items)
        train_split = int(total * 0.8)
        
        # Preserve original list, create new lists
        self.train_items = items[:train_split]
        self.test_items = items[train_split:]
        

    def extract_unique_rows(self, columns: List[str]) -> pd.DataFrame:
        if not set(columns).issubset(self.train.columns):
            raise ValueError("Some specified columns do not exist in the DataFrame.")
        return self.train.drop_duplicates(subset=columns)
