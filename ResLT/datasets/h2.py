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
# Load CLIP preprocessing
_, preprocess = clip.load("ViT-B/32", device="cuda" if torch.cuda.is_available() else "cpu")


@DATASET_REGISTRY.register()
class HerbariumDatasetTWO(DatasetBase):
    dataset_dir = "custom_dataset"

    def __init__(self, cfg):
        self.root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.meta_path = os.path.join(self.root, "train_metadata.json")
        self.preprocess = preprocess
        self.ratio = 0.4  # Ratio for dataset splitting
        self.count = 0
        # Load metadata
        if not os.path.exists(self.meta_path):
            raise FileNotFoundError(f"Metadata file not found at {self.meta_path}")
        with open(self.meta_path, 'r') as f:
            self.metadata = json.load(f)

        # Prepare genus classnames and dataset
        # print("Extracting Classnames")
        # self.genus_classnames = self.read_classnames()
        print("Extracting Train Samples")
        self.train_items = self.read_dataset()

        print(f"Files skipped: {self.count}")
        # Split dataset into train, val, and test
        print("Creating Dataset Split")
        self.split_dataset()

        super().__init__(train_x=self.train_items, val=self.val_items, test=self.test_items)

    # def read_classnames(self):
    #     """Load genus classnames from metadata and return an OrderedDict."""
    #     genera = self.metadata.get("genera", [])
    #     classnames = OrderedDict(
    #         (str(genus["genus_id"]), genus["genus"]) for genus in genera
    #     )
    #     return classnames

    # def read_dataset(self):
    #     """
    #     Filter dataset to contain unique genus categories,
    #     then convert it to Datum objects.
    #     """
    #     # Extract relevant metadata
    #     images = {img["image_id"]: img["file_name"] for img in self.metadata["images"]}
    #     annotations = self.metadata["annotations"]

    #     # Create Datum objects for unique genus annotations
    #     items = []
    #     for annot in annotations:
    #         image_id = annot["image_id"]
    #         impath = os.path.join(self.root, "Sample/", images[image_id])
    #         label = annot["genus_id"]
    #         classname = self.genus_classnames.get(str(label), "Unknown")

    #         if not os.path.exists(impath):
    #             self.count += 1
    #             continue  # Skip missing files

    #         items.append(Datum(impath=impath, label=label, classname=classname))

    #     return items


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

        items = []
        for _, row in tqdm(self.unique_samples.iterrows(), total=len(self.unique_samples), desc="Processing rows"):
            item = Datum(
                impath=row["directory"],
                label=int(row["genus_id"]),
                classname=row["genus"]
            )
            items.append(item)

        return items
    def split_dataset(self):
        """Split dataset into train, val, and test based on ratio."""
        total = len(self.train_items)
        # train_split = int(total * 0.8 * self.ratio)
        # val_split = int(total * 0.1 * self.ratio)
        print(f"Total Length of preprocessed data: {total}")
        train_split = int(total * 0.8)
        val_split = int(total * 0.1)
        self.train_items = self.train_items[:train_split]
        self.val_items = self.train_items[train_split:train_split + val_split]
        self.test_items = self.train_items[train_split + val_split:]

    def extract_unique_rows(self, columns: List[str]) -> pd.DataFrame:
        if not set(columns).issubset(self.train.columns):
            raise ValueError("Some specified columns do not exist in the DataFrame.")
        return self.train.drop_duplicates(subset=columns)

    def get_img(self, impath):
        """Load an image and preprocess it using CLIP."""
        if not os.path.exists(impath):
            raise FileNotFoundError(f"Image path not found: {impath}")
        img = Image.open(impath).convert("RGB")
        img = self.preprocess(img)
        return img
