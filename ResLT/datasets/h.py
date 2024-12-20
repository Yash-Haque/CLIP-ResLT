import json 
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import clip
import numpy as np
from PIL import Image
from typing import List

import os
import pickle
from collections import OrderedDict

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import listdir_nohidden, mkdir_if_missing

from clip import clip
import torch 

import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

dataset_dir = "custom_dataset"
_, preprocess = clip.load("ViT-B/32", device="cuda" if torch.cuda.is_available() else "cpu")


@DATASET_REGISTRY.register()
class HerbariumDataset(DatasetBase):
    """Custom Herbarium Dataset with CLIP preprocessing."""
    
    def __init__(self, cfg):
        # Parent constructor
        # super().__init__(cfg)

        # Initialize paths and dataset metadata
        root = cfg.DATASET.ROOT
        self.preprocess = preprocess
        self.ratio = 0.4
        self.train_dir = os.path.join(root, "Sample/")  # Directory with images
        self.meta_path = os.path.join(root, "train_metadata.json")
        
        # Load metadata from JSON file
        if not os.path.exists(self.meta_path):
            raise FileNotFoundError(f"Metadata file not found at {self.meta_path}")
        with open(self.meta_path, 'r') as json_file:
            self.train_meta = json.load(json_file)

        # Prepare the dataset
        self.read_dataset()
        
        super().__init__(train_x=self.train_items)


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
        unique_files = self.extract_unique_rows(["category"])
        self.file_paths = unique_files["directory"].astype(str).tolist()
        # self.split_index = int(len(self.file_paths) * self.ratio)
        # self.selected_rows = self.split_df()

        # Perform splits
        split_data = self.split_df()
        self.train_data = split_data["train"]
        self.val_data = split_data["val"]
        self.test_data = split_data["test"]

        # Pack items for DASSL compatibility
        # Validate image paths and labels, then create Datum objects
        self.train_items = []
        for _, row in self.train_data.iterrows():
            impath = row["directory"]
            label = row["genus_id"]

            if not os.path.exists(impath):
                logger.warning(f"Image path not found: {impath}. Skipping.")
                continue

            if not isinstance(label, int):
                logger.warning(f"Invalid label for image {impath}. Label: {label}. Skipping.")
                continue

            # Create Datum object and append to train_items
            self.train_items.append(Datum(impath=impath, label=label))
        # self.train_items = [
        #     Datum(impath=row["directory"], label=row["genus_id"], category=row["category"])
        #     for _, row in self.train_data.iterrows()
        # ]
        # self.val_items = [
        #     Datum(impath=row["directory"], label=row["genus_id"])
        #     for _, row in self.val_data.iterrows()
        # ]
        # self.test_items = [
        #     Datum(impath=row["directory"], label=row["genus_id"])
        #     for _, row in self.test_data.iterrows()
        # ]

    def extract_unique_rows(self, columns: List[str]) -> pd.DataFrame:
        if not set(columns).issubset(self.train.columns):
            raise ValueError("Some specified columns do not exist in the DataFrame.")
        return self.train.drop_duplicates(subset=columns)

    # def split_df(self) -> pd.DataFrame:
    #     return self.train.iloc[:self.split_index]

    def split_df(self):
        """Split the dataset into training, validation, and test sets."""
        total_samples = len(self.train)

        # Compute split indices based on the ratio
        train_split = int(total_samples * 0.8 * self.ratio)
        val_split = int(total_samples * 0.1 * self.ratio)

        # Split the DataFrame into train, validation, and test sets
        self.train_data = self.train.iloc[:train_split]
        self.val_data = self.train.iloc[train_split:train_split + val_split]
        self.test_data = self.train.iloc[train_split + val_split:]

        return {
            "train": self.train_data,
            "val": self.val_data,
            "test": self.test_data
        }

    def get_img(self, impath: str):
        """Load an image from path and apply CLIP preprocessing."""
        if not os.path.exists(impath):
            raise FileNotFoundError(f"Image path not found: {impath}")
        img = Image.open(impath).convert("RGB")
        img = self.preprocess(img)
        return img
