import cv2
import torch
import torch.nn as nn
import json
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from src.augments import complete_resize_pipeline, incomplete_resize_pipeline

parent = Path(__file__).resolve().parent.parent
train_complete_path = parent / "data" / "train_complete.json"
test_complete_path = parent / "data" / "test_complete.json"
valid_complete_path = parent / "data" / "valid_complete.json"
train_incomplete_path = parent / "data" / "train_incomplete.json"
test_incomplete_path = parent / "data" / "test_incomplete.json"
valid_incomplete_path = parent / "data" / "valid_incomplete.json"

class CompleteImageDataset(Dataset):
    def __init__(self, data_path, transform=None) -> None:
        self.data_path = data_path
        self.data = pd.read_json(self.data_path).T
        # print(self.data.head())
        self.labels = list(self.data["label"])
        self.image_paths = list(self.data["image"])
        xml = list(self.data["xml"])
        self.bbox = [i['bbox'] for i in xml]
        self.trimap = list(self.data["trimap"])
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image
        kaggle_path = "/kaggle/input/datasets/pranavraoda25/oxford-dataset" + self.image_paths[idx]
        image = cv2.imread(kaggle_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # type: ignore

        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image,
                                       bboxes=[self.bbox[idx]],
                                       class_labels=[self.labels[idx]]
                                       )
            
            image = augmented['image']

        return image, self.labels[idx]

class IncompleteImageDataset(Dataset):
    def __init__(self, data_path, transform=None) -> None:
        self.data_path = data_path
        self.data = pd.read_json(self.data_path).T
        self.labels = list(self.data["label"])
        self.image_paths = list(self.data["image"])
        # xml = list(self.data["xml"])
        # self.bbox = [i['bbox'] for i in xml]
        self.trimap = list(self.data["trimap"])
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image
        kaggle_path = "/kaggle/input/datasets/pranavraoda25/oxford-dataset" + self.image_paths[idx]
        image = cv2.imread(kaggle_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # type: ignore

        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        return image, self.labels[idx]
