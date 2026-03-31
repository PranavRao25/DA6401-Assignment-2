import cv2
import torch
import torch.nn as nn
import json
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from augments import resize_pipeline

parent = Path(__file__).resolve().parent.parent
data_path = parent / "data" / "train_complete.json"

data = pd.read_json(data_path).T
xml = list(data["xml"])
print(type(xml[0]["bbox"]))
# bbox = data["xml"]["bbox"]  # TODO: check data type
