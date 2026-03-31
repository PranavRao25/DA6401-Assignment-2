"""
SPLIT DATA
"""

import json
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

parent = Path(__file__).resolve().parent.parent

data_path = parent / "data" / "data.json"
with open(data_path, "r") as f:
    data = json.load(f)

complete_data = dict()
incomplete_data = dict()
for key in data:
    if len(data[key]['xml'])!=0:
        complete_data[key] = data[key]
    else:
        incomplete_data[key] = data[key]

with open(parent / "data" / "complete.json", "w") as f:
    json.dump(complete_data, f)
with open(parent / "data" / "incomplete.json", "w") as f:
    json.dump(incomplete_data, f)

complete_path = parent / "data" / "complete.json"
incomplete_path = parent / "data" / "incomplete.json"

# with open(complete_path, "r") as f:
#     complete_data = json.load(f)
# with open(incomplete_path, "r") as f:
#     incomplete_data = json.load(f)

print(len(complete_data), len(incomplete_data))  # 3671 3678

complete_df = pd.DataFrame(complete_data).T

print(complete_df.head())

train_complete, test_complete = train_test_split(complete_df, test_size=0.1, stratify=complete_df["label"])
train_complete, valid_complete = train_test_split(train_complete, test_size=0.15, stratify=train_complete["label"])

train_complete.T.to_json(parent / "data" / "train_complete.json")
valid_complete.T.to_json(parent / "data" / "valid_complete.json")
test_complete.T.to_json(parent / "data" / "test_complete.json")

incomplete_df = pd.DataFrame(incomplete_data).T
train_incomplete, test_incomplete = train_test_split(incomplete_df, test_size=0.1, stratify=incomplete_df["label"])
train_incomplete, valid_incomplete = train_test_split(train_incomplete, test_size=0.15, stratify=train_incomplete["label"])

train_incomplete.T.to_json(parent / "data" / "train_incomplete.json")
valid_incomplete.T.to_json(parent / "data" / "valid_incomplete.json")
test_incomplete.T.to_json(parent / "data" / "test_incomplete.json")
