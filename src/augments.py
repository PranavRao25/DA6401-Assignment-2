import json
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from pathlib import Path

TARGET_HEIGHT = 224
TARGET_WIDTH = 224

# A.Compose([
#     A.RandomCrop(height=224, width=224),      # Step 1: Size
#     A.HorizontalFlip(p=0.5),                  # Step 2: Basic geometric
#     A.CoarseDropout(num_holes_range=(0.02, 0.1),  # Step 3: Dropout
#                     hole_height_range=(0.05, 0.15),
#                     hole_width_range=(0.05, 0.15), p=0.5),
#     A.Normalize(),                            # Step 7: Normalization
# ], seed=137)


# IMAGE RESIZING
# complete

complete_resize_pipeline = A.Compose([
    A.LetterBox(size=(TARGET_HEIGHT, TARGET_WIDTH), fill=0, p=1.0),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
    ],
    bbox_params=A.BboxParams(coord_format='pascal_voc', # Specify input format # type: ignore
                           label_fields=['class_labels']), # Specify label argument name(s)
    seed=137)

incomplete_resize_pipeline = A.Compose([
    A.LetterBox(size=(TARGET_HEIGHT, TARGET_WIDTH), fill=0, p=1.0),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
    ],seed=137)

# **Important: Transform Compatibility**: Not all transforms support bounding boxes.
# Before building your pipeline, check https://albumentations.ai/docs/reference/supported-targets-by-transform/
# incomplete
