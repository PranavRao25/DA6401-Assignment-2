from PIL import Image
import torch
import yaml
from pathlib import Path
import pathlib
import xml.etree.ElementTree as ET

def read_annotations(folder_path):
    folder = Path(folder_path)
    data = dict()
    for f in folder.iterdir():
        tree = ET.parse(f)
        root = tree.getroot()
        
        file_name = root.find("filename")
        bndbox = []
        for bndbox in root.findall("bndbox"):
            bbox = [
                bndbox.find("xmin").text,  # xmin
                bndbox.find("ymin").text,  # ymin
                bndbox.find("xmax").text,  # xmax
                bndbox.find("ymax").text   # ymax
            ]
        size = []
        for sizes in root.findall("size"):
            size = [
                sizes.find("width").text,
                sizes.find("height").text,
                sizes.find("depth").text
            ]

        category = root.find("name").text
        pose = root.find("pose").text
        data[file_name] = {
            "category": category,
            "bndbox": bndbox,
            "size": size,
            "pose": pose
        }
    return data
