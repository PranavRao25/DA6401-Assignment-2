"""
TO MERGE ALL DATA INTO ONE
"""

import os
from pathlib import Path
import xml
import json
import xml.etree.ElementTree as ET

def parse_voc_xml(xml_path):
    """
    Parses a PASCAL VOC XML annotation file and returns a structured dictionary.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)
    depth = int(size.find('depth').text) # Channels (e.g., 3 for RGB)

    segmented = int(root.find('segmented').text)

    # Extract bounding boxes and labels
    obj = root.findall('object')[0]
    name = obj.find('name').text
    bndbox = obj.find('bndbox')
    pose = obj.find('pose').text
    truncated = obj.find('truncated').text
    occluded = obj.find('occluded').text
    difficult = obj.find('difficult').text
    
    # Extract spatial coordinates
    xmin = int(bndbox.find('xmin').text)
    ymin = int(bndbox.find('ymin').text)
    xmax = int(bndbox.find('xmax').text)
    ymax = int(bndbox.find('ymax').text)

    return {
        'dimensions': (width, height, depth),
        "segmented": segmented,
        'name': name,
        'bbox': [xmin, ymin, xmax, ymax],
        "pose": pose,
        "truncated": truncated,
        "occluded": occluded,
        "difficult": difficult
    }

parent = Path(__file__).resolve().parent.parent

images_folder = parent / "data" / "images" / "raw"
annotations_folder = parent / "data" / "annotations" / "raw"

list_data = annotations_folder / "list.txt"
with open(list_data, "r") as f:
    data = f.readlines()
data = data[6:]
mapping = dict()
for i in data:
    image_name, label, species, breed_id= i.split(" ")
    mapping[image_name] = {
        "label": int(label),
        "species": int(species),
        "breed": int(breed_id[:-1])
    }

mapping_path = annotations_folder / "mapping.json"
# with open(mapping_path, "r") as f:
#     mapping = json.load(f)

for key in mapping:
    trimap_file_path = "data/" + "annotations/" + "raw/" + "trimaps/" + f"._{key}.png"
    mapping[key]["trimap"] = trimap_file_path

    try:
        xml_file_path = "data/" + "annotations/" + "raw/" + "xmls/" + f"{key}.xml"
        mapping[key]["xml"] = parse_voc_xml(parent / xml_file_path)
    except FileNotFoundError:
        mapping[key]["xml"] = {}

# with open(mapping_path, "w") as f:
#     json.dump(mapping, f)

# mapping_path = parent / "data" / "annotations" / "processed" / "mapping.json"
# with open(mapping_path, "r") as f:
#     mapping = json.load(f)

for key in mapping:
    try:
        image_file_path = "data/" + "images/" + "raw/" + f"{key}.jpg"
        mapping[key]["image"] = image_file_path
    except FileNotFoundError:
        mapping[key]["image"] = ""

with open(parent / "data" / "data.json", "w") as f:
    json.dump(mapping, f)
