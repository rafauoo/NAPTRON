import torch
import torchvision.transforms.functional as TF
from PIL import Image

def transform_image(image):
    # Uproszczona transformacja
    return TF.to_tensor(image)

def convert_voc_annotation(voc_target):
    boxes = []
    labels = []

    objects = voc_target['annotation']['object']
    if not isinstance(objects, list):
        objects = [objects]

    for obj in objects:
        bbox = obj['bndbox']
        box = [
            float(bbox['xmin']),
            float(bbox['ymin']),
            float(bbox['xmax']),
            float(bbox['ymax'])
        ]
        label = obj['name']
        boxes.append(box)
        labels.append(label)

    return boxes, labels
