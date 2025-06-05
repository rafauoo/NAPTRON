from torchvision.datasets import VOCDetection
from torch.utils.data import Dataset
from NAPTRON.utils import transform_image, convert_voc_annotation
import torchvision.transforms.functional as TF
import torch

VOC_CLASSES = {
    'aeroplane': 1, 'bicycle': 2, 'bird': 3, 'boat': 4, 'bottle': 5,
    'bus': 6, 'car': 7, 'cat': 8, 'chair': 9, 'cow': 10,
    'diningtable': 11, 'dog': 12, 'horse': 13, 'motorbike': 14, 'person': 15,
    'pottedplant': 16, 'sheep': 17, 'sofa': 18, 'train': 19, 'tvmonitor': 20
}

class VOCDatasetWrapper(Dataset):
    def __init__(self, root, year='2007', image_set='train'):
        self.dataset = VOCDetection(root=root, year=year, image_set=image_set, download=True)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, target = self.dataset[idx]
        image = transform_image(image)
        boxes, labels = convert_voc_annotation(target)
        labels = [VOC_CLASSES[l] for l in labels]

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)
        target_dict = {"boxes": boxes, "labels": labels}
        return image, target_dict
