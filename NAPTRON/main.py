import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import ToTensor
from torchvision.datasets import VOCDetection
from torch.utils.data import DataLoader
from NAPTRON.naptron import NAPTRON
import torchvision.transforms.functional as TF

dataset = VOCDetection(root='./data', year='2007', image_set='train', download=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = fasterrcnn_resnet50_fpn(pretrained=True).to(device)
naptron = NAPTRON(model, layer_name='backbone.body.layer2')

# Dummy dataset (VOC)
def collate_fn(batch):
    return tuple(zip(*batch))

transform = lambda x: TF.to_tensor(x)

train_dataset = VOCDetection('./data', year='2007', image_set='train', download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True, collate_fn=collate_fn)

# Fill memory
for i, (imgs, targets) in enumerate(train_loader):
    imgs = [img.to(device) for img in imgs]
    naptron.add_to_memory(imgs, targets)
    if i >= 9:  # 10 batchy
        break

# Test
test_imgs, _ = next(iter(train_loader))
test_imgs = [img.to(device) for img in test_imgs]
uncertainties = naptron.compute_uncertainty(test_imgs)
print("Uncertainties:", uncertainties)
