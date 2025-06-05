import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from NAPTRON.dataset import VOCDatasetWrapper

class NAPTRON:
    def __init__(self, model, layer_name='backbone.body.layer2'):
        self.model = model.eval()
        self.layer_name = layer_name
        self.nap_memory = {}  # {class_id: [NAPs]}
        self.activations = {}

        self._register_hook()

    def _register_hook(self):
        # Extract NAPs from chosen layer
        def hook_fn(module, input, output):
            self.activations['nap_layer'] = output.detach()

        layer = dict([*self.model.named_modules()])[self.layer_name]
        layer.register_forward_hook(hook_fn)

    def extract_naps(self, inputs):
        _ = self.model(inputs)  # forward pass to trigger hook
        act = self.activations['nap_layer']  # shape: [B, C, H, W]
        naps = F.adaptive_avg_pool2d(act, (1, 1))  # shape: [B, C, 1, 1]
        return naps.view(naps.size(0), -1)  # shape: [B, C]

    def add_to_memory(self, images, targets):
        naps = self.extract_naps(images)

        with torch.no_grad():
            preds = self.model(images)

        for i, pred in enumerate(preds):
            labels = pred['labels']
            boxes = pred['boxes']
            scores = pred['scores']

            for j in range(len(labels)):
                if scores[j] > 0.5:  # true positive approximation
                    c = labels[j].item()
                    nap = naps[i].clone().cpu().numpy()
                    if c not in self.nap_memory:
                        self.nap_memory[c] = []
                    self.nap_memory[c].append(nap)

    def hamming_distance(self, a, b):
        # Binarize and compare
        a_bin = (a > 0).astype(np.int32)
        b_bin = (b > 0).astype(np.int32)
        return np.sum(a_bin != b_bin)

    def compute_uncertainty(self, images):
        naps = self.extract_naps(images)
        preds = self.model(images)
        uncertainties = []

        for i, pred in enumerate(preds):
            labels = pred['labels']
            scores = pred['scores']
            for j, label in enumerate(labels):
                if scores[j] > 0.5:
                    c = label.item()
                    nap = naps[i].detach().cpu().numpy()
                    memory = self.nap_memory.get(c, [])
                    if not memory:
                        uncertainties.append(1.0)  # max uncertainty
                        continue
                    # NN search
                    distances = [self.hamming_distance(nap, m) for m in memory]
                    uncertainties.append(min(distances))
        return uncertainties
