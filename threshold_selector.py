import math
import torch
import numpy as np


class DynamicThresholdEMA:
    def __init__(self, num_classes, max_iterations,
                 ema_init=0.5, ema_final=0.99):
        self.num_classes = num_classes
        self.max_iterations = max_iterations
        self.ema_init = ema_init
        self.ema_final = ema_final
        self.thresholds = torch.tensor([float('inf')] * num_classes)  # Init values: inf

    def update_ema_weight(self, step):
        """Adjust the weight of EMA dynamically"""
        alpha = step / self.max_iterations
        return self.ema_init + (self.ema_final - self.ema_init) * alpha

    def update_thresholds(self, step, max_probs, pseudo_labels):
        """Adjust the weight of threshold for each class dynamically"""

        current_conf = [[] for _ in range(self.num_classes)]

        with torch.no_grad():
            ratio = max((step / self.max_iterations) ** 0.5, 0.2)
            k = int(len(max_probs) * ratio)
            _, topk_indices = torch.topk(max_probs, k=k)
            topk_probs = max_probs[topk_indices]
            topk_labels = pseudo_labels[topk_indices]

            for c in range(self.num_classes):
                mask = (topk_labels == c)
                if mask.any():
                    current_conf[c].append(topk_probs[mask])

        # update EMA
        ema_weight = self.update_ema_weight(step)
        # print(ema_weight)
        for c in range(self.num_classes):
            if len(current_conf[c]) > 0:
                current_mean = torch.cat(current_conf[c]).mean()
                if math.isinf(self.thresholds[c]):
                    self.thresholds[c] = current_mean
                else:
                    self.thresholds[c] = (1 - ema_weight) * current_mean + ema_weight * self.thresholds[c]

    def get_threshold(self, class_idx):
        return self.thresholds[class_idx].item()
