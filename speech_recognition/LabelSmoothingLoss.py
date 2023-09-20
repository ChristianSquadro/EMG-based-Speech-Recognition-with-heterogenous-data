from torch import nn
import torch
from absl import flags
import numpy as np
FLAGS = flags.FLAGS

class LabelSmoothingLoss(nn.Module):
    def __init__(self, epsilon=0.1, num_classes=2):
        super(LabelSmoothingLoss, self).__init__()
        self.epsilon = epsilon
        self.num_classes = num_classes

    def forward(self, input, target):
        result = ((1 - self.epsilon) * nn.CrossEntropyLoss(ignore_index=FLAGS.pad)(input, target) + (self.epsilon / input.shape[2]) * torch.sum(torch.exp(input)))
        return result