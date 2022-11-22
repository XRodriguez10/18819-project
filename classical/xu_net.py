"""
Simple neural network model example.
"""

import torch.nn as nn


class Net(nn.Module):

    def __init__(self):
        super().__init__()

        self.linear_relu_stack = nn.Sequential(
            nn.Linear(784, 20),
            nn.ReLU(),
            nn.Linear(20, 20),
            nn.ReLU(),
            nn.Linear(20, 10),
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits
