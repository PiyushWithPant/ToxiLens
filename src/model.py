

# ========================================= IMPORTS ========================================= 

import torch
import torch.nn as nn

# ========================================= CONFIG ========================================= 

INPUT_DIM = 10000


# ======================================= MODEL CLASS =========================================


class ToxicANN(nn.module):

    def __init__(self, INPUT_DIM):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(INPUT_DIM, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.binary_head = nn.Linear(128, 1)    # Output: Toxic or Not Toxic

        self.multi_head = nn.Linear(128, 6)        # Output: 6 Toxicity Categories

    def forward(self, x):
        shared_out = self.shared(x)

        binary_out = torch.sigmoid(self.binary_head(shared_out))
        multi_out = torch.sigmoid(self.multi_head(shared_out))

        return binary_out, multi_out

    # Since we have 2 heads, we need 2 losses for each head
    criterion_binary = nn.BCELoss()
    criterion_multi = nn.BCELoss()