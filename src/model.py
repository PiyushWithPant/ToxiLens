# ------------------------------------------------------------------------------------
#                               model.py - ANN Model
# ------------------------------------------------------------------------------------

# ========================================= IMPORTS ========================================= 

import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader



# ========================================= CONFIG ========================================= 

INPUT_DIM = 5000
BATCH_SIZE = 64
EPOCHS = 10
LR = 1e-4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======================================= MODEL CLASS =========================================


class ToxicANN(nn.Module):

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


# ========================================= DATASET CLASS =========================================


class SparseDataset(Dataset):
    def __init__(self, X_sparse, y_bin, y_multi):
        self.X_sparse = X_sparse
        self.y_bin = torch.tensor(y_bin.values, dtype=torch.float32)
        self.y_multi = torch.tensor(y_multi.values, dtype=torch.float32)

    def __len__(self):
        return self.X_sparse.shape[0]

    def __getitem__(self, idx):
        x = torch.tensor(self.X_sparse[idx].toarray(), dtype=torch.float32).squeeze(0)
        y_bin = self.y_bin[idx]
        y_multi = self.y_multi[idx]
        return x, y_bin, y_multi

















