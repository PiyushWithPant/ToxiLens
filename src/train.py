# ------------------------------------------------------------------------------------
#                               train.py - Model Training
# ------------------------------------------------------------------------------------

# ========================================= IMPORTS ========================================= 

import torch
import joblib
import numpy as np
import pandas as pd
import torch.nn as nn
from sklearn.metrics import f1_score, accuracy_score
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from model import ToxicANN
from model import SparseDataset

# ========================================= CONFIG ========================================= 

INPUT_DIM = 5000
BATCH_SIZE = 64
EPOCHS = 10
LR = 1e-4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========================================= LOAD PROCESSED DATA ========================================= 

# Load your saved TF-IDF and label files
X_tfidf = joblib.load("data/preprocessed data/X_tfidf_sparse.pkl")      # sparse matrix
y_bin = pd.read_csv("data/preprocessed data/y_binary.csv")
y_multi = pd.read_csv("data/preprocessed data/y_multi.csv")


print("Data loaded (TF-IDF, Y_binary, Y_multi):", X_tfidf.shape, y_bin.shape, y_multi.shape)

 
# Split training data: 80% train, 20% validation
X_train, X_val, y_bin_train, y_bin_val, y_multi_train, y_multi_val = train_test_split(
    X_tfidf, 
    y_bin, 
    y_multi, 
    test_size=0.1, 
    random_state=100
)

print("Train size:", X_train.shape[0])
print("Validation size:", X_val.shape[0])




# ========================================= DATALOADER =========================================

# Training dataset
train_dataset = SparseDataset(X_train, y_bin_train, y_multi_train)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Validation dataset
val_dataset = SparseDataset(X_val, y_bin_val, y_multi_val)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

print("DataLoaders created successfully!")


# ========================================= MODEL  =========================================

model = ToxicANN(INPUT_DIM)
model.to(device)
print("Model initialized successfully!")


# ========================================= LOSS & OPTIMIZER =========================================

criterion_binary = nn.BCELoss()
criterion_multi = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)


# ========================================= TRAINING LOOP =========================================

for epoch in range(EPOCHS):

    # ---------------------- TRAINING ----------------------
    model.train()
    running_loss = 0.0
    print("\n")

    for x_batch, y_bin_batch, y_multi_batch in train_loader:
        x_batch, y_bin_batch, y_multi_batch = (
            x_batch.to(device),
            y_bin_batch.to(device),
            y_multi_batch.to(device)
        )

        optimizer.zero_grad()
        pred_bin, pred_multi = model(x_batch)

        loss_bin = criterion_binary(pred_bin, y_bin_batch)
        loss_multi = criterion_multi(pred_multi, y_multi_batch)
        loss = loss_bin + loss_multi

        loss.backward()
        optimizer.step()
        running_loss += loss.item() * x_batch.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    print(f"🟢 Epoch [{epoch+1}/{EPOCHS}] Training Loss: {epoch_loss:.4f}")

    # ---------------------- VALIDATION ----------------------
    model.eval()
    val_loss = 0.0

    all_bin_preds, all_bin_true = [], []
    all_multi_preds, all_multi_true = [], []

    with torch.no_grad():
        for x_batch, y_bin_batch, y_multi_batch in val_loader:
            x_batch = x_batch.to(device)
            y_bin_batch = y_bin_batch.to(device)
            y_multi_batch = y_multi_batch.to(device)

            pred_bin, pred_multi = model(x_batch)

            # Compute validation loss
            loss_bin = criterion_binary(pred_bin, y_bin_batch)
            loss_multi = criterion_multi(pred_multi, y_multi_batch)
            val_loss += (loss_bin + loss_multi).item() * x_batch.size(0)

            # Store preds + true labels
            all_bin_preds.append(pred_bin.cpu())
            all_multi_preds.append(pred_multi.cpu())
            all_bin_true.append(y_bin_batch.cpu())
            all_multi_true.append(y_multi_batch.cpu())

    # Concatenate all batches
    all_bin_preds = torch.cat(all_bin_preds).numpy()
    all_multi_preds = torch.cat(all_multi_preds).numpy()
    all_bin_true = torch.cat(all_bin_true).numpy()
    all_multi_true = torch.cat(all_multi_true).numpy()

    # Threshold outputs
    bin_labels = (all_bin_preds > 0.5).astype(int)
    multi_labels = (all_multi_preds > 0.5).astype(int)

    # Metrics
    binary_acc = accuracy_score(all_bin_true, bin_labels)
    multi_f1 = f1_score(all_multi_true, multi_labels, average='macro')

    avg_val_loss = val_loss / len(val_loader.dataset)

    print(f"🔵 Epoch [{epoch+1}/{EPOCHS}] Validation Loss: {avg_val_loss:.4f}")
    print(f"✅ Binary Accuracy: {binary_acc:.2f}")
    print(f"🎯 Multi-label F1 Score (macro): {multi_f1:.2f}")
    
    print("\n")
    print("*"*50)


# ========================================= SAVE MODEL =========================================

torch.save(model.state_dict(), "model/toxilens.pth")
print("Model saved successfully!")
