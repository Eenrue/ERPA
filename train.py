import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model.erpa import SparseBinaryDataset, BitAutoencoder
import json
from pathlib import Path
import sys

BATCH_SIZE = 128 # Change if needed
EPOCHS = 500 # Change if needed
LR = 1e-2 # Change if needed, but recommended LR
ENCODE_TYPE = "DCSS"

if len(sys.argv) != 2:
    print("Usage: python train.py [PROB]")
    sys.exit(1)

PROB=float(sys.argv[1]) # Prob of Error

######################### Changable Environment #########################
### Change the below If the environment is different from ours (N, M can be different)
N = 64         # Input dimension
M = 64         # Compressed dimension
###

### Change the below If the environment is different from ours (SPARSITY, FLIP_PROB can be different)
SPARSITY = PROB    # Prob of 1's in Input
FLIP_PROB = PROB    # Prob of Error in ERPA Encoder's output
##########################################################################

train_losses, test_losses = [], []

model = BitAutoencoder(N=N, M=M, flip_prob=FLIP_PROB).cuda()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

test_dataset = SparseBinaryDataset(num_samples=1000, N=N, sparsity=SPARSITY)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

def evaluate(model, loader):
    model.eval()
    total_loss = 0
    total_correct_bits = 0
    total_bits = 0
    with torch.no_grad():
        for x in loader:
            x = x.cuda()
            x_hat = model(x)
            loss = criterion(x_hat, x)
            total_loss += loss.item()
            correct_bits = (x == (x_hat > 0.5).float()).float().sum().item()
            total_correct_bits += correct_bits
            total_bits += x.numel()
    return total_loss / len(loader), total_correct_bits / total_bits

def train(model):
    model.train()
    total_loss = 0
    train_dataset = SparseBinaryDataset(num_samples=10000, N=N, sparsity=SPARSITY)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    for x in train_loader:
        x = x.cuda()
        x_hat = model(x)
        loss = criterion(x_hat, x)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(train_loader)

best_acc = 0.0
for epoch in range(EPOCHS):
    train_loss = train(model)
    test_loss, test_acc = evaluate(model, test_loader)
    train_losses.append(train_loss)
    test_losses.append(test_loss)
    print(f"[Epoch {epoch+1}] Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc*100:.2f}%")
    if test_acc > best_acc:
        best_acc = test_acc
        torch.save(model.state_dict(), f"weights/best_model_{SPARSITY}.pth")
        print(f"→ Best model saved with accuracy: {best_acc*100:.2f}%")

Path("results").mkdir(exist_ok=True)
json_path = f"results/result_encode_{ENCODE_TYPE}_DCSS_size_7_sparsity{SPARSITY}_flip{FLIP_PROB}.json"
with open(json_path, 'w') as f:
    json.dump({
        "permutation_type": ENCODE_TYPE,
        "DCSS_size": 7,
        "sparsity": SPARSITY,
        "flip_prob": FLIP_PROB,
        "final_test_loss": test_losses[-1],
        "final_test_accuracy": test_acc
    }, f, indent=2)
