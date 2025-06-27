import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np

DCSS = [1, 3, 7, 12, 20, 30]  # Distinct Circular Subsum Sequences. Fixed for ERPA, as our SOTA permutation
DCSS_SIZE = 7  # Using SOTA permutation size as default, If your environment has higher error rate than 10%, we recommend to be lower than 7.

# Flip bits with given probability
def flip_bits(tensor: torch.Tensor, flip_prob: float) -> torch.Tensor:
    rand = torch.rand_like(tensor)
    flip_mask = rand < flip_prob
    flipped_tensor = torch.logical_xor(tensor.bool(), flip_mask)
    return flipped_tensor.float()

# ERPA Encoder
def encoder(L_batch: torch.Tensor, n: int = DCSS_SIZE) -> torch.Tensor:
    B, N = L_batch.shape
    L_enc = L_batch.clone()
    for j in range(n - 1):
        offset = DCSS[j]
        L_shifted = torch.roll(L_batch, shifts=offset, dims=1)
        L_enc = torch.logical_or(L_enc.bool(), L_shifted.bool())
    return L_enc.float()

# Dataset
class SparseBinaryDataset(Dataset):
    def __init__(self, num_samples=10000, N=64, sparsity=0.1):
        self.data = np.random.rand(num_samples, N) < sparsity
        self.data = self.data.astype(np.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.from_numpy(self.data[idx])

# Model Define
class BitAutoencoder(nn.Module):
    def __init__(self, N=64, M=64, flip_prob=0.1):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(M, N),
            nn.Sigmoid()
        )
        self.flip_prob = flip_prob

    def forward(self, x):
        z = encoder(x)
        z = flip_bits(z, self.flip_prob) # For Error environment. If you have yours, eval with your real data. But for training, bernoulli works as SOTA 
        x_hat = self.decoder(z) 
        return x_hat
