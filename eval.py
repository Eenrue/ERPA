import torch
from torch.utils.data import DataLoader
from model.erpa import BitAutoencoder, SparseBinaryDataset
import sys

if len(sys.argv) != 3:
    print("Usage: python eval.py [PROB] [MODEL_PATH]")
    sys.exit(1)

SPARSITY = float(sys.argv[1])
MODEL_PATH = sys.argv[2]
N = 64
M = 64
BATCH_SIZE = 128

model = BitAutoencoder(N=N, M=M, flip_prob=SPARSITY).cuda()
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

test_dataset = SparseBinaryDataset(num_samples=1000, N=N, sparsity=SPARSITY)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

def evaluate(model, loader):
    criterion = torch.nn.BCELoss()
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

loss, acc = evaluate(model, test_loader)
print(f"[Evaluation] Loss: {loss:.4f}, Bit Accuracy: {acc*100:.2f}%")
