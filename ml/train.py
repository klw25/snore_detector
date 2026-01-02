import torch
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim

from dataset import SnoreDataset
from model import SnoreCNN

# ---------- CONFIG ----------
BATCH_SIZE = 16
EPOCHS = 10
LR = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- DATA ----------
dataset = SnoreDataset("data/clips")

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_set, val_set = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

# ---------- MODEL ----------
model = SnoreCNN()
model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# ---------- TRAINING LOOP ----------
for epoch in range(1, EPOCHS+1):
    model.train()
    running_loss = 0
    correct = 0
    total = 0

    for x, y in train_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * x.size(0)
        _, predicted = outputs.max(1)
        correct += (predicted == y).sum().item()
        total += y.size(0)

    train_loss = running_loss / total
    train_acc = correct / total

    # ---------- VALIDATION ----------
    model.eval()
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            outputs = model(x)
            _, predicted = outputs.max(1)
            val_correct += (predicted == y).sum().item()
            val_total += y.size(0)

    val_acc = val_correct / val_total

    print(f"Epoch {epoch}: Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")

# ---------- SAVE MODEL ----------
torch.save(model.state_dict(), "snore_cnn.pth")
print("Model saved as snore_cnn.pth")
