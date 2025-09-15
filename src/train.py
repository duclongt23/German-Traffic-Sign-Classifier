import os, random, math
import numpy as np
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from dataset import GTSRBDataset, IMG_SIZE
import torch
import torch.nn as nn
from model import build_model
from pathlib import Path

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total, correct, loss_sum = 0, 0 , 0.0
    criterion = nn.CrossEntropyLoss()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        loss_sum  += loss.item() * x.size(0)
        pred= logits.argmax(1)
        correct += (pred == y).sum().item()
        total += x.size(0)
    return loss_sum / total, correct/ total

#root_dir = "/kaggle/input/gtsrb-german-traffic-sign"
root_dir = "./data/GTSRB"
train_csv = os.path.join(root_dir, "Train.csv")
test_csv = os.path.join(root_dir, "Test.csv")
img_size = IMG_SIZE
epochs = 10
batch_size = 64
lr = 3e-4
train_ds = GTSRBDataset(train_csv, root_dir, img_size, is_train= True)
val_ds = GTSRBDataset(test_csv, root_dir, img_size, is_train= False)

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)


device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = build_model(pretrained=True).to(device)

optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

scaler = torch.amp.GradScaler('cuda') if device=='cuda' else None
criterion = nn.CrossEntropyLoss()


MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, # create parent directories if needed
                 exist_ok=True # if models directory already exists, don't error
)
best_acc, best_ep = 0.0, -1
out = MODEL_PATH / 'best.pt'
backbone = 'resnet18'

for epoch in range(1, epochs+1):
    model.train()
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
    running_loss = 0.0
    for x, y in pbar:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        if scaler:
            with torch.amp.autocast('cuda'):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
        running_loss += loss.item() * x.size(0)
        pbar.set_postfix(loss=running_loss/((pbar.n+1)*x.size(0)))

    scheduler.step()
    val_loss, val_acc = evaluate(model, val_loader, device)
    print(f"[Val] loss={val_loss:.4f} acc={val_acc:.4f}")

    if val_acc > best_acc:
        best_acc, best_ep = val_acc, epoch
        torch.save({'model': model.state_dict(),
                    'backbone': backbone,
                    'img_size': img_size}, out)
        print(f"Saved best to {out} (acc={best_acc:.4f})")

print(f"Best Acc: {best_acc:.4f} at epoch {best_ep}")
