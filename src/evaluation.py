import argparse, torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from dataset import GTSRBDataset
from model import build_model

@torch.no_grad()
def collect_preds(model, loader, device):
    model.eval()
    y_true, y_pred = [], []
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        pred = logits.argmax(1).cpu().numpy()
        y_pred.append(pred)
        y_true.append(y.numpy())
    return np.concatenate(y_true), np.concatenate(y_pred)

def plot_cm(cm, classes=43, save_path='confusion_matrix.png'):
    plt.figure(figsize=(10,10))
    plt.imshow(cm, interpolation='nearest')
    plt.title('Confusion Matrix'); plt.colorbar()
    ticks = np.arange(classes)
    plt.xticks(ticks, ticks, rotation=90); plt.yticks(ticks, ticks)
    plt.tight_layout()
    plt.xlabel('Predicted'); plt.ylabel('True')
    plt.savefig(save_path, dpi=200)
    print(f"Saved {save_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--val_csv', type=str, default='datasets//Train.csv')  # bạn có thể tạo file val riêng
    parser.add_argument('--root_dir', type=str, default='data/GTSRB')
    parser.add_argument('--weights', type=str, default='checkpoints/best.pt')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_classes', type=int, default=43)
    args = parser.parse_args()

    ckpt = torch.load(args.weights, map_location='cpu')
    backbone = ckpt.get('backbone', 'resnet18')
    img_size = ckpt.get('img_size', 224)

    ds = GTSRBDataset(args.val_csv, args.root_dir, img_size=img_size, is_train=False)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=2)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = build_model(num_classes=args.num_classes, backbone=backbone, pretrained=False).to(device)
    model.load_state_dict(ckpt['model'])

    y_true, y_pred = collect_preds(model, loader, device)
    print(classification_report(y_true, y_pred, digits=4))
    cm = confusion_matrix(y_true, y_pred)
    plot_cm(cm, classes=args.num_classes)

if __name__ == '__main__':
    main()
