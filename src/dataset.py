import os
import cv2
import pandas as pd
from torch.utils.data import Dataset
from albumentations import (
    Compose, RandomBrightnessContrast, HueSaturationValue,
    ShiftScaleRotate, HorizontalFlip, Normalize, Resize
)
from albumentations.pytorch import ToTensorV2
IMG_SIZE = 224

class GTSRBDataset(Dataset):
    def __init__(self, csv_file, root_dir, img_size = IMG_SIZE, is_train = True):
        self.df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.img_size = img_size
        self.is_train = is_train

        if is_train:
            self.tf = Compose([
                Resize(img_size, img_size),
                ShiftScaleRotate(shift_limit=0.06, scale_limit= 0.1, rotate_limit =15, p = 0.6),
                RandomBrightnessContrast(p=0.5),
                HueSaturationValue(p=0.4),
                HorizontalFlip(p=0.2),
                Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
                ToTensorV2()
            ])
        else:
            self.tf = Compose([
                Resize(img_size, img_size),
                Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
                ToTensorV2()
            ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(root_dir, row["Path"])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        label = row["ClassId"]
        img = self.tf(image = img)['image']
        return img, label