import os
import random
import numpy as np
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

class CXRDataset(Dataset):
    def __init__(self, root_dir, classes, transform=None):

        print("Loading Data....\n")
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = []
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        for cls in classes:
            cls_path = self.root_dir / cls
            if not cls_path.exists():
                print(f"Warning: Directory not found: {cls_path}")
                continue
            for img_name in sorted(os.listdir(cls_path)):
                if img_name.startswith('.'):
                    continue
                self.samples.append((str(cls_path / img_name), self.class_to_idx[cls]))
        random.shuffle(self.samples)
        print(f"Found {len(self.samples)} images in {root_dir}") # Add this

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = np.array(Image.open(img_path).convert("RGB"))
        if self.transform:
            augmented = self.transform(image=img)
            img = augmented["image"]
        return img, label, img_path

def get_transforms():
    print("Transforming Data.....\n")
    train_aug = A.Compose([
        A.RandomResizedCrop(size=(224, 224), scale=(0.85, 1.0)),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.Affine(scale=(0.95, 1.05), translate_percent=(0.02, 0.02), rotate=(-10, 10), p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.CLAHE(clip_limit=2.0, p=0.3),
        A.CoarseDropout(num_holes_range=(3, 6), hole_height_range=(10, 20), hole_width_range=(10, 20), fill="random_uniform", p=0.4),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    val_aug = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    return train_aug, val_aug