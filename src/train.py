import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from pathlib import Path
from data_utils import CXRDataset, get_transforms
from model import get_model
from evaluate import evaluate_model, show_samples

SEED = 42
BATCH_SIZE = 32
NUM_WORKERS = 2
CLASSES = ['COVID19', 'NORMAL', 'PNEUMONIA', 'TURBERCULOSIS']

BASE_DIR = Path("./data") 

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for imgs, labels, _ in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * imgs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
    return running_loss / total, 100. * correct / total

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(SEED)


    train_aug, val_aug = get_transforms()
    train_ds = CXRDataset(BASE_DIR / "train", CLASSES, transform=train_aug)
    test_ds = CXRDataset(BASE_DIR / "test", CLASSES, transform=val_aug)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)


    show_samples(train_loader, CLASSES)

    print("Loading Model EfficientNet B0\n")
    model = get_model(model_name='efficientnet_b0', num_classes=len(CLASSES), device=device)
    optimizer = AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

    print("Starting Training")
    num_epochs = 10 
    for epoch in range(num_epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        print(f"Epoch {epoch+1}/{num_epochs} | Loss: {train_loss:.4f} | Acc: {train_acc:.2f}%")



    evaluate_model(model, test_loader, device, CLASSES)
    

    torch.save(model.state_dict(), "../Model/lung_disease_model.pth")
    print("Model saved to lung_disease_model.pth")

if __name__ == "__main__":
    main()