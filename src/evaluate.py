import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

def show_samples(loader, classes, n=6):
    
    """Visualizes a grid of sample images from a DataLoader."""
    print("Visualizes a grid of sample images from a DataLoader...\n")
    imgs = []
    labels = []
    for x, y, p in loader:
        imgs.append(x)
        labels += list(y.numpy())
        if len(labels) >= n:
            break
            
    imgs = torch.cat(imgs, dim=0)[:n]
    fig, axes = plt.subplots(1, n, figsize=(15, 4))
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    for i in range(n):
        img = imgs[i].permute(1, 2, 0).cpu().numpy()
        img = (img * std + mean) 
        img = np.clip(img, 0, 1)
        axes[i].imshow(img)
        axes[i].axis('off')
        axes[i].set_title(classes[labels[i]])
    plt.tight_layout()
    plt.show()

def evaluate_model(model, test_loader, device, classes):
    """Generates a Confusion Matrix and Classification Report."""
    model.eval()
    y_true = []
    y_pred = []

    print("Running evaluation...")
    with torch.no_grad():
        for imgs, labels, _ in test_loader:
            imgs = imgs.to(device)
            outputs = model(imgs) 
            preds = torch.argmax(outputs, dim=1)
            y_true.extend(labels.numpy())
            y_pred.extend(preds.cpu().numpy())


    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(cmap="Blues", xticks_rotation=45)
    plt.title("Confusion Matrix")
    plt.show()

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=classes))