import torch
import torch.nn as nn
import timm

def get_model(model_name='efficientnet_b0', num_classes=4, pretrained=True, device='cpu'):
    
    model = timm.create_model(model_name, pretrained=pretrained)

    in_features = model.get_classifier().in_features

    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, num_classes)
    )
    
    model = model.to(device)
    
    return model

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    test_model = get_model(num_classes=4, device=device)
    test_input = torch.randn(1, 3, 224, 224).to(device)
    output = test_model(test_input)
    print(f"Model created successfully. Output shape: {output.shape}") 