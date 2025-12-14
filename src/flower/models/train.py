import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pathlib import Path
from flower.models.cnn import SimpleCNN
from flower.models.resnet import get_resnet

def run_train(model_cfg_path: Path, train_cfg_path: Path):
    """Training loop"""
    # Load configs
    with open(model_cfg_path) as f:
        model_cfg = yaml.safe_load(f)
    with open(train_cfg_path) as f:
        train_cfg = yaml.safe_load(f)
    
    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Datasets
    train_dataset = datasets.ImageFolder('data/splits/train', transform=transform)
    val_dataset = datasets.ImageFolder('data/splits/val', transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=train_cfg['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=train_cfg['batch_size'])
    
    # Model
    if model_cfg['model_type'] == 'cnn':
        model = SimpleCNN(**model_cfg['architecture'])
    else:
        model = get_resnet(**model_cfg['architecture'])
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Loss & Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg['learning_rate'])
    
    # Training loop
    best_val_acc = 0.0
    for epoch in range(train_cfg['epochs']):
        # Train
        model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_acc = correct / total
        print(f"Epoch {epoch+1}/{train_cfg['epochs']} - Loss: {train_loss/len(train_loader):.4f}, Val Acc: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'artifacts/registry/best_model.pt')
            print(f"✅ Best model saved! (Val Acc: {val_acc:.4f})")