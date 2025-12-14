import torch
from flower.models.cnn import SimpleCNN
from flower.models.resnet import get_resnet

def test_cnn_forward():
    """Test CNN forward pass"""
    model = SimpleCNN(num_classes=5)
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    
    assert out.shape == (2, 5), f"Expected (2, 5), got {out.shape}"

def test_resnet_forward():
    """Test ResNet forward pass"""
    model = get_resnet('resnet18', num_classes=5, pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    
    assert out.shape == (2, 5), f"Expected (2, 5), got {out.shape}"