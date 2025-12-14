import torch.nn as nn
from torchvision import models

def get_resnet(variant='resnet18', num_classes=5, pretrained=True):
    if variant == 'resnet18':
        model = models.resnet18(pretrained=pretrained)
    elif variant == 'resnet34':
        model = models.resnet34(pretrained=pretrained)
    else:
        raise ValueError(f"Unknown variant: {variant}")
    
    # Replace final layer
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model