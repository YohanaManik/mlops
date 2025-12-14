import pytest
import torch
from pathlib import Path
from flower.models.cnn import SimpleCNN

def test_smoke_training():
    """Smoke test: can we overfit on 10 samples?"""
    
    # Create dummy data
    x = torch.randn(10, 3, 224, 224)
    y = torch.randint(0, 5, (10,))
    
    model = SimpleCNN(num_classes=5)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Train for 10 epochs
    for epoch in range(10):
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
    
    # Check if loss decreased
    final_loss = loss.item()
    assert final_loss < 1.0, f"Model didn't learn! Loss: {final_loss}"