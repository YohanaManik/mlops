"""
Unit tests for utility functions
"""

import pytest
import torch
from pathlib import Path
from flower.utils import (
    set_seed,
    count_parameters,
    get_device,
    AverageMeter,
    EarlyStopping,
    format_time
)
from flower.models import SimpleCNN


def test_set_seed():
    """Test seed setting"""
    set_seed(42)
    
    # Generate random numbers
    rand1 = torch.rand(10)
    
    # Reset seed
    set_seed(42)
    rand2 = torch.rand(10)
    
    # Should be identical
    assert torch.allclose(rand1, rand2), "Seed not working properly"


def test_count_parameters():
    """Test parameter counting"""
    model = SimpleCNN(num_classes=5, base_filters=32)
    num_params = count_parameters(model)
    
    assert num_params > 0, "No parameters counted"
    assert isinstance(num_params, int), "Parameter count should be integer"


def test_get_device():
    """Test device detection"""
    device = get_device()
    assert isinstance(device, torch.device)


def test_average_meter():
    """Test AverageMeter class"""
    meter = AverageMeter()
    
    meter.update(10, n=1)
    assert meter.avg == 10
    
    meter.update(20, n=1)
    assert meter.avg == 15
    
    meter.reset()
    assert meter.avg == 0


def test_early_stopping():
    """Test EarlyStopping"""
    early_stop = EarlyStopping(patience=3, mode='max')
    
    # Improving scores
    assert not early_stop(0.7)
    assert not early_stop(0.8)
    assert not early_stop(0.9)
    
    # No improvement
    assert not early_stop(0.9)
    assert not early_stop(0.85)
    assert not early_stop(0.85)
    
    # Should trigger stop
    assert early_stop(0.85)


def test_format_time():
    """Test time formatting"""
    assert format_time(45) == "45.00s"
    assert "1m" in format_time(90)
    assert "1h" in format_time(3700)