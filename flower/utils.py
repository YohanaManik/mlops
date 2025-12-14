"""
Utility functions untuk flower classification pipeline
"""

import os
import random
import torch
import numpy as np
import yaml
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional


def set_seed(seed: int = 42):
    """
    Set random seed untuk reproducibility
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Untuk reproducibility yang lebih ketat (sedikit slower)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set PYTHONHASHSEED
    os.environ['PYTHONHASHSEED'] = str(seed)


def count_parameters(model: torch.nn.Module) -> int:
    """
    Hitung total trainable parameters dalam model
    
    Args:
        model: PyTorch model
        
    Returns:
        Total number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_device() -> torch.device:
    """
    Get available device (CUDA or CPU)
    
    Returns:
        torch.device
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"🚀 Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("💻 Using CPU")
    
    return device


def load_config(config_path: Path) -> Dict[str, Any]:
    """
    Load YAML config file
    
    Args:
        config_path: Path to YAML config file
        
    Returns:
        Dictionary containing config
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_config(config: Dict[str, Any], save_path: Path):
    """
    Save config to YAML file
    
    Args:
        config: Config dictionary
        save_path: Path to save YAML file
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)


def load_json(json_path: Path) -> Dict[str, Any]:
    """Load JSON file"""
    with open(json_path, 'r') as f:
        return json.load(f)


def save_json(data: Dict[str, Any], save_path: Path):
    """Save data to JSON file"""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(data, f, indent=2)


def setup_logging(
    log_file: Optional[Path] = None,
    level: int = logging.INFO
) -> logging.Logger:
    """
    Setup logging configuration
    
    Args:
        log_file: Optional path to log file
        level: Logging level
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger('flower')
    logger.setLevel(level)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(console_format)
        logger.addHandler(file_handler)
    
    return logger


def format_time(seconds: float) -> str:
    """
    Format seconds into human-readable time
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        mins = int(seconds // 60)
        secs = seconds % 60
        return f"{mins}m {secs:.0f}s"
    else:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        return f"{hours}h {mins}m"


def create_experiment_folder(base_path: Path, experiment_name: str = None) -> Path:
    """
    Create timestamped experiment folder
    
    Args:
        base_path: Base path for experiments
        experiment_name: Optional experiment name
        
    Returns:
        Path to experiment folder
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if experiment_name:
        folder_name = f"{timestamp}_{experiment_name}"
    else:
        folder_name = timestamp
    
    experiment_path = base_path / folder_name
    experiment_path.mkdir(parents=True, exist_ok=True)
    
    return experiment_path


class AverageMeter:
    """
    Computes and stores the average and current value
    Useful untuk tracking metrics during training
    """
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class EarlyStopping:
    """
    Early stopping untuk mencegah overfitting
    """
    def __init__(self, patience: int = 7, min_delta: float = 0.0, mode: str = 'max'):
        """
        Args:
            patience: Berapa epoch menunggu sebelum stop
            min_delta: Minimum perubahan untuk dianggap improvement
            mode: 'max' untuk accuracy, 'min' untuk loss
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score: float) -> bool:
        """
        Args:
            score: Current metric score
            
        Returns:
            True if should stop training
        """
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        
        return False


def print_training_info(config: Dict[str, Any], model: torch.nn.Module):
    """
    Print training configuration info
    
    Args:
        config: Training config
        model: PyTorch model
    """
    print("\n" + "="*60)
    print("🚀 TRAINING CONFIGURATION")
    print("="*60)
    print(f"Model: {config.get('model_type', 'Unknown')}")
    print(f"Total Parameters: {count_parameters(model):,}")
    print(f"Batch Size: {config.get('batch_size', 'N/A')}")
    print(f"Learning Rate: {config.get('learning_rate', 'N/A')}")
    print(f"Epochs: {config.get('epochs', 'N/A')}")
    print(f"Optimizer: {config.get('optimizer', 'N/A')}")
    print(f"Device: {get_device()}")
    print("="*60 + "\n")


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    save_path: Path
):
    """
    Save training checkpoint
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        epoch: Current epoch
        metrics: Dictionary of metrics
        save_path: Path to save checkpoint
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        'timestamp': datetime.now().isoformat()
    }
    
    torch.save(checkpoint, save_path)


def load_checkpoint(
    checkpoint_path: Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None
) -> Dict[str, Any]:
    """
    Load training checkpoint
    
    Args:
        checkpoint_path: Path to checkpoint
        model: PyTorch model
        optimizer: Optional optimizer
        
    Returns:
        Dictionary containing checkpoint info
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint


def validate_config(config: Dict[str, Any], required_keys: list) -> bool:
    """
    Validate if config contains all required keys
    
    Args:
        config: Config dictionary
        required_keys: List of required keys
        
    Returns:
        True if valid
        
    Raises:
        ValueError if missing keys
    """
    missing_keys = [key for key in required_keys if key not in config]
    
    if missing_keys:
        raise ValueError(f"Missing required config keys: {missing_keys}")
    
    return True


def get_timestamp() -> str:
    """Get current timestamp string"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def get_model_size_mb(model: torch.nn.Module) -> float:
    """
    Calculate model size in MB
    
    Args:
        model: PyTorch model
        
    Returns:
        Model size in MB
    """
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    
    size_mb = (param_size + buffer_size) / (1024 ** 2)
    return size_mb