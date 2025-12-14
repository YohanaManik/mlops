"""
Create dummy model and registry for API testing in CI
"""

import json
import torch
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from flower.models import SimpleCNN

def create_dummy_model():
    """Create dummy model for CI testing"""
    
    # Create directories
    registry_path = Path('artifacts/registry')
    registry_path.mkdir(parents=True, exist_ok=True)
    
    # Create simple model
    model = SimpleCNN(num_classes=5, base_filters=16, dropout=0.3)
    
    # Save model
    model_path = registry_path / 'dummy_model.pt'
    torch.save(model.state_dict(), model_path)
    
    # Create dummy label map
    label_map = {
        'Lilly': 0,
        'Lotus': 1,
        'Orchid': 2,
        'Sunflower': 3,
        'Tulip': 4
    }
    
    label_map_path = registry_path / 'label_map.json'
    with open(label_map_path, 'w') as f:
        json.dump(label_map, f)
    
    # Create dummy model config
    model_config = {
        'model_type': 'cnn',
        'architecture': {
            'num_classes': 5,
            'base_filters': 16,
            'dropout': 0.3
        }
    }
    
    import yaml
    model_config_path = registry_path / 'model_config.yaml'
    with open(model_config_path, 'w') as f:
        yaml.dump(model_config, f)
    
    # Create registry latest.json
    registry_data = {
        'version': 'dummy',
        'experiment_name': 'ci_test',
        'model_path': str(model_path),
        'model_config': str(model_config_path),
        'label_map': str(label_map_path),
        'metrics': {
            'accuracy': 0.85,
            'f1_macro': 0.83
        }
    }
    
    with open(registry_path / 'latest.json', 'w') as f:
        json.dump(registry_data, f, indent=2)
    
    print("✅ Dummy model created for CI testing")

if __name__ == '__main__':
    create_dummy_model()