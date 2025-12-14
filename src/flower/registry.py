import json
import shutil
from pathlib import Path
from datetime import datetime
import yaml

def register_model(
    model_path: Path,
    model_config: Path,
    train_config: Path,
    metrics: dict,
    experiment_name: str = None
):
    """
    Register best model to artifacts/registry/
    
    Structure:
    - artifacts/registry/latest.json (pointer to best model)
    - artifacts/registry/{timestamp}/ (versioned artifacts)
    """
    
    registry_path = Path('artifacts/registry')
    registry_path.mkdir(parents=True, exist_ok=True)
    
    # Create version folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    version_folder = registry_path / timestamp
    version_folder.mkdir(exist_ok=True)
    
    # Copy artifacts
    shutil.copy(model_path, version_folder / 'model.pt')
    shutil.copy(model_config, version_folder / 'model_config.yaml')
    shutil.copy(train_config, version_folder / 'train_config.yaml')
    shutil.copy('data/processed/label_map.json', version_folder / 'label_map.json')
    
    # Save metrics
    with open(version_folder / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Update latest.json
    latest_registry = {
        'version': timestamp,
        'experiment_name': experiment_name or 'default',
        'model_path': str(version_folder / 'model.pt'),
        'model_config': str(version_folder / 'model_config.yaml'),
        'train_config': str(version_folder / 'train_config.yaml'),
        'label_map': str(version_folder / 'label_map.json'),
        'metrics': metrics,
        'registered_at': datetime.now().isoformat()
    }
    
    with open(registry_path / 'latest.json', 'w') as f:
        json.dump(latest_registry, f, indent=2)
    
    print(f"✅ Model registered: {version_folder}")
    print(f"📊 Metrics: {metrics}")
    
    return latest_registry


def load_latest_model():
    """Load latest registered model"""
    registry_path = Path('artifacts/registry/latest.json')
    
    if not registry_path.exists():
        raise FileNotFoundError("No model registered yet!")
    
    with open(registry_path) as f:
        registry = json.load(f)
    
    return registry


def list_models():
    """List all registered models"""
    registry_path = Path('artifacts/registry')
    
    if not registry_path.exists():
        print("No models registered yet!")
        return []
    
    versions = [d for d in registry_path.iterdir() if d.is_dir()]
    
    print(f"\n📦 Registered Models ({len(versions)}):")
    print("="*60)
    
    for version in sorted(versions, reverse=True):
        metrics_path = version / 'metrics.json'
        if metrics_path.exists():
            with open(metrics_path) as f:
                metrics = json.load(f)
            print(f"Version: {version.name}")
            print(f"  - Accuracy: {metrics.get('accuracy', 'N/A'):.4f}")
            print(f"  - F1-Macro: {metrics.get('f1_macro', 'N/A'):.4f}")
            print()
    
    return versions