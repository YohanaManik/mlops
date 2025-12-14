import yaml
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List

def run_ingest(config_path: Path):
    """
    Scan data/raw folder, create:
    - data/processed/metadata.csv (columns: image_path, class_name, class_id)
    - data/processed/label_map.json
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    raw_path = Path(config['raw_data_path'])
    processed_path = Path(config['processed_path'])
    processed_path.mkdir(parents=True, exist_ok=True)
    
    classes = config['classes']
    label_map = {cls: idx for idx, cls in enumerate(classes)}
    
    # Scan images
    data = []
    for cls_name in classes:
        cls_folder = raw_path / cls_name
        if not cls_folder.exists():
            print(f"⚠️  Warning: {cls_folder} not found, skipping...")
            continue
        
        for img_path in cls_folder.glob("*.jpg"):
            data.append({
                'image_path': str(img_path),
                'class_name': cls_name,
                'class_id': label_map[cls_name]
            })
    
    # Save metadata
    df = pd.DataFrame(data)
    metadata_path = processed_path / "metadata.csv"
    df.to_csv(metadata_path, index=False)
    print(f"📄 Metadata saved: {metadata_path} ({len(df)} images)")
    
    # Save label map
    label_map_path = processed_path / "label_map.json"
    with open(label_map_path, 'w') as f:
        json.dump(label_map, f, indent=2)
    print(f"🏷️  Label map saved: {label_map_path}")
    
    # Print class distribution
    print("\n📊 Class Distribution:")
    print(df['class_name'].value_counts())