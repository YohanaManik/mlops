import yaml
import pandas as pd
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split

def run_preprocess(config_path: Path):
    """
    Split data into train/val/test and copy to data/splits/
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Load metadata
    metadata = pd.read_csv("data/processed/metadata.csv")
    
    # Split
    train_ratio = config['split_ratio']['train']
    val_ratio = config['split_ratio']['val']
    test_ratio = config['split_ratio']['test']
    seed = config['seed']
    
    # Train+Val vs Test
    train_val, test = train_test_split(
        metadata, 
        test_size=test_ratio, 
        stratify=metadata['class_name'],
        random_state=seed
    )
    
    # Train vs Val
    val_size = val_ratio / (train_ratio + val_ratio)
    train, val = train_test_split(
        train_val,
        test_size=val_size,
        stratify=train_val['class_name'],
        random_state=seed
    )
    
    print(f"📊 Split sizes - Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
    
    # Copy files to splits folder
    splits_path = Path("data/splits")
    for split_name, split_df in [('train', train), ('val', val), ('test', test)]:
        split_folder = splits_path / split_name
        split_folder.mkdir(parents=True, exist_ok=True)
        
        for _, row in split_df.iterrows():
            src = Path(row['image_path'])
            cls_folder = split_folder / row['class_name']
            cls_folder.mkdir(exist_ok=True)
            dst = cls_folder / src.name
            
            if not dst.exists():
                shutil.copy(src, dst)
        
        # Save split metadata
        split_csv = splits_path / f"{split_name}.csv"
        split_df.to_csv(split_csv, index=False)
    
    print("✅ Data split and copied to data/splits/")