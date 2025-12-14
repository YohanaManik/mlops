import pytest
import pandas as pd
from pathlib import Path

def test_metadata_exists():
    """Test if metadata.csv exists after ingestion"""
    metadata_path = Path('data/processed/metadata.csv')
    assert metadata_path.exists(), "Metadata file not found!"

def test_metadata_columns():
    """Test metadata has required columns"""
    df = pd.read_csv('data/processed/metadata.csv')
    required_cols = ['image_path', 'class_name', 'class_id']
    
    for col in required_cols:
        assert col in df.columns, f"Missing column: {col}"

def test_label_map():
    """Test label map structure"""
    import json
    
    with open('data/processed/label_map.json') as f:
        label_map = json.load(f)
    
    assert len(label_map) == 5, "Should have 5 flower classes"
    assert 'rose' in label_map, "Missing 'rose' class"