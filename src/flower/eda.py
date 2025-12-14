import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def run_eda(metadata_path: Path):
    """Generate EDA plots"""
    df = pd.read_csv(metadata_path)
    
    # Class distribution
    plt.figure(figsize=(10, 5))
    sns.countplot(data=df, x='class_name', order=df['class_name'].value_counts().index)
    plt.title("Class Distribution")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("artifacts/eda_class_distribution.png")
    print("📊 Saved: artifacts/eda_class_distribution.png")
    
    # Show sample images (optional)
    print("\n✅ EDA complete!")