import yaml
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from flower.models.cnn import SimpleCNN
from flower.models.resnet import get_resnet

def run_evaluate(checkpoint_path: Path):
    """
    Evaluate model on test set:
    - Accuracy, F1-score
    - Confusion matrix
    - Error analysis (save misclassified samples)
    """
    
    # Load label map
    with open('data/processed/label_map.json') as f:
        label_map = json.load(f)
    id2label = {v: k for k, v in label_map.items()}
    
    # Load model config from registry
    registry_path = Path('artifacts/registry/latest.json')
    if registry_path.exists():
        with open(registry_path) as f:
            registry = json.load(f)
        model_cfg_path = registry['model_config']
    else:
        # Fallback
        model_cfg_path = 'configs/model_cnn.yaml'
    
    with open(model_cfg_path) as f:
        model_cfg = yaml.safe_load(f)
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if model_cfg['model_type'] == 'cnn':
        model = SimpleCNN(**model_cfg['architecture'])
    else:
        model = get_resnet(**model_cfg['architecture'])
    
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    # Test dataset
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    test_dataset = datasets.ImageFolder('data/splits/test', transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Inference
    all_preds = []
    all_labels = []
    all_probs = []
    all_image_paths = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Metrics
    accuracy = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    f1_weighted = f1_score(all_labels, all_preds, average='weighted')
    
    print("\n" + "="*50)
    print("📊 EVALUATION RESULTS")
    print("="*50)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1-Score (Macro): {f1_macro:.4f}")
    print(f"F1-Score (Weighted): {f1_weighted:.4f}")
    print()
    
    # Classification report
    class_names = [id2label[i] for i in range(len(label_map))]
    report = classification_report(all_labels, all_preds, target_names=class_names, digits=4)
    print("Classification Report:")
    print(report)
    
    # Save report
    artifacts_path = Path('artifacts/evaluation')
    artifacts_path.mkdir(parents=True, exist_ok=True)
    
    with open(artifacts_path / 'classification_report.txt', 'w') as f:
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"F1-Score (Macro): {f1_macro:.4f}\n")
        f.write(f"F1-Score (Weighted): {f1_weighted:.4f}\n\n")
        f.write(report)
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(artifacts_path / 'confusion_matrix.png', dpi=150)
    print(f"✅ Confusion matrix saved: {artifacts_path / 'confusion_matrix.png'}")
    
    # Error analysis - find misclassified samples
    misclassified_idx = np.where(all_preds != all_labels)[0]
    print(f"\n❌ Total misclassified: {len(misclassified_idx)} / {len(all_labels)}")
    
    # Save error analysis
    error_data = []
    test_samples = test_dataset.samples
    
    for idx in misclassified_idx[:20]:  # Top 20 errors
        img_path, true_label = test_samples[idx]
        pred_label = all_preds[idx]
        confidence = all_probs[idx][pred_label]
        
        error_data.append({
            'image_path': img_path,
            'true_label': id2label[true_label],
            'predicted_label': id2label[pred_label],
            'confidence': f"{confidence:.4f}"
        })
    
    error_df = pd.DataFrame(error_data)
    error_df.to_csv(artifacts_path / 'error_analysis.csv', index=False)
    print(f"✅ Error analysis saved: {artifacts_path / 'error_analysis.csv'}")
    
    # Save metrics to JSON
    metrics = {
        'accuracy': float(accuracy),
        'f1_macro': float(f1_macro),
        'f1_weighted': float(f1_weighted),
        'total_samples': len(all_labels),
        'misclassified': int(len(misclassified_idx))
    }
    
    with open(artifacts_path / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print("\n✅ Evaluation complete!")
    return metrics