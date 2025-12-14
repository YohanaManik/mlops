import io
import json
import time
import torch
import yaml
from pathlib import Path
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from torchvision import transforms

from flower.models.cnn import SimpleCNN
from flower.models.resnet import get_resnet
from api.schema import PredictionResponse, HealthResponse

app = FastAPI(
    title="Flower Classification API",
    description="MLOps Pipeline for 5 Flower Types Classification",
    version="1.0.0"
)

# Global variables for model
MODEL = None
DEVICE = None
LABEL_MAP = None
ID2LABEL = None
TRANSFORM = None

@app.on_event("startup")
def load_model():
    """Load model on startup"""
    global MODEL, DEVICE, LABEL_MAP, ID2LABEL, TRANSFORM
    
    print("🚀 Loading model...")
    
    # Load registry
    registry_path = Path('artifacts/registry/latest.json')
    if not registry_path.exists():
        raise RuntimeError("No model registered! Run 'flower register' first.")
    
    with open(registry_path) as f:
        registry = json.load(f)
    
    # Load label map
    with open(registry['label_map']) as f:
        LABEL_MAP = json.load(f)
    ID2LABEL = {v: k for k, v in LABEL_MAP.items()}
    
    # Load model config
    with open(registry['model_config']) as f:
        model_cfg = yaml.safe_load(f)
    
    # Initialize model
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if model_cfg['model_type'] == 'cnn':
        MODEL = SimpleCNN(**model_cfg['architecture'])
    else:
        MODEL = get_resnet(**model_cfg['architecture'])
    
    MODEL.load_state_dict(torch.load(registry['model_path'], map_location=DEVICE))
    MODEL = MODEL.to(DEVICE)
    MODEL.eval()
    
    # Define transform
    TRANSFORM = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    print(f"✅ Model loaded from: {registry['model_path']}")
    print(f"📊 Model metrics: {registry['metrics']}")


@app.get("/", response_model=HealthResponse)
def root():
    """Root endpoint"""
    return {
        "status": "ok",
        "message": "Flower Classification API is running!"
    }


@app.get("/health", response_model=HealthResponse)
def health_check():
    """Health check endpoint"""
    model_loaded = MODEL is not None
    
    return {
        "status": "ok" if model_loaded else "error",
        "message": "Model loaded" if model_loaded else "Model not loaded"
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """
    Predict flower class from uploaded image
    
    Args:
        file: Image file (jpg, png, jpeg)
    
    Returns:
        Prediction with class name, confidence, and all probabilities
    """
    
    # Validate file type
    if file.content_type not in ["image/jpeg", "image/jpg", "image/png"]:
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Only JPEG and PNG are supported."
        )
    
    try:
        # Start timer
        start_time = time.time()
        
        # Read and preprocess image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        image_tensor = TRANSFORM(image).unsqueeze(0).to(DEVICE)
        
        # Inference
        with torch.no_grad():
            outputs = MODEL(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]
            confidence, predicted_idx = torch.max(probabilities, 0)
        
        # Prepare response
        predicted_class = ID2LABEL[predicted_idx.item()]
        confidence_score = confidence.item()
        
        all_probabilities = {
            ID2LABEL[i]: float(probabilities[i])
            for i in range(len(probabilities))
        }
        
        # Calculate latency
        latency = time.time() - start_time
        
        # Log prediction (simple stdout logging)
        print(f"✅ Prediction: {predicted_class} (confidence: {confidence_score:.4f}, latency: {latency:.3f}s)")
        
        return {
            "success": True,
            "predicted_class": predicted_class,
            "confidence": confidence_score,
            "all_probabilities": all_probabilities,
            "latency_ms": round(latency * 1000, 2)
        }
    
    except Exception as e:
        print(f"❌ Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/classes")
def get_classes():
    """Get all available flower classes"""
    return {
        "classes": list(LABEL_MAP.keys()),
        "num_classes": len(LABEL_MAP)
    }