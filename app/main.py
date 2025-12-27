"""
FastAPI application for EuroSAT image classification.
"""
import os
import sys
import traceback
import pickle
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import torch
from PIL import Image
import io
from typing import Literal

from app.utils import preprocess_image, get_top_predictions
# Import model classes so they're available when loading saved models
from app.models import SEBlock

# Make SEBlock available in common module namespaces for unpickling
# This is needed because models saved from notebooks/scripts might reference these modules
import types
if '__main__' not in sys.modules:
    sys.modules['__main__'] = types.ModuleType('__main__')
sys.modules['__main__'].SEBlock = SEBlock

# Create __mp_main__ module if it doesn't exist (used by multiprocessing)
if '__mp_main__' not in sys.modules:
    sys.modules['__mp_main__'] = types.ModuleType('__mp_main__')
sys.modules['__mp_main__'].SEBlock = SEBlock

# Register SEBlock with torch's safe globals system (PyTorch 2.6+)
try:
    torch.serialization.add_safe_globals([SEBlock])
except Exception as e:
    print(f"Note: Could not register SEBlock as safe global: {e}")

# EuroSAT class labels
CLASSES = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake"
]

# Initialize FastAPI app
app = FastAPI(title="EuroSAT Image Classification API")

# Enable CORS for all origins (development)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Determine device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Model paths - check both weights/ subdirectory and models/ directory
BASE_DIR = Path(__file__).parent.parent
RESNET_MODEL_PATH = BASE_DIR / "models" / "weights" / "resnet_2_model.pt"
VIT_MODEL_PATH = BASE_DIR / "models" / "weights" / "vit_2_model.pt"

# Fallback paths if models are directly in models/ directory
if not RESNET_MODEL_PATH.exists():
    RESNET_MODEL_PATH = BASE_DIR / "models" / "resnet_2_model.pt"
if not VIT_MODEL_PATH.exists():
    VIT_MODEL_PATH = BASE_DIR / "models" / "vit_2_model.pt"

# Global model variables
resnet_model = None
vit_model = None


def safe_torch_load(path, map_location, **kwargs):
    """
    Safely load a PyTorch model, ensuring SEBlock is available for unpickling.
    """
    # Ensure SEBlock is available in all necessary namespaces before loading
    import types
    
    # Make sure __main__ has SEBlock
    if '__main__' not in sys.modules:
        sys.modules['__main__'] = types.ModuleType('__main__')
    if not hasattr(sys.modules['__main__'], 'SEBlock'):
        sys.modules['__main__'].SEBlock = SEBlock
    
    # Make sure __mp_main__ has SEBlock (used by multiprocessing)
    if '__mp_main__' not in sys.modules:
        sys.modules['__mp_main__'] = types.ModuleType('__mp_main__')
    if not hasattr(sys.modules['__mp_main__'], 'SEBlock'):
        sys.modules['__mp_main__'].SEBlock = SEBlock
    
    # Now load the model
    return torch.load(str(path), map_location=map_location, **kwargs)


def load_models():
    """Load models at startup."""
    global resnet_model, vit_model
    
    try:
        if RESNET_MODEL_PATH.exists():
            print(f"Loading ResNet model from {RESNET_MODEL_PATH}")
            # Use safe loader to ensure SEBlock is available
            resnet_model = safe_torch_load(RESNET_MODEL_PATH, device, weights_only=False)
            # Handle case where model might be wrapped in a dict or other structure
            if isinstance(resnet_model, dict):
                if 'model' in resnet_model:
                    resnet_model = resnet_model['model']
                elif 'state_dict' in resnet_model:
                    print("Warning: Model file contains state_dict, not full model")
                    resnet_model = None
            if resnet_model is not None:
                resnet_model.eval()
                resnet_model.to(device)
                print("ResNet model loaded successfully")
        else:
            print(f"Warning: ResNet model not found at {RESNET_MODEL_PATH}")
            
        if VIT_MODEL_PATH.exists():
            print(f"Loading ViT model from {VIT_MODEL_PATH}")
            # Use standard torch.load for ViT (no custom classes expected)
            vit_model = torch.load(str(VIT_MODEL_PATH), map_location=device, weights_only=False)
            # Handle case where model might be wrapped in a dict or other structure
            if isinstance(vit_model, dict):
                if 'model' in vit_model:
                    vit_model = vit_model['model']
                elif 'state_dict' in vit_model:
                    print("Warning: Model file contains state_dict, not full model")
                    vit_model = None
            if vit_model is not None:
                vit_model.eval()
                vit_model.to(device)
                print("ViT model loaded successfully")
        else:
            print(f"Warning: ViT model not found at {VIT_MODEL_PATH}")
            
    except Exception as e:
        print(f"Error loading models: {e}")
        traceback.print_exc()
        raise


@app.on_event("startup")
async def startup_event():
    """Load models on startup."""
    load_models()


@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve the frontend HTML."""
    index_path = BASE_DIR / "static" / "index.html"
    if index_path.exists():
        with open(index_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>Frontend not found</h1>"


# Mount static files
static_dir = BASE_DIR / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "device": str(device),
        "resnet_loaded": resnet_model is not None,
        "vit_loaded": vit_model is not None
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    model: Literal["resnet", "vit"] = Query("resnet")
):
    """
    Predict image class using specified model.
    
    Query parameters:
    - model: "resnet" or "vit" (default: "resnet")
    
    Request:
    - multipart/form-data with field name "file"
    """
    return await _predict_image(file, model)


@app.post("/predict/resnet")
async def predict_resnet(file: UploadFile = File(...)):
    """Predict using ResNet model."""
    return await _predict_image(file, "resnet")


@app.post("/predict/vit")
async def predict_vit(file: UploadFile = File(...)):
    """Predict using ViT model."""
    return await _predict_image(file, "vit")


async def _predict_image(file: UploadFile, model_name: str):
    """Internal prediction function."""
    # Validate file type
    if file.content_type not in ["image/jpeg", "image/png", "image/webp"]:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}. Only JPEG, PNG, and WebP are supported."
        )
    
    # Select model
    if model_name == "resnet":
        model = resnet_model
        if model is None:
            raise HTTPException(status_code=500, detail="ResNet model not loaded")
    elif model_name == "vit":
        model = vit_model
        if model is None:
            raise HTTPException(status_code=500, detail="ViT model not loaded")
    else:
        raise HTTPException(status_code=400, detail=f"Invalid model: {model_name}")
    
    try:
        # Read image file
        contents = await file.read()
        
        # Open and validate image
        try:
            image = Image.open(io.BytesIO(contents)).convert("RGB")
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid image file: {str(e)}"
            )
        
        # Preprocess image
        input_tensor = preprocess_image(image)
        input_tensor = input_tensor.unsqueeze(0).to(device)
        
        # Run inference
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
        
        # Get top predictions
        top1, top5 = get_top_predictions(probabilities, CLASSES)
        
        return {
            "model": model_name,
            "top1": top1,
            "top5": top5
        }
        
    except HTTPException:
        raise
    except Exception as e:
        # Log full traceback for debugging
        error_trace = traceback.format_exc()
        print(f"Prediction error: {str(e)}")
        print(f"Full traceback:\n{error_trace}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )

