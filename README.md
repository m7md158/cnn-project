# EuroSAT Image Classification API

A complete FastAPI application for deploying two PyTorch image classification models (ResNet and Vision Transformer) with a modern web frontend.

## Features

- **Two Models**: ResNet and Vision Transformer (ViT) for EuroSAT land cover classification
- **RESTful API**: FastAPI endpoints for image classification
- **Modern Frontend**: Beautiful, responsive web interface with drag-and-drop image upload
- **Top-5 Predictions**: Display top-1 and top-5 predictions with confidence scores

## Project Structure

```
cnn_project/
├── app/
│   ├── __init__.py
│   ├── main.py          # FastAPI application
│   └── utils.py         # Preprocessing utilities
├── models/
│   └── weights/
│       ├── resnet_2_model.pt    # ResNet model (full model saved with torch.save)
│       └── vit_2_model.pt      # ViT model (full model saved with torch.save)
├── static/
│   ├── index.html      # Frontend HTML
│   ├── style.css       # Frontend styles
│   └── script.js       # Frontend JavaScript
├── run_server.py       # Server startup script
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## Setup

### 1. Install Dependencies

Create a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

Install required packages:

```bash
pip install -r requirements.txt
```

### 2. Place Model Files

**Important**: Place your model files in the `models/weights/` directory:
- `models/weights/resnet_2_model.pt` (ResNet model - full model saved with `torch.save(model, ...)`)
- `models/weights/vit_2_model.pt` (ViT model - full model saved with `torch.save(model, ...)`)

The models must be saved as **FULL MODELS** using `torch.save(model, "path")`, not just state_dict.

If the models are missing, the application will start but prediction endpoints will return errors.

## Running the Application

### Option 1: Using run_server.py (Recommended)

```bash
python run_server.py
```

### Option 2: Using uvicorn directly

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The server will start on `http://localhost:8000`

## API Endpoints

### POST `/predict`
Predict image class using specified model.

**Query Parameters:**
- `model` (optional): `"resnet"` or `"vit"` (default: `"resnet"`)

**Request:**
- Content-Type: `multipart/form-data`
- Field name: `file` (image file - JPEG, PNG, or WebP only)

**Response:**
```json
{
  "model": "resnet",
  "top1": {
    "label": "Forest",
    "confidence": 0.9542
  },
  "top5": [
    {"label": "Forest", "confidence": 0.9542},
    {"label": "HerbaceousVegetation", "confidence": 0.0234},
    {"label": "Pasture", "confidence": 0.0123},
    {"label": "AnnualCrop", "confidence": 0.0056},
    {"label": "Residential", "confidence": 0.0021}
  ]
}
```

### POST `/predict/resnet`
Predict using ResNet model (same as `/predict?model=resnet`).

### POST `/predict/vit`
Predict using ViT model (same as `/predict?model=vit`).

### GET `/health`
Health check endpoint. Returns model loading status.

### GET `/`
Serves the frontend web interface.

## Example API Usage

### Using curl

```bash
# Predict with ResNet
curl -X POST "http://localhost:8000/predict?model=resnet" \
  -F "file=@path/to/your/image.jpg"

# Predict with ViT
curl -X POST "http://localhost:8000/predict/vit" \
  -F "file=@path/to/your/image.jpg"
```

## Model Details

Models are loaded as **full models** using `torch.load()` (not state_dict). Both models are:
- Loaded once at server startup
- Set to evaluation mode (`model.eval()`)
- Moved to available device (CUDA if available, else CPU)

### Preprocessing
All images are preprocessed with:
- Resize to 224x224
- Center crop 224x224
- Convert to tensor
- Normalize with ImageNet statistics: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]

### Classes (EuroSAT)
1. AnnualCrop
2. Forest
3. HerbaceousVegetation
4. Highway
5. Industrial
6. Pasture
7. PermanentCrop
8. Residential
9. River
10. SeaLake

## Frontend Usage

1. Open `http://localhost:8000` in your browser
2. Upload an image by clicking the upload box or dragging and dropping
3. Select a model (ResNet or ViT)
4. Click "Predict"
5. View the top-1 and top-5 predictions with confidence scores

The frontend uses relative API paths, so it works seamlessly when served from the same FastAPI server.

## Error Handling

The application handles:
- Invalid image file types (only JPEG, PNG, WebP accepted)
- Corrupted image files (returns HTTP 400 with clear error message)
- Missing model files (returns HTTP 500 when trying to use unloaded model)
- Network errors
- File size limits (handled by browser/client)

## Development

### Auto-reload
The server runs with `reload=True` by default for development. Changes to Python files will automatically restart the server.

### Static Files
Static files (HTML, CSS, JS) are served from the `static/` directory:
- Root path (`/`) serves `static/index.html`
- Static assets are mounted at `/static` (CSS, JS files)

## Troubleshooting

### Models not loading
- Check that model files exist at:
  - `models/weights/resnet_2_model.pt`
  - `models/weights/vit_2_model.pt`
- Verify models were saved as full models using `torch.save(model, "path")`
- Check console output for error messages during startup

### Port already in use
- Change the port in `run_server.py` or use:
  ```bash
  uvicorn app.main:app --reload --host 0.0.0.0 --port 8001
  ```

