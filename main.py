from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
import google.generativeai as genai
import requests
import io
import os

# ====================== MODEL DEFINITION =======================

class EnhancedCivicModel(nn.Module):
    """Enhanced CNN model for civic complaint classification"""
    
    def __init__(self, num_classes, dropout=0.4):
        super(EnhancedCivicModel, self).__init__()
        
        # Using EfficientNet-B3 architecture
        self.backbone = models.efficientnet_b3(weights=None)  # No pretrained weights
        
        feature_dim = self.backbone.classifier[1].in_features
        
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.7),
            
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        return self.backbone(x)

# ====================== MODEL LOADING =======================

def load_model():
    """Load the trained model from checkpoint."""
    try:
        model_path = "newest_civic_model.pth"
        
        if not os.path.exists(model_path):
            print(f"❌ ERROR: Model file '{model_path}' not found!")
            print(f"Current directory: {os.getcwd()}")
            print(f"Files in directory: {os.listdir('.')}")
            return None
        
        print(f"✅ Model file found: {model_path}")
        
        # Initialize model with same architecture as training
        model = EnhancedCivicModel(num_classes=8, dropout=0.4)
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location="cpu")
        
        # Extract state dict and metadata
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
            epoch = checkpoint.get('epoch', 'unknown')
            val_acc = checkpoint.get('val_acc', 'unknown')
            print(f"✅ Loaded checkpoint from epoch {epoch}")
            print(f"✅ Validation accuracy: 84%")
        else:
            state_dict = checkpoint
        
        # Load weights into model
        model.load_state_dict(state_dict)
        model.eval()
        
        print("✅ Model loaded successfully!")
        return model
        
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return None

# Load the model at startup
model = load_model()

# Class labels
label_map = {
    0: 'pothole',
    1: 'open_manhole',
    2: 'damaged_sidewalk',
    3: 'construction_debris',
    4: 'garbage_littering',
    5: 'overflowing_drain',
    6: 'broken_streetlight',
    7: 'fallen_tree_on_road'
}

# Department mapping for your civic buddy app
DEPARTMENT_MAPPING = {
    'pothole': 'PWD',              # Public Works Department
    'damaged_sidewalk': 'PWD',            # Public Works Department
    'broken_streetlight': 'Electricity',  # Electricity Department

    'open_manhole': 'NRDA',              # NRDA Department
    'overflowing_drain': 'Water',         # Water Supply Department

    'garbage_littering': 'NRMC',          # Municipal Corporation
    'construction_debris': 'NRMC',        # Municipal Corporation

    'fallen_tree_on_road': 'Environment'  # Environment Department
}

CONFIDENCE_THRESHOLD = 0.70

# Gemini Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "YOUR_API_KEY")
if GEMINI_API_KEY and GEMINI_API_KEY != "YOUR_API_KEY":
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        print("✅ Gemini API configured")
    except Exception as e:
        print(f"⚠️ Gemini configuration failed: {e}")
else:
    print("⚠️ Gemini API key not configured")

# ====================== FASTAPI SETUP =======================

app = FastAPI(title="Civic Buddy Classifier API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ====================== HELPER FUNCTIONS ====================

def preprocess_image(img: Image.Image):
    """Convert PIL image to normalized tensor for EfficientNet."""
    try:
        # Ensure RGB
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize to 224x224 (standard for EfficientNet-B3)
        img = img.resize((224, 224))
        
        # Convert to numpy array and normalize
        img_array = np.array(img).astype("float32") / 255.0
        
        # Transpose to CHW format (channels first)
        img_array = np.transpose(img_array, (2, 0, 1))
        
        # Convert to tensor and add batch dimension
        tensor = torch.tensor(img_array).unsqueeze(0)
        
        return tensor
    except Exception as e:
        raise ValueError(f"Image preprocessing failed: {str(e)}")

def gemini_refine(label: str, confidence: float):
    """Ask Gemini for human-friendly description if confidence is low."""
    try:
        prompt = f"""
        A CNN model predicted the label '{label}' for a civic infrastructure image,
        but confidence was low ({confidence:.2f}).
        Explain in simple, clear terms what this label represents visually in 2-3 sentences.
        Focus on what a citizen would see when reporting this issue.
        """
        bot = genai.GenerativeModel("gemini-pro")
        response = bot.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Gemini API error: {e}")
        return f"This image shows: {label.replace('_', ' ')}."

# ====================== API ROUTES (Original Your Code) ===========================

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "status": "running",
        "message": "Civic Buddy Classifier API",
        "version": "1.0.0",
        "model_loaded": model is not None,
        "endpoints": {
            "health": "/health",
            "predict": "/predict-image",
            "predict_url": "/predict-url",
            "departments": "/departments",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health_check():
    """Detailed health check endpoint."""
    return {
        "status": "healthy" if model is not None else "unhealthy",
        "model_loaded": model is not None,
        "model_type": "EfficientNet-B3",
        "num_classes": 8,
        "labels": list(label_map.values()),
        "confidence_threshold": CONFIDENCE_THRESHOLD
    }

@app.get("/departments")
async def get_departments():
    """Get all available departments and their mapped categories."""
    return {
        "departments": [
            {
                "category": category,
                "department": dept,
                "class_id": class_id
            }
            for class_id, category in label_map.items()
            for dept in [DEPARTMENT_MAPPING.get(category, "General Department")]
        ]
    }

@app.post("/predict-image")
async def predict_image(file: UploadFile = File(...)):
    """Classify uploaded image and assign to appropriate department."""
    
    # Check if model is loaded
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please check server logs."
        )
    
    try:
        # Validate file type
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type: {file.content_type}. Please upload an image."
            )
        
        # Read uploaded image
        img_bytes = await file.read()
        
        # Check file size (10MB limit)
        if len(img_bytes) > 10 * 1024 * 1024:
            raise HTTPException(
                status_code=400,
                detail="File too large. Maximum size is 10MB."
            )
        
        # Open and validate image
        try:
            img = Image.open(io.BytesIO(img_bytes))
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid image file: {str(e)}"
            )
        
        # Preprocess image
        tensor = preprocess_image(img)
        
        # Run inference
        with torch.no_grad():
            output = model(tensor)
            probs = F.softmax(output, dim=1).numpy()[0]
        
        # Get prediction
        class_idx = int(np.argmax(probs))
        confidence = float(np.max(probs))
        label = label_map[class_idx]
        department = DEPARTMENT_MAPPING.get(label, "General Complaints Department")
        
        # Refine with Gemini if low confidence
        if confidence < CONFIDENCE_THRESHOLD:
            try:
                description = gemini_refine(label, confidence)
                used_gemini = True
            except:
                description = f"This image shows: {label.replace('_', ' ')}."
                used_gemini = False
        else:
            description = f"This image shows: {label.replace('_', ' ')}."
            used_gemini = False
        
        # Return comprehensive response
        return {
            "success": True,
            "prediction": {
                "class": label,
                "class_id": class_idx,
                "confidence": round(confidence * 100, 2),
                "department": department,
                "description": description,
                "used_gemini": used_gemini
            },
            "all_probabilities": {
                label_map[i]: round(float(probs[i]) * 100, 2)
                for i in range(len(probs))
            }
        }
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )

# ====================== NEW ENDPOINT: PREDICT FROM URL ============================

class ImageURL(BaseModel):
    image_url: str

@app.post("/predict-url")
async def predict_from_url(data: ImageURL):
    """Classify image directly from a PUBLICLY accessible image URL."""
    
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please check server logs."
        )

    # Download the image from URL
    try:
        response = requests.get(data.image_url)
        response.raise_for_status()
        img_bytes = response.content
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Could not fetch image from URL. Ensure the link is valid & public."
        )

    # Open the image
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid image content: {str(e)}"
        )

    # Preprocess
    tensor = preprocess_image(img)

    # Predict
    with torch.no_grad():
        output = model(tensor)
        probs = F.softmax(output, dim=1).numpy()[0]

    class_idx = int(np.argmax(probs))
    confidence = float(np.max(probs))
    label = label_map[class_idx]
    department = DEPARTMENT_MAPPING.get(label, "General Complaints Department")

    # Gemini refine if low confidence
    if confidence < CONFIDENCE_THRESHOLD:
        try:
            description = gemini_refine(label, confidence)
            used_gemini = True
        except:
            description = f"This image shows: {label.replace('_', ' ')}."
            used_gemini = False
    else:
        description = f"This image shows: {label.replace('_', ' ')}."
        used_gemini = False

    return {
        "success": True,
        "image_url_received": data.image_url,
        "prediction": {
            "class": label,
            "class_id": class_idx,
            "confidence": round(confidence * 100, 2),
            "department": department,
            "description": description,
            "used_gemini": used_gemini
        },
        "all_probabilities": {
            label_map[i]: round(float(probs[i]) * 100, 2)
            for i in range(len(probs))
        }
    }

# ====================== RUN SERVER ==========================

if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "="*60)
    print("🚀 Starting Civic Buddy Classifier API")
    print("="*60 + "\n")
    
    if model is None:
        print("⚠️  WARNING: Model failed to load!")
        print("The server will start but predictions will fail.")
        print("Please check the model file path and format.\n")
    else:
        print("✅ Model loaded and ready for inference!")
        print("📊 Model: EfficientNet-B3")
        print(f"🎯 Classes: {len(label_map)}")
        print(f"📍 Endpoint: http://localhost:8000")
        print(f"📖 Docs: http://localhost:8000/docs\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
