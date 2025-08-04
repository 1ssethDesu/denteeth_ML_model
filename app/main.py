"""
Dental X-Ray Detection API using Faster R-CNN
A FastAPI application for detecting dental diseases in X-ray images.
"""

import logging
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.config import (
    API_TITLE,
    API_DESCRIPTION,
    API_VERSION,
    CORS_ORIGINS,
    LOG_LEVEL,
    LOG_FORMAT,
    CLASS_MAPPING
)
from app.services import prediction_service
from app.validators import (
    validate_image_file,
    validate_file_size,
    load_and_validate_image,
    validate_disease_name
)
from app.util.helper import get_disease_info

# Configure logging
logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

# Pydantic Models
class Detection(BaseModel):
    """Represents a single detection result"""
    class_name: str = Field(..., description="Name of the detected disease")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    bbox: List[float] = Field(..., description="Bounding box coordinates [x1, y1, x2, y2]")

class DiseaseInfo(BaseModel):
    """Represents disease information"""
    disease: str = Field(..., description="Name of the disease")
    description: str = Field(..., description="Description of the disease")
    causes: str = Field(..., description="Causes of the disease")
    symptoms: str = Field(..., description="Symptoms of the disease")
    treatment: str = Field(..., description="Treatment options")
    prevention: str = Field(..., description="Prevention methods")

class PredictionResponse(BaseModel):
    """API response model"""
    filename: str = Field(..., description="Original filename")
    predictions: List[Detection] = Field(..., description="List of detections")
    image: str = Field(..., description="Base64 encoded processed image")
    disease_info: List[DiseaseInfo] = Field(..., description="Information about detected diseases")
    processing_time: Optional[float] = Field(None, description="Processing time in seconds")

class HealthResponse(BaseModel):
    """Health check response"""
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether the model is loaded")
    version: str = Field(..., description="API version")

# Lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle application startup and shutdown"""
    logger.info("Starting application...")
    try:
        # Initialize prediction service (this will load the model)
        prediction_service._get_model()
        logger.info("Model loaded successfully!")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise RuntimeError("Application startup failed due to model loading error")
    
    yield
    
    logger.info("Application shutting down...")

# Initialize FastAPI app
app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Endpoints
@app.get("/", response_model=dict)
async def root():
    """Root endpoint"""
    return {
        "message": "Welcome to Dental X-Ray Detection API",
        "version": API_VERSION,
        "docs": "/docs"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    try:
        model = prediction_service._get_model()
        model_loaded = model is not None
    except Exception:
        model_loaded = False
    
    return HealthResponse(
        status="healthy" if model_loaded else "unhealthy",
        model_loaded=model_loaded,
        version=API_VERSION
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """
    Perform dental disease detection on uploaded X-ray image.
    
    Args:
        file: Image file (JPEG, PNG, BMP, TIFF)
    
    Returns:
        PredictionResponse with detections, processed image, and disease information
    """
    try:
        # Validate file
        validate_image_file(file)
        
        # Read and validate file size
        contents = await file.read()
        validate_file_size(contents)
        
        # Load and validate image
        image = load_and_validate_image(contents)
        logger.info(f"Processing image: {file.filename}")
        
        # Perform prediction
        result = prediction_service.predict(image)
        
        # Convert detections to Pydantic models
        detections = [Detection(**detection) for detection in result["predictions"]]
        disease_info = [DiseaseInfo(**info) for info in result["disease_info"]]
        
        return PredictionResponse(
            filename=file.filename,
            predictions=detections,
            image=result["image"],
            disease_info=disease_info,
            processing_time=result["processing_time"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during prediction"
        )

@app.get("/diseases")
async def get_available_diseases():
    """Get list of available diseases that can be detected"""
    diseases = [name for name in CLASS_MAPPING.values() if name != "background"]
    return {"diseases": diseases}

@app.get("/disease/{disease_name}", response_model=DiseaseInfo)
async def get_disease_information(disease_name: str):
    """Get detailed information about a specific disease"""
    # Validate disease name
    normalized_name = validate_disease_name(disease_name)
    
    # Get disease information
    info = get_disease_info(normalized_name)
    if "error" in info:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Disease '{disease_name}' not found"
        )
    return DiseaseInfo(**info)

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions"""
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "Internal server error"}
    )

if __name__ == "__main__":
    import uvicorn
    from app.config import HOST, PORT, DEBUG
    
    uvicorn.run(
        "main:app",
        host=HOST,
        port=PORT,
        reload=DEBUG,
        log_level=LOG_LEVEL.lower()
    )