"""
Configuration settings for the Dental X-Ray Detection API
"""

import os
from typing import Dict, Tuple

# API Configuration
API_TITLE = "Dental X-Ray Detection API"
API_DESCRIPTION = "A FastAPI application for detecting dental diseases in X-ray images using Faster R-CNN"
API_VERSION = "1.0.0"

# Server Configuration
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
DEBUG = os.getenv("DEBUG", "False").lower() == "true"

# Model Configuration
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.7"))
# Use relative path for model, will be resolved in model.py
MODEL_PATH = os.getenv("CUSTOM_MODEL_PATH", "model/faster_rcnn_model.pth")

# File Upload Configuration
MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", str(10 * 1024 * 1024)))  # 10MB
SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}

# Font Configuration
FONT_PATH = os.getenv("FONT_PATH", "arialbd.ttf")
FONT_SIZE = int(os.getenv("FONT_SIZE", "20"))

# Class Mapping for the Model
CLASS_MAPPING: Dict[int, str] = {
    0: "background",
    1: "gum-disease",
    2: "tooth-decay", 
    3: "tooth-loss"
}

# Color Mapping for Bounding Boxes (RGB format)
CLASS_COLORS: Dict[str, Tuple[int, int, int]] = {
    "gum-disease": (0, 255, 0),    # Green
    "tooth-decay": (255, 0, 0),    # Red
    "tooth-loss": (0, 0, 255),     # Blue
    "unknown": (255, 255, 0)       # Yellow (fallback)
}

# CORS Configuration
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")

# Logging Configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Health Check Configuration
HEALTH_CHECK_ENABLED = os.getenv("HEALTH_CHECK_ENABLED", "True").lower() == "true"

# Image Processing Configuration
IMAGE_QUALITY = int(os.getenv("IMAGE_QUALITY", "85"))  # JPEG quality
BOUNDING_BOX_WIDTH = int(os.getenv("BOUNDING_BOX_WIDTH", "4"))

# Disease Information Configuration
DISEASE_INFO_CSV_PATH = os.getenv("DISEASE_INFO_CSV_PATH", "app/util/dental_diseases_info.csv") 