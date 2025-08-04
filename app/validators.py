"""
Validation utilities for the Dental X-Ray Detection API
"""

import os
from typing import Optional
from fastapi import HTTPException, UploadFile, status
from PIL import Image
import io

from app.config import MAX_FILE_SIZE, SUPPORTED_FORMATS

def validate_image_file(file: UploadFile) -> None:
    """
    Validate uploaded file for image processing
    
    Args:
        file: Uploaded file to validate
        
    Raises:
        HTTPException: If file is invalid
    """
    # Check content type
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is not an image"
        )
    
    # Check file extension
    if file.filename:
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in SUPPORTED_FORMATS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported file format. Supported formats: {', '.join(SUPPORTED_FORMATS)}"
            )

def validate_file_size(contents: bytes) -> None:
    """
    Validate file size
    
    Args:
        contents: File contents as bytes
        
    Raises:
        HTTPException: If file is too large
    """
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File size exceeds maximum limit of {MAX_FILE_SIZE // (1024*1024)}MB"
        )

def load_and_validate_image(contents: bytes) -> Image.Image:
    """
    Load and validate image from bytes
    
    Args:
        contents: Image file contents as bytes
        
    Returns:
        PIL Image object
        
    Raises:
        HTTPException: If image cannot be loaded
    """
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        return image
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Could not load image: {str(e)}"
        )

def validate_disease_name(disease_name: str) -> str:
    """
    Validate disease name
    
    Args:
        disease_name: Name of the disease to validate
        
    Returns:
        Normalized disease name
        
    Raises:
        HTTPException: If disease name is invalid
    """
    if not disease_name or not disease_name.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Disease name cannot be empty"
        )
    
    return disease_name.strip().lower() 