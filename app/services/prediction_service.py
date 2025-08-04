"""
Prediction service for dental disease detection
"""

import logging
import time
from typing import List, Tuple, Optional
import torch
import torchvision.transforms.functional as F
from PIL import Image, ImageDraw, ImageFont
import io
import base64

from app.model import get_model
from app.util.helper import get_disease_info
from app.config import (
    CONFIDENCE_THRESHOLD,
    CLASS_MAPPING,
    CLASS_COLORS,
    FONT_PATH,
    FONT_SIZE,
    IMAGE_QUALITY,
    BOUNDING_BOX_WIDTH
)

logger = logging.getLogger(__name__)

class PredictionService:
    """Service class for handling prediction operations"""
    
    def __init__(self):
        self.model = None
        self.score_font = None
        self._load_font()
    
    def _load_font(self):
        """Load font for drawing text on images"""
        try:
            self.score_font = ImageFont.truetype(FONT_PATH, FONT_SIZE)
            logger.info(f"Successfully loaded font from {FONT_PATH}")
        except (IOError, OSError):
            logger.warning(f"Could not load font from {FONT_PATH}. Using default PIL font.")
            self.score_font = ImageFont.load_default()
    
    def _get_model(self):
        """Get the loaded model instance"""
        if self.model is None:
            self.model = get_model()
        return self.model
    
    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """Preprocess image for model input"""
        img_tensor = F.to_tensor(image)
        return img_tensor
    
    def perform_inference(self, image_tensor: torch.Tensor) -> dict:
        """Perform inference on the preprocessed image"""
        model = self._get_model()
        model.eval()
        
        input_batch = [image_tensor]
        
        with torch.no_grad():
            prediction = model(input_batch)
        
        return prediction
    
    def process_detections(
        self, 
        prediction: dict, 
        image: Image.Image
    ) -> Tuple[List[dict], List[dict], Image.Image]:
        """
        Process model predictions and draw bounding boxes
        
        Returns:
            Tuple of (detections, disease_info, processed_image)
        """
        detections = []
        unique_diseases = set()
        disease_info_list = []
        
        draw = ImageDraw.Draw(image)
        
        if prediction and len(prediction) > 0:
            pred_data = prediction[0]
            boxes = pred_data["boxes"].cpu()
            labels = pred_data["labels"].cpu()
            scores = pred_data["scores"].cpu()
            
            # Filter by confidence threshold
            keep = scores > CONFIDENCE_THRESHOLD
            boxes = boxes[keep]
            labels = labels[keep]
            scores = scores[keep]
            
            logger.info(f"Found {len(boxes)} detections above confidence threshold {CONFIDENCE_THRESHOLD}")
            
            for box, label, score in zip(boxes, labels, scores):
                x1, y1, x2, y2 = [int(coord) for coord in box.tolist()]
                class_name = CLASS_MAPPING.get(int(label), f"unknown_label_{int(label)}")
                
                # Skip background class
                if class_name == "background":
                    continue
                
                # Add to detections
                detections.append({
                    "class_name": class_name,
                    "confidence": float(score.item()),
                    "bbox": [float(x1), float(y1), float(x2), float(y2)]
                })
                
                # Draw bounding box
                box_color = CLASS_COLORS.get(class_name, CLASS_COLORS["unknown"])
                draw.rectangle(
                    [(x1, y1), (x2, y2)], 
                    outline=box_color, 
                    width=BOUNDING_BOX_WIDTH
                )
                
                # Draw confidence score
                score_text = f"{score.item():.2f}"
                score_bbox = draw.textbbox((x1, y1), score_text, font=self.score_font)
                score_text_height = score_bbox[3] - score_bbox[1]
                
                # Position score text above the box
                score_y = max(0, y1 - score_text_height - 5)
                draw.text(
                    (x1 + 2, score_y + 2), 
                    score_text, 
                    fill=box_color, 
                    font=self.score_font
                )
                
                # Get disease information
                if class_name not in unique_diseases:
                    unique_diseases.add(class_name)
                    info = get_disease_info(class_name)
                    if "error" not in info and 'disease' in info:
                        disease_info_list.append(info)
                    else:
                        logger.warning(f"Disease info not found for '{class_name}'")
        
        return detections, disease_info_list, image
    
    def image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL image to base64 string"""
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG", quality=IMAGE_QUALITY)
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return img_base64
    
    def predict(self, image: Image.Image) -> dict:
        """
        Perform complete prediction pipeline
        
        Args:
            image: PIL Image to process
            
        Returns:
            Dictionary containing prediction results
        """
        start_time = time.time()
        
        try:
            # Preprocess image
            image_tensor = self.preprocess_image(image)
            
            # Perform inference
            prediction = self.perform_inference(image_tensor)
            
            # Process results
            detections, disease_info, processed_image = self.process_detections(prediction, image)
            
            # Convert to base64
            img_base64 = self.image_to_base64(processed_image)
            
            processing_time = time.time() - start_time
            
            logger.info(f"Prediction completed in {processing_time:.2f}s")
            
            return {
                "predictions": detections,
                "disease_info": disease_info,
                "image": img_base64,
                "processing_time": processing_time
            }
            
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise

# Global service instance
prediction_service = PredictionService() 