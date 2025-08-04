"""
Tests for the Dental X-Ray Detection API
"""

import pytest
from fastapi.testclient import TestClient
from PIL import Image
import io
import base64

from app.main import app

client = TestClient(app)

class TestAPIEndpoints:
    """Test cases for API endpoints"""
    
    def test_root_endpoint(self):
        """Test the root endpoint"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data
        assert "docs" in data
    
    def test_health_check(self):
        """Test the health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "version" in data
        assert isinstance(data["model_loaded"], bool)
    
    def test_get_diseases(self):
        """Test getting available diseases"""
        response = client.get("/diseases")
        assert response.status_code == 200
        data = response.json()
        assert "diseases" in data
        assert isinstance(data["diseases"], list)
        # Should include the three main diseases
        expected_diseases = ["gum-disease", "tooth-decay", "tooth-loss"]
        for disease in expected_diseases:
            assert disease in data["diseases"]
    
    def test_get_disease_information_valid(self):
        """Test getting disease information for valid disease"""
        response = client.get("/disease/gum-disease")
        assert response.status_code == 200
        data = response.json()
        assert "disease" in data
        assert "description" in data
        assert "causes" in data
        assert "symptoms" in data
        assert "treatment" in data
        assert "prevention" in data
    
    def test_get_disease_information_invalid(self):
        """Test getting disease information for invalid disease"""
        response = client.get("/disease/invalid-disease")
        assert response.status_code == 404
        data = response.json()
        assert "error" in data
    
    def test_get_disease_information_empty(self):
        """Test getting disease information with empty name"""
        response = client.get("/disease/")
        assert response.status_code == 404

class TestPredictionEndpoint:
    """Test cases for prediction endpoint"""
    
    def create_test_image(self, size=(224, 224)):
        """Create a test image for testing"""
        image = Image.new('RGB', size, color='white')
        # Add some simple shapes to make it more realistic
        from PIL import ImageDraw
        draw = ImageDraw.Draw(image)
        draw.rectangle([50, 50, 150, 150], fill='gray')
        return image
    
    def test_predict_with_valid_image(self):
        """Test prediction with a valid image"""
        # Create a test image
        image = self.create_test_image()
        
        # Convert to bytes
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='JPEG')
        img_byte_arr.seek(0)
        
        # Make request
        files = {"file": ("test.jpg", img_byte_arr.getvalue(), "image/jpeg")}
        response = client.post("/predict", files=files)
        
        # Check response
        assert response.status_code == 200
        data = response.json()
        assert "filename" in data
        assert "predictions" in data
        assert "image" in data
        assert "disease_info" in data
        assert "processing_time" in data
        assert isinstance(data["predictions"], list)
        assert isinstance(data["disease_info"], list)
        assert isinstance(data["processing_time"], (int, float))
    
    def test_predict_with_invalid_file_type(self):
        """Test prediction with invalid file type"""
        # Create a text file instead of image
        files = {"file": ("test.txt", b"not an image", "text/plain")}
        response = client.post("/predict", files=files)
        assert response.status_code == 400
        data = response.json()
        assert "error" in data
    
    def test_predict_without_file(self):
        """Test prediction without file"""
        response = client.post("/predict")
        assert response.status_code == 422  # Validation error
    
    def test_predict_with_large_file(self):
        """Test prediction with file that's too large"""
        # Create a large image
        image = self.create_test_image((1000, 1000))
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='JPEG', quality=100)
        img_byte_arr.seek(0)
        
        # Make the file larger by repeating the data
        large_data = img_byte_arr.getvalue() * 1000  # Make it much larger
        
        files = {"file": ("large.jpg", large_data, "image/jpeg")}
        response = client.post("/predict", files=files)
        assert response.status_code == 413  # Request Entity Too Large

class TestValidation:
    """Test cases for validation functions"""
    
    def test_validate_disease_name_valid(self):
        """Test validation of valid disease names"""
        from app.validators import validate_disease_name
        
        # Test valid names
        assert validate_disease_name("gum-disease") == "gum-disease"
        assert validate_disease_name("TOOTH-DECAY") == "tooth-decay"
        assert validate_disease_name("  tooth-loss  ") == "tooth-loss"
    
    def test_validate_disease_name_invalid(self):
        """Test validation of invalid disease names"""
        from app.validators import validate_disease_name
        from fastapi import HTTPException
        
        # Test empty names
        with pytest.raises(HTTPException) as exc_info:
            validate_disease_name("")
        assert exc_info.value.status_code == 400
        
        with pytest.raises(HTTPException) as exc_info:
            validate_disease_name("   ")
        assert exc_info.value.status_code == 400

class TestConfiguration:
    """Test cases for configuration"""
    
    def test_config_imports(self):
        """Test that configuration can be imported"""
        from app.config import (
            API_TITLE,
            API_VERSION,
            CONFIDENCE_THRESHOLD,
            CLASS_MAPPING
        )
        
        assert API_TITLE == "Dental X-Ray Detection API"
        assert API_VERSION == "1.0.0"
        assert isinstance(CONFIDENCE_THRESHOLD, float)
        assert isinstance(CLASS_MAPPING, dict)
        assert len(CLASS_MAPPING) > 0

class TestServices:
    """Test cases for services"""
    
    def test_prediction_service_import(self):
        """Test that prediction service can be imported"""
        from app.services import prediction_service
        assert prediction_service is not None
    
    def test_prediction_service_methods(self):
        """Test prediction service methods exist"""
        from app.services import prediction_service
        
        # Check that required methods exist
        assert hasattr(prediction_service, 'predict')
        assert hasattr(prediction_service, 'preprocess_image')
        assert hasattr(prediction_service, 'perform_inference')
        assert hasattr(prediction_service, 'process_detections')

if __name__ == "__main__":
    pytest.main([__file__]) 