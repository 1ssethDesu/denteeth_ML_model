#!/usr/bin/env python3
"""
Example client for the Dental X-Ray Detection API
"""

import requests
import json
import base64
from PIL import Image
import io
import argparse
from pathlib import Path

class DentalDetectionClient:
    """Client for the Dental X-Ray Detection API"""
    
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url.rstrip('/')
    
    def health_check(self):
        """Check API health"""
        try:
            response = requests.get(f"{self.base_url}/health")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Health check failed: {e}")
            return None
    
    def get_available_diseases(self):
        """Get list of available diseases"""
        try:
            response = requests.get(f"{self.base_url}/diseases")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Failed to get diseases: {e}")
            return None
    
    def get_disease_info(self, disease_name):
        """Get information about a specific disease"""
        try:
            response = requests.get(f"{self.base_url}/disease/{disease_name}")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Failed to get disease info: {e}")
            return None
    
    def predict(self, image_path):
        """Upload image and get prediction"""
        try:
            with open(image_path, 'rb') as f:
                files = {'file': (Path(image_path).name, f, 'image/jpeg')}
                response = requests.post(f"{self.base_url}/predict", files=files)
                response.raise_for_status()
                return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Prediction failed: {e}")
            return None
        except FileNotFoundError:
            print(f"Image file not found: {image_path}")
            return None
    
    def save_prediction_image(self, prediction_result, output_path):
        """Save the processed image from prediction result"""
        try:
            if 'image' in prediction_result:
                image_data = base64.b64decode(prediction_result['image'])
                with open(output_path, 'wb') as f:
                    f.write(image_data)
                print(f"Processed image saved to: {output_path}")
            else:
                print("No image data in prediction result")
        except Exception as e:
            print(f"Failed to save image: {e}")

def main():
    """Main function to demonstrate API usage"""
    parser = argparse.ArgumentParser(description="Dental Detection API Client")
    parser.add_argument(
        "--url",
        default="http://localhost:8000",
        help="API base URL (default: http://localhost:8000)"
    )
    parser.add_argument(
        "--image",
        help="Path to image file for prediction"
    )
    parser.add_argument(
        "--output",
        default="output.jpg",
        help="Output path for processed image (default: output.jpg)"
    )
    parser.add_argument(
        "--disease",
        help="Get information about a specific disease"
    )
    parser.add_argument(
        "--health",
        action="store_true",
        help="Check API health"
    )
    parser.add_argument(
        "--diseases",
        action="store_true",
        help="List available diseases"
    )
    
    args = parser.parse_args()
    
    client = DentalDetectionClient(args.url)
    
    # Health check
    if args.health:
        print("Checking API health...")
        health = client.health_check()
        if health:
            print(f"Status: {health['status']}")
            print(f"Model loaded: {health['model_loaded']}")
            print(f"Version: {health['version']}")
        return
    
    # List diseases
    if args.diseases:
        print("Getting available diseases...")
        diseases = client.get_available_diseases()
        if diseases:
            print("Available diseases:")
            for disease in diseases['diseases']:
                print(f"  - {disease}")
        return
    
    # Get disease info
    if args.disease:
        print(f"Getting information about {args.disease}...")
        info = client.get_disease_info(args.disease)
        if info:
            print(f"Disease: {info['disease']}")
            print(f"Description: {info['description']}")
            print(f"Causes: {info['causes']}")
            print(f"Symptoms: {info['symptoms']}")
            print(f"Treatment: {info['treatment']}")
            print(f"Prevention: {info['prevention']}")
        return
    
    # Prediction
    if args.image:
        print(f"Making prediction for image: {args.image}")
        result = client.predict(args.image)
        if result:
            print(f"Filename: {result['filename']}")
            print(f"Processing time: {result['processing_time']:.2f}s")
            print(f"Found {len(result['predictions'])} detections:")
            
            for i, pred in enumerate(result['predictions'], 1):
                print(f"  {i}. {pred['class_name']} (confidence: {pred['confidence']:.2f})")
                print(f"     Bounding box: {pred['bbox']}")
            
            if result['disease_info']:
                print(f"\nDisease information:")
                for info in result['disease_info']:
                    print(f"  - {info['disease']}: {info['description'][:100]}...")
            
            # Save processed image
            client.save_prediction_image(result, args.output)
        return
    
    # Default: show help
    print("Dental Detection API Client")
    print("Use --help to see available options")
    print("\nExample usage:")
    print("  python client_example.py --health")
    print("  python client_example.py --diseases")
    print("  python client_example.py --disease gum-disease")
    print("  python client_example.py --image xray.jpg --output result.jpg")

if __name__ == "__main__":
    main() 