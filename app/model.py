import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import os

# ... (other imports)

def load_faster_rcnn_model(num_classes: int = 4):
    print("Starting model loading process...")
    print("Attempting to load pre-trained Faster R-CNN with default weights...")
    # This line will download weights if not cached
    model = fasterrcnn_resnet50_fpn(weights='DEFAULT')
    print("Pre-trained model loaded. Modifying classifier head...")

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    print("Classifier head modified. Attempting to load custom state_dict...")

    # Get model path from environment variable, or use a default path
    # The path should be relative to the application root
    model_path = os.getenv('CUSTOM_MODEL_PATH', 'model/faster_rcnn_model.pth')
    
    # If the path doesn't exist, try alternative locations
    if not os.path.exists(model_path):
        # Try different possible paths
        possible_paths = [
            model_path,
            f"/app/{model_path}",
            f"app/{model_path}",
            "/app/model/faster_rcnn_model.pth",
            "model/faster_rcnn_model.pth",
            "/Users/chhortchhorraseth/Desktop/Machine_learning/ML_model/model/faster_rcnn_model.pth"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                model_path = path
                print(f"Found model at: {model_path}")
                break
        else:
            print(f"Error: Model file not found. Tried the following paths:")
            for path in possible_paths:
                print(f"  - {path}")
            raise FileNotFoundError(f"Model file not found. Please ensure the model file exists.")

    try:
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        print(f"Custom model weights loaded successfully from {model_path}")
    except FileNotFoundError:
        print(f"Error: Custom model file not found at {model_path}. "
              "Please ensure the path is correct and the file exists in the deployment environment.")
        raise
    except Exception as e:
        print(f"Error loading custom weights: {e}")
        raise
    print("Model initialization complete.")
    return model

_global_model = None

def get_model():
    global _global_model
    if _global_model is None:
        _global_model = load_faster_rcnn_model(num_classes=4)
    return _global_model