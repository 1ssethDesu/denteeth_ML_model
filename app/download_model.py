import gdown
import os
def download(model_path: str):
    """Download the latest model from Google Drive."""

    # TEMPORARY: Replace this with your real Google Drive file URL
    url = "https://drive.google.com/file/d/1uLiLu7RE8n3vUIyj5aGQRx6ytgKz-vel/view?usp=sharing"

    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    print("Getting latest model from Google Drive...")
    gdown.download(url=url, output=model_path, quiet=False, fuzzy=True)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at path: {model_path}")
    
    print("Model downloaded successfully!")
