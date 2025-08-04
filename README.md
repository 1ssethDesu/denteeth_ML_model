# Dental X-Ray Detection API

A FastAPI application for detecting dental diseases in X-ray images using Faster R-CNN. This application provides a RESTful API for uploading dental X-ray images and receiving predictions with bounding boxes and disease information.

## Features

- **Object Detection**: Uses Faster R-CNN to detect dental diseases in X-ray images
- **Disease Classification**: Detects gum disease, tooth decay, and tooth loss
- **Bounding Box Visualization**: Draws colored bounding boxes around detected diseases
- **Disease Information**: Provides detailed information about detected diseases
- **RESTful API**: Clean, well-documented API endpoints
- **Health Monitoring**: Built-in health check endpoints
- **Configurable**: Environment-based configuration
- **Production Ready**: Proper error handling, logging, and validation

## Architecture

The application follows a clean, modular architecture:

```
ML_model/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI application entry point
│   ├── config.py            # Configuration settings
│   ├── model.py             # Model loading and management
│   ├── validators.py        # Input validation utilities
│   ├── services/
│   │   ├── __init__.py
│   │   └── prediction_service.py  # Business logic for predictions
│   └── util/
│       ├── helper.py        # Utility functions
│       └── dental_diseases_info.csv  # Disease information database
├── model/
│   └── faster_rcnn_model.pth  # Trained model weights
├── requirements.txt         # Python dependencies
├── dockerfile              # Docker configuration
└── README.md              # This file
```

## Installation

### Prerequisites

- Python 3.8 or higher
- PyTorch (CPU or GPU version)
- At least 4GB RAM

### Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd ML_model
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables** (optional):
   ```bash
   # Create .env file
   cp .env.example .env
   # Edit .env with your configuration
   ```

## Configuration

The application can be configured using environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `HOST` | `0.0.0.0` | Server host |
| `PORT` | `8000` | Server port |
| `DEBUG` | `False` | Debug mode |
| `CONFIDENCE_THRESHOLD` | `0.7` | Minimum confidence for detections |
| `MAX_FILE_SIZE` | `10485760` | Maximum file size (10MB) |
| `LOG_LEVEL` | `INFO` | Logging level |
| `CORS_ORIGINS` | `*` | CORS allowed origins |

## Usage

### Starting the Server

```bash
# Development mode
python -m app.main

# Or using uvicorn directly
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### API Endpoints

#### 1. Root Endpoint
```http
GET /
```
Returns basic API information.

#### 2. Health Check
```http
GET /health
```
Returns the health status of the application and model.

#### 3. Prediction
```http
POST /predict
Content-Type: multipart/form-data

file: <image_file>
```
Upload an X-ray image and receive predictions.

**Response**:
```json
{
  "filename": "xray.jpg",
  "predictions": [
    {
      "class_name": "tooth-decay",
      "confidence": 0.85,
      "bbox": [100, 150, 200, 250]
    }
  ],
  "image": "base64_encoded_image",
  "disease_info": [
    {
      "disease": "tooth-decay",
      "description": "Dental caries...",
      "causes": "Bacterial infection...",
      "symptoms": "Toothache, sensitivity...",
      "treatment": "Dental filling...",
      "prevention": "Regular brushing..."
    }
  ],
  "processing_time": 1.23
}
```

#### 4. Available Diseases
```http
GET /diseases
```
Returns list of diseases the model can detect.

#### 5. Disease Information
```http
GET /disease/{disease_name}
```
Returns detailed information about a specific disease.

### API Documentation

Once the server is running, you can access:

- **Interactive API docs**: http://localhost:8000/docs
- **ReDoc documentation**: http://localhost:8000/redoc

## Model Information

The application uses a Faster R-CNN model trained on dental X-ray images to detect:

- **Gum Disease**: Periodontal disease affecting the gums
- **Tooth Decay**: Dental caries or cavities
- **Tooth Loss**: Missing teeth or severe decay

### Model Classes

| Class ID | Class Name | Description |
|----------|------------|-------------|
| 0 | background | No disease detected |
| 1 | gum-disease | Periodontal disease |
| 2 | tooth-decay | Dental caries |
| 3 | tooth-loss | Missing teeth |

## Development

### Project Structure

- **`app/main.py`**: FastAPI application with endpoints
- **`app/config.py`**: Configuration management
- **`app/model.py`**: Model loading and management
- **`app/services/`**: Business logic layer
- **`app/validators.py`**: Input validation
- **`app/util/`**: Utility functions and data

### Adding New Features

1. **New Endpoints**: Add to `app/main.py`
2. **Business Logic**: Create services in `app/services/`
3. **Configuration**: Add to `app/config.py`
4. **Validation**: Add to `app/validators.py`

### Testing

```bash
# Run tests
pytest

# Run with coverage
pytest --cov=app
```

### Code Quality

```bash
# Type checking
mypy app/

# Linting
flake8 app/

# Formatting
black app/
```

## Docker Deployment

### Build the Docker Image

```bash
docker build -t dental-detection-api .
```

### Run with Docker

```bash
docker run -p 8000:8000 dental-detection-api
```

### Docker Compose

```yaml
version: '3.8'
services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - CONFIDENCE_THRESHOLD=0.7
      - MAX_FILE_SIZE=10485760
    volumes:
      - ./model:/app/model
```

## Production Deployment

### Environment Variables

Set these in production:

```bash
export HOST=0.0.0.0
export PORT=8000
export DEBUG=False
export LOG_LEVEL=INFO
export CORS_ORIGINS=https://yourdomain.com
```

### Security Considerations

1. **CORS**: Configure `CORS_ORIGINS` for your domain
2. **File Upload**: Set appropriate `MAX_FILE_SIZE`
3. **Rate Limiting**: Consider adding rate limiting middleware
4. **Authentication**: Add authentication for production use

### Monitoring

The application includes:

- **Health checks**: `/health` endpoint
- **Structured logging**: Configurable log levels
- **Error handling**: Proper HTTP status codes
- **Performance metrics**: Processing time tracking

## Troubleshooting

### Common Issues

1. **Model Loading Error**:
   - Ensure the model file exists at the specified path
   - Check file permissions
   - Verify PyTorch installation

2. **Memory Issues**:
   - Reduce `MAX_FILE_SIZE`
   - Use CPU-only PyTorch if GPU memory is limited

3. **Font Loading Error**:
   - The application will fall back to default fonts
   - Install system fonts or update `FONT_PATH`

### Logs

Check application logs for detailed error information:

```bash
# View logs
tail -f logs/app.log
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

[Add your license information here]

## Support

For support and questions:

- Create an issue on GitHub
- Check the API documentation at `/docs`
- Review the logs for error details 