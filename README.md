# Fundusnap AI - Retinal Analysis Models & API

## Overview
Fundusnap AI is a comprehensive retinal analysis system that combines:

1. **Diabetic Retinopathy Classification Model** - Deep learning model for classifying diabetic retinopathy stages from fundus images
2. **Retinal Artifacts Object Detection Model** - YOLO-based model for detecting and localizing retinal features and artifacts
3. **FastAPI Web Service** - RESTful API that currently serves the object detection model for real-time inference

This repository serves as both a model storage solution and an API service for retinal image analysis, designed to support the Fundusnap mobile application.

## Current API Functionality
The FastAPI web service currently provides:
- **Object Detection**: Real-time detection of retinal artifacts using YOLO models
- **JSON Response**: Structured detection results with bounding boxes and confidence scores  
- **Visual Output**: Annotated images with detection overlays
- **Health Monitoring**: API status and model availability endpoints

> **Note**: The diabetic retinopathy classification model is included in this repository but not currently exposed through the API. It serves as a fallback solution for offline classification capabilities.

## Technology Stack

### API Service (Currently Active)
- **Web Framework**: FastAPI 0.104.1
- **ASGI Server**: Uvicorn with standard extensions
- **Object Detection**: YOLO (Ultralytics 8.0.196)
- **Image Processing**: OpenCV 4.8.1, NumPy 1.24.3
- **Data Validation**: Pydantic 2.4.2
- **File Upload**: Python-multipart

### Classification Model (Repository Storage)
- **Deep Learning Framework**: FastAI
- **Base Architecture**: ResNet34 (pretrained)
- **Data Augmentation**: Albumentations
- **Loss Function**: Focal Loss
- **Training Data**: 2015-2019 diabetic retinopathy detection dataset

### Deployment
- **Containerization**: Docker with Python 3.10-slim base
- **Dependencies**: OpenCV system libraries, CUDA-ready environment
- **Port**: 8000 (HTTP)

## API Endpoints

### Object Detection Service
- **POST** `/inspect/fundus-artifacts/` - Returns JSON detection results
- **POST** `/visualize/fundus-artifacts/` - Returns annotated image with detections
- **GET** `/` - Health check and model status

### Example Usage
```bash
# JSON detection results
curl -X POST "http://localhost:8000/inspect/fundus-artifacts/" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@fundus_image.jpg"

# Visual detection overlay
curl -X POST "http://localhost:8000/visualize/fundus-artifacts/" \
     -H "accept: image/jpeg" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@fundus_image.jpg" \
     --output detection_result.jpg
```

## Diabetic Retinopathy Classification Model Performance

### Purpose
This classification model serves as a fallback solution for the Fundusnap mobile app when:
- The device is offline
- The primary Azure Custom Vision API server is unreachable
- Network connectivity is poor

### Performance Metrics
The model has been trained and evaluated on a dataset of fundus images, achieving the following performance:

#### Classification Results by Stage
| Stage | Precision | Recall | F1-Score | Support |
|-------|-----------|---------|-----------|----------|
| 0     | 0.63      | 0.65    | 0.64      | 1,013    |
| 1     | 0.71      | 0.67    | 0.69      | 1,032    |
| 2     | 0.75      | 0.76    | 0.76      | 954      |
| 3     | 0.98      | 0.98    | 0.98      | 1,003    |
| 4     | 0.97      | 0.99    | 0.98      | 998      |

#### Overall Performance
- **Accuracy**: 81%
- **Macro Average F1-Score**: 0.81
- **Weighted Average F1-Score**: 0.81

#### Performance Analysis
**Strengths:**
- Excellent detection of advanced stages (3 and 4) with F1-scores above 0.98
- Balanced performance across all classes with macro average F1-score of 0.81
- Consistent precision and recall metrics

**Areas for Improvement:**
- Moderate performance in early-stage detection (stage 0) with F1-score of 0.64
- Slightly lower precision and recall in stage 1 detection

## Deployment

### Using Docker
```bash
# Build the image
docker build -t fundusnap-ai .

# Run the container
docker run -p 8000:8000 fundusnap-ai

# Access the API
curl http://localhost:8000/
```

### Direct Python Execution
```bash
# Install dependencies
pip install -r requirements.txt

# Run the FastAPI server
uvicorn main:app --host 0.0.0.0 --port 8000
```

## Model Files
- `/models/fundus_artifacts.pt` - YOLO object detection model for retinal artifacts
- Classification model files are stored separately and not currently served by the API

## Integration
This API is designed to integrate with the Fundusnap mobile application for:
- Real-time retinal artifact detection and analysis
- Offline fallback classification capabilities
- Clinical decision support through automated image analysis
