# Fundusnap AI - Diabetic Retinopathy Classification

## Overview
Fundusnap AI is an offline-capable image classification system designed to detect and classify diabetic retinopathy stages from fundus images. This model serves as a fallback solution for the Fundusnap mobile application when it cannot connect to the primary Azure Custom Vision API server.

## Purpose
The primary purpose of this AI model is to ensure continuous functionality of the Fundusnap mobile app by providing local image classification capabilities when:
- The device is offline
- The API server is unreachable
- Network connectivity is poor

## Technology Stack
- **Deep Learning Framework**: FastAI
- **Base Model**: ResNet34 (pretrained)
- **Data Augmentation**: Albumentations
- **Loss Function**: Focal Loss
- **Training Data**: Resized fundus images from 2015-2019 diabetic retinopathy detection dataset
- **Metrics**: Accuracy, Precision, Recall, and F1 Score

## Model Performance
The model has been trained and evaluated on a dataset of fundus images, achieving the following performance metrics:

### Classification Metrics by Stage
| Stage | Precision | Recall | F1-Score | Support |
|-------|-----------|---------|-----------|----------|
| 0     | 0.63      | 0.65    | 0.64      | 1,013    |
| 1     | 0.71      | 0.67    | 0.69      | 1,032    |
| 2     | 0.75      | 0.76    | 0.76      | 954      |
| 3     | 0.98      | 0.98    | 0.98      | 1,003    |
| 4     | 0.97      | 0.99    | 0.98      | 998      |

### Overall Performance
- **Accuracy**: 81%
- **Macro Average F1-Score**: 0.81
- **Weighted Average F1-Score**: 0.81

## Performance Analysis
The model demonstrates strong performance as a fallback solution:

### Strengths
- Excellent performance in detecting advanced stages (3 and 4) with F1-scores above 0.98
- Balanced performance across all classes with a macro average F1-score of 0.81
- Consistent performance across precision and recall metrics

### Areas for Improvement
- Moderate performance in early-stage detection (stage 0) with an F1-score of 0.64
- Slightly lower precision and recall in stage 1 detection

## Conclusion
The model provides a reliable fallback solution with an overall accuracy of 81%. While it performs exceptionally well in detecting advanced stages of diabetic retinopathy, there is room for improvement in early-stage detection. The balanced performance across all metrics suggests that the model is well-suited for its role as a fallback system, providing consistent and reliable predictions when the primary API is unavailable.

## Usage
The model is integrated into the Fundusnap mobile application and automatically activates when:
1. The device is offline
2. The primary API server is unreachable
3. Network connectivity is poor

## Dataset
The model was trained on the resized 2015-2019 diabetic retinopathy detection dataset, with images resized for optimal processing while maintaining diagnostic quality.
