# CNN DenseNet121 on CIFAR-10

A deep learning project implementing DenseNet121 architecture for image classification on the CIFAR-10 dataset.

## Project Overview

This project implements a Convolutional Neural Network using DenseNet121 architecture to classify images from the CIFAR-10 dataset. The model is trained with data augmentation and transfer learning techniques.

## Dataset

- **CIFAR-10**: 60,000 32x32 color images in 10 classes
- **Training set**: 40,000 images (80% of training data)
- **Validation set**: 10,000 images (20% of training data)  
- **Test set**: 10,000 images

## Model Architecture

- **Base Model**: DenseNet121 with ImageNet weights
- **Custom Layers**: Global Average Pooling + 2 Dense layers (1024 units) + Dropout (0.2) + Output layer (10 classes)
- **Optimizer**: SGD with learning rate 0.5
- **Loss Function**: Binary Crossentropy
- **Data Augmentation**: Rotation, width/height shift, horizontal flip

## Results

### Training Performance
- **Epochs**: 50
- **Final Training Accuracy**: 90.32%
- **Final Training Loss**: 0.0495

### Validation Performance  
- **Final Validation Accuracy**: 80.88%
- **Final Validation Loss**: 0.1051

### Test Performance
- **Test Accuracy**: 85.69%
- **Test Loss**: 0.0967
- **Correct Predictions**: 8,569 out of 10,000

## Key Metrics

| Metric | Value |
|--------|-------|
| Training Accuracy | 90.32% |
| Validation Accuracy | 80.88% |
| Test Accuracy | 85.69% |
| Training Loss | 0.0495 |
| Validation Loss | 0.1051 |
| Test Loss | 0.0967 |

## Files

- `main.ipynb`: Main Jupyter notebook containing the complete implementation
- `README.md`: This project documentation

## Requirements

- TensorFlow
- Keras
- NumPy
- Matplotlib
- scikit-learn
- tensorflow-datasets
