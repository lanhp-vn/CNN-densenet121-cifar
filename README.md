# CNN DenseNet121 on CIFAR-10

A comprehensive deep learning project implementing DenseNet121 architecture for image classification on the CIFAR-10 dataset with extensive hyperparameter exploration and model analysis.

## Project Overview

This project contains two complementary notebooks that explore CNN architectures using DenseNet121 for CIFAR-10 classification:

1. **DenseNet_training.ipynb**: Main implementation with data augmentation and transfer learning
2. **DenseNet_explore.ipynb**: Comprehensive exploration of different architectures, loss functions, and optimizers

## Dataset

- **CIFAR-10**: 60,000 32x32 color images in 10 classes
- **Training set**: 40,000 images (80% of training data)
- **Validation set**: 10,000 images (20% of training data)  
- **Test set**: 10,000 images

## Model Architectures

### Main Training Model (DenseNet_training.ipynb)
- **Base Model**: DenseNet121 with ImageNet weights (frozen features)
- **Custom Layers**: Global Average Pooling + 2 Dense layers (1024 units each) + Dropout (0.2) + Output layer (10 classes)
- **Optimizer**: SGD with learning rate 0.5
- **Loss Function**: Binary Crossentropy
- **Data Augmentation**: Rotation (20°), width/height shift (0.2), horizontal flip
- **Training**: 50 epochs, batch size 256

### Exploration Models (DenseNet_explore.ipynb)
Four different configurations tested:
1. **Base Model**: 2×1024 dense layers, dropout 0.25, SGD lr=0.05, binary crossentropy
2. **Expanded Architecture**: Additional 4×512 dense layers with extra dropout
3. **Loss Function Change**: Switched to categorical crossentropy
4. **Optimizer Change**: Switched to Adam optimizer (failed due to high learning rate)

## Results

### Main Training Model Performance (DenseNet_training.ipynb)
- **Final Training Accuracy**: 96.40%
- **Final Training Loss**: 0.0198
- **Final Validation Accuracy**: 86.17%
- **Final Validation Loss**: 0.0889
- **Test Accuracy**: 85.69%
- **Test Loss**: 0.0967
- **Correct Predictions**: 8,569 out of 10,000

### Exploration Model Comparison (DenseNet_explore.ipynb)

| Model Configuration | Learning Rate | Training Accuracy | Test Accuracy |
|---------------------|---------------|-------------------|---------------|
| **Base Model** (2×1024 dense + dropout 0.25) | 0.05 (SGD) | ~99.6% | ~77.8% |
| **Expanded Architecture** (Extra 4×512 layers) | 0.05 (SGD) | ~99.5% | ~76.8% |
| **Categorical Crossentropy** (Base model) | 0.05 (SGD) | ~99.2% | ~78.0% |
| **Adam Optimizer** (Base model) | 0.05 (Adam) | ~10% | ~10% (failed) |

### Key Findings

1. **Main training model achieved excellent performance**: 85.69% test accuracy with proper data augmentation
2. **Exploration models showed overfitting**: Training accuracies near 99% but lower test performance (76-78%)
3. **Architecture expansion didn't help**: Additional layers reduced generalization
4. **Learning rate critical for Adam**: High learning rate (0.05) prevented convergence
5. **Data augmentation improved generalization**: Main model with augmentation outperformed exploration models

## Performance Summary

| Metric | Main Model | Best Exploration Model |
|--------|------------|----------------------|
| Training Accuracy | 96.40% | ~99.6% |
| Test Accuracy | **85.69%** | 78.0% |
| Training Loss | 0.0198 | N/A |
| Test Loss | 0.0967 | N/A |
| Generalization Gap | 10.71% | ~21.6% |

## Files

- `DenseNet_training.ipynb`: Main implementation with data augmentation and optimal training
- `DenseNet_explore.ipynb`: Comprehensive exploration of different model configurations

## Notebook Contents

### DenseNet_training.ipynb
- Complete CIFAR-10 classification pipeline
- DenseNet121 transfer learning implementation
- Data augmentation with ImageDataGenerator
- 50-epoch training with SGD optimizer
- Model evaluation and visualization
- **Achieves 85.69% test accuracy**

### DenseNet_explore.ipynb
- Comparative analysis of 4 different model configurations
- Hyperparameter tuning experiments
- Loss function comparison (binary vs categorical crossentropy)
- Optimizer comparison (SGD vs Adam)
- Architecture modification experiments
- Performance analysis and conclusions

## Requirements

- TensorFlow
- Keras
- NumPy
- Matplotlib
- scikit-learn
- tensorflow-datasets
