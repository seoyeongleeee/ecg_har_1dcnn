# Towards Real-World ECG-Based Human Activity Recognition: Optimal Window Size and Subject-Independent 1-D CNN Approach
Authors: Sunghan Lee, Seoyeong Lee, Suyeon Yun, Semin Ryu, In cheol Jeong

This repository contains the official implementation of the 1D convolutional neural network (1D-CNN) model used in our study on human activity recognition (HAR) using single-lead electrocardiogram (ECG) signals.

The provided code supports subject-independent 5-fold cross-validation and replicates the core experimental design presented in the manuscript.


## FILE OVERVIEW
'ECG-HAR.py': Full implementation of the 1D-CNN model with subject-wise 5-fold cross-validation using simulated data.


## MODEL OVERVIEW
- Architecture: Sequential 1D-CNN with SELU activation, LeCun normal initialization, batch normalization, and dropout.
- Input shape: (number of samples, 30000, 1) — where 30000 corresponds to a 60-second ECG segment sampled at 500 Hz.
- Output shape: (number of samples,) — representing the class label per segment.
- Number of classes: 5 (Sleep, Sit, Stairs, Walk, Run)
- Optimizer: Nadam (learning rate = 0.0001)
- Evaluation metrics: Accuracy, Precision, Recall, F1 Score, AUC

## PARAMETERS  
- Random seed: 825
- Epochs: 1000
- Batch size: 128
- Input length: 30000 (60 seconds at 500 Hz)
- Number of classes: 5
- Activation function: SELU
- Kernel initializer: LeCun normal
- Optimizer: Nadam (learning rate = 0.0001)
- Loss function: Categorical Crossentropy
- Evaluation metrics: Accuracy, Precision, Recall, F1 Score, AUC
- Pooling: Average pooling (kernel size = 3, stride = 2)
- Dropout: Progressive increase from 0.3 to 0.7
- Final pooling: GlobalAveragePooling1D


## USAGE
To run the code:
    python ECG-HAR.py
This script uses synthetic data for demonstration purposes.
Users should adapt the data loading section to match their own dataset.

'ECG-HAR.py' provides the full model and evaluation pipeline using sample data structure only.
Users must modify the code to fit their own ECG datasets, using the following expected format:
- ECG data: NumPy array of shape (number of samples, 30000, 1)  
- Labels: NumPy array of shape (number of samples,)

This code is provided to ensure transparency and to support reproducibility of the methodology.
