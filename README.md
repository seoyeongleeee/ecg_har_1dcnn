# Towards Real-World ECG-Based Human Activity Recognition: Optimal Window Size and Subject-Independent 1-D CNN Approach

**Authors:** Sunghan Lee, Seoyeong Lee, Suyeon Yun, Semin Ryu, In Cheol Jeong

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


## Abstract
Human activity recognition (HAR) is essential for advancing healthcare, fitness, and patient monitoring because it provides critical insights into human physical movements. This study \textcolor{red}{proposes,} a novel one-dimensional convolutional neural network (1-D CNN) model to classify five everyday activities—Sleep, Sit, Stairs, Walk, and Run—using only electrocardiogram (ECG) signals. Data were collected from 40 healthy participants, and various window sizes (3–150 s) and sampling frequencies (125–625 Hz) to identify the optimal configuration. The proposed model achieved a classification accuracy of 93.2% ± 2.86% in a subject-independent validation scheme, with the best performance observed at a 60-s window size and 500 Hz sampling frequency. This approach exhibits competitive performance compared to other ECG-based HAR studies. Furthermore, this study incorporates a larger participant pool, addressing the limitations of previous research with small datasets and ensuring more robust and generalizable results. This research highlights the potential of ECG-based HAR systems for personalized health monitoring, real-time activity tracking, and rehabilitation, offering promising applications for wearable technologies and broader healthcare applications.
