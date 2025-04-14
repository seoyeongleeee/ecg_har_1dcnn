# Towards Real-World ECG-Based Human Activity Recognition: Optimal Window Size and Subject-Independent 1-D CNN Approach

**Authors:** Sunghan Lee, Seoyeong Lee, Suyeon Yun, Semin Ryu, In Cheol Jeong

This repository contains the official implementation of the 1D convolutional neural network (1D-CNN) model used in our study on human activity recognition (HAR) using single-lead electrocardiogram (ECG) signals.

The provided code supports subject-independent 5-fold cross-validation and replicates the core experimental design presented in the manuscript.



## FILE OVERVIEW
- `ECG-HAR.py`: Main script that implements the subject-independent 1D-CNN training and evaluation pipeline using 5-fold cross-validation. It loads preprocessed ECG segments and labels, builds the model, performs training, and saves results (confusion matrices and training curves).
  
- `preprocess.py`: Preprocessing pipeline that converts raw ECG text files into fixed-length segments suitable for model input. It includes filtering (notch + high-pass), resampling, segmentation, and labeling. The output is saved as `.npy` files per subject.

- `utils.py`: Collection of utility functions for signal filtering, visualization (e.g., confusion matrix and accuracy/loss plots), directory handling, and dataset statistics. It supports both preprocessing and evaluation phases.


## MODEL OVERVIEW
- Architecture: Sequential 1D-CNN with SELU activation, LeCun normal initialization, batch normalization, and dropout.
- Input shape: (number of samples, 30000, 1) — where 30000 corresponds to a 60-second ECG segment sampled at 500 Hz.
- Output shape: (number of samples,) — representing the class label per segment.
- Number of classes: 5 (Sleep, Sit, Stairs, Walk, Run)
- Optimizer: Nadam (learning rate = 0.0001)
- Evaluation metrics: Accuracy, Precision, Recall, F1 Score, AUC


## Dataset Format and Structure
The model requires subject-specific ECG recordings stored as plain text files. These raw ECG files must be preprocessed using the provided pipeline before training.
1. Expected Raw Data Format
    - Directory: All raw ECG text files should be placed in the ./raw_data/ directory.
    - Filename format: {subject_id}_ECG.txt (e.g., 101_ECG.txt)
    - Content:
        The first 11 lines contain metadata and are ignored during parsing.
        Each subsequent line must contain at least three comma-separated values, where the third column corresponds to the ECG value.
2. Preprocessing Steps
Run the process_all_subjects() function in preprocess.py to automatically:
    1) Load raw ECG files per subject.
    2) Apply signal filtering:
        - 60 Hz notch filter
        - 0.5 Hz high-pass filter
    3) Resample signals to 500 Hz.
    4) Segment the signal into 60-second windows (30,000 samples per segment).
    5) Label each segment based on the user-defined label parameter.
    6) Save preprocessed data into the ./data/ directory.
3. Output Files
After preprocessing, each subject will have the following files:
    - subject_{ID}.npy: ECG segments with shape (num_segments, 30000)
    - labels_{ID}.npy: Corresponding integer activity labels with shape (num_segments,)
These .npy files are automatically loaded by ECG-HAR.py for model training and evaluation.

## USAGE
This section outlines the typical workflow for preparing data and running the model.
1. Preprocess Raw ECG Files
Before training, you must convert the raw ECG .txt files into NumPy arrays using the preprocessing pipeline.
Edit and run the process_all_subjects() function in preprocess.py:
  ```python
  from preprocess import process_all_subjects
  process_all_subjects()
  ```

You can customize the following parameters inside the function:
- subject_list: List of subject IDs to process (e.g., [101, 102, 103])
- suffix: File suffix (default: _ECG.txt)
- target_rate: Target sampling frequency (default: 500)
- window_size: Number of samples per segment (default: 30000 for 60s)
After running this function, segmented ECG data and labels will be saved in the ./data/ directory as .npy files.

2. Run Training and Evaluation
Once preprocessing is complete, you can train and evaluate the model using subject-independent 5-fold cross-validation:
    python ECG-HAR.py
This script will:
- Automatically load the .npy files from ./data/
- Group subjects into 5 folds
- Normalize input signals using RobustScaler
- Train a 1D-CNN model for each fold
- Compute and print average performance metrics (Accuracy, Precision, Recall, F1-score, AUC)
- Save the following output for each fold:
    Confusion matrix: confmat_fold{X}.png
    Accuracy plot: acc_plot_fold{X}.png
    Loss plot: loss_plot_fold{X}.png 
Note: If you have more or fewer subjects than 40, adjust the NUM_SUBJECTS variable and related logic in ECG-HAR.py.


## Requirements
Dependencies are specified in `requirements.txt`.  
Install them using:

```bash
pip install -r requirements.txt
```


## Abstract
Human activity recognition (HAR) is essential for advancing healthcare, fitness, and patient monitoring because it provides critical insights into human physical movements. This study \textcolor{red}{proposes,} a novel one-dimensional convolutional neural network (1-D CNN) model to classify five everyday activities—Sleep, Sit, Stairs, Walk, and Run—using only electrocardiogram (ECG) signals. Data were collected from 40 healthy participants, and various window sizes (3–150 s) and sampling frequencies (125–625 Hz) to identify the optimal configuration. The proposed model achieved a classification accuracy of 93.2% ± 2.86% in a subject-independent validation scheme, with the best performance observed at a 60-s window size and 500 Hz sampling frequency. This approach exhibits competitive performance compared to other ECG-based HAR studies. Furthermore, this study incorporates a larger participant pool, addressing the limitations of previous research with small datasets and ensuring more robust and generalizable results. This research highlights the potential of ECG-based HAR systems for personalized health monitoring, real-time activity tracking, and rehabilitation, offering promising applications for wearable technologies and broader healthcare applications.
