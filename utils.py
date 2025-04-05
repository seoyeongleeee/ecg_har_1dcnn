import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from datetime import datetime
from scipy.signal import butter, iirnotch, filtfilt
from sklearn.metrics import confusion_matrix

# =======================================
# Filtering: Apply notch and high-pass filter
# =======================================
def apply_filters(ecg_data, fs=500, notch_freq=60, highpass_freq=0.5):
    """
    Apply notch filter and high-pass filter to ECG signal.

    Parameters:
        ecg_data (np.ndarray): 1D ECG time series signal
        fs (int): Sampling frequency in Hz
        notch_freq (float): Frequency to remove via notch filter (default: 60Hz)
        highpass_freq (float): High-pass filter cutoff frequency (default: 0.5Hz)

    Returns:
        ecg_filtered (np.ndarray): Filtered ECG signal
    """
    b_notch, a_notch = iirnotch(notch_freq, 30, fs)
    ecg_notched = filtfilt(b_notch, a_notch, ecg_data)

    b_highpass, a_highpass = butter(1, highpass_freq / (0.5 * fs), btype='high')
    ecg_filtered = filtfilt(b_highpass, a_highpass, ecg_notched)
    return ecg_filtered


# =======================================
# Directory Creation: Timestamped output folder
# =======================================
def create_timestamped_dir(base_dir):
    """
    Create a new directory with current timestamp inside the base directory.

    Parameters:
        base_dir (str): Path to the base directory

    Returns:
        new_dir (str): Path to the newly created timestamped directory
    """
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    new_dir = os.path.join(base_dir, current_time)
    os.makedirs(new_dir, exist_ok=True)
    return new_dir


# =======================================
# Confusion Matrix Visualization
# =======================================
def plot_confusion_matrix(y_true, y_pred, labels, class_names, figsize=(17, 14), title="Confusion Matrix", save_path=None):
    """
    Plot and annotate confusion matrix with both counts and percentage.

    Parameters:
        y_true (array-like): Ground truth labels
        y_pred (array-like): Predicted labels
        labels (list): List of label indices
        class_names (list): Class name strings (used for axis labels)
        figsize (tuple): Figure size
        title (str): Title for the plot
        save_path (str): Optional file path to save the figure
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100

    annot = np.empty_like(cm).astype(str)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            c = cm[i, j]
            p = cm_perc[i, j]
            annot[i, j] = f'{c}\n{p:.1f}%'

    cm_df = pd.DataFrame(cm_perc, index=class_names, columns=class_names)

    plt.figure(figsize=figsize)
    sns.heatmap(cm_df, annot=annot, fmt='', cmap='Blues', vmin=0, vmax=100)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    if save_path:
        plt.savefig(save_path)
    plt.show()


# =======================================
# Accuracy and Loss Plotting
# =======================================
def plot_training_curves(history, acc_path, loss_path):
    """
    Plot training and validation accuracy/loss curves.

    Parameters:
        history: Keras training history object
        acc_path (str): File path to save accuracy plot
        loss_path (str): File path to save loss plot
    """
    plt.figure()
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.savefig(acc_path)
    plt.close()

    plt.figure()
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.savefig(loss_path)
    plt.close()


# =======================================
# Class Distribution Summary
# =======================================
def summarize_class_shapes(X, Y, class_labels):
    """
    Print and return the shape of input data for each class.

    Parameters:
        X (np.ndarray): Input data array
        Y (np.ndarray): One-hot encoded label array
        class_labels (dict): Dictionary mapping class name to label index

    Returns:
        class_shapes (dict): Dictionary with class name as key and shape as value
    """
    class_shapes = {}
    for class_name, class_index in class_labels.items():
        indices = np.where(np.argmax(Y, axis=1) == class_index)
        class_data = X[indices]
        class_shapes[class_name] = class_data.shape
        print(f"Class '{class_name}' shape: {class_data.shape}")
    return class_shapes
