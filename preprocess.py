import os
import numpy as np
from scipy.signal import resample
from utils import filters_def


def preprocess_data(path, subject_list, label, suffix, target_rate, window_size, duration=60):
    """
    Load and preprocess raw ECG data for each subject and return segmented arrays.

    Parameters:
        path (str): Directory path containing raw ECG text files.
        subject_list (list): List of subject IDs to process.
        label (int): Integer label assigned to this activity.
        suffix (str): Suffix to match filenames (e.g., '_ECG.txt').
        target_rate (int): Target sampling rate (e.g., 500).
        window_size (int): Number of samples per segment (e.g., 30000 for 60s at 500Hz).
        duration (int): Duration in seconds to resample from (default: 60).

    Returns:
        X (list of np.ndarray): List of segmented ECG arrays
        Y (list of int): Corresponding list of activity labels
    """
    X, Y = [], []
    for subject_number in subject_list:
        for file_name in os.listdir(path):
            if file_name.endswith(suffix) and file_name.startswith(f"{subject_number}{suffix}"):
                file_path = os.path.join(path, file_name)
                with open(file_path, 'r') as f:
                    lines = f.readlines()[11:]  # Skip metadata lines
                try:
                    ecg_raw = [float(line.split(',')[2]) for line in lines]
                except Exception as e:
                    print(f"Failed to parse {file_name}: {e}")
                    continue
                ecg_resampled = resample(ecg_raw, target_rate * duration)
                ecg_filtered = filters_def(ecg_resampled, fs=target_rate)
                for i in range(len(ecg_filtered) // window_size):
                    start = i * window_size
                    end = (i + 1) * window_size
                    X.append(ecg_filtered[start:end])
                    Y.append(label)
    return X, Y


def save_subject_data(X, Y, subject_id, out_dir):
    """
    Save segmented ECG data and corresponding labels for a subject.

    Parameters:
        X (list or np.ndarray): List of segments
        Y (list or np.ndarray): List of labels
        subject_id (int): Subject identifier
        out_dir (str): Directory to save the output .npy files
    """
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, f"subject_{subject_id:03}.npy"), np.array(X))
    np.save(os.path.join(out_dir, f"labels_{subject_id:03}.npy"), np.array(Y))


# Optional wrapper (example usage)
def process_all_subjects():
    """
    Example high-level wrapper function to process and save multiple subjects.
    This is provided for reference and is not called automatically.
    """
    base_path = './raw_data/'
    output_path = './data/'
    subject_list = [101, 102, 103]  # example subject IDs
    activity_label = 0
    suffix = '_ECG.txt'
    target_rate = 500
    window_size = 30000  # 60s at 500Hz

    for subject_id in subject_list:
        X, Y = preprocess_data(
            path=base_path,
            subject_list=[subject_id],
            label=activity_label,
            suffix=suffix,
            target_rate=target_rate,
            window_size=window_size
        )
        save_subject_data(X, Y, subject_id, output_path)
