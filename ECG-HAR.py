import os
import time
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv1D, AvgPool1D, Dropout, GlobalAveragePooling1D,
    Dense, BatchNormalization
)
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.utils import to_categorical
from utils import plot_confusion_matrix, plot_training_curves

# Set random seed for reproducibility
SEED = 825
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Hyperparameters
EPOCHS = 1000
BATCH_SIZE = 128
INPUT_LENGTH = 30000  # ECG segment length (60s at 500Hz)
NUM_CLASSES = 5
NUM_SUBJECTS = 40

# Load ECG data
def load_data():
    """
    Load preprocessed segment-level ECG data.
    Must return:
        X_subjects (list of ndarray): (segments, 30000)
        Y_subjects (list of ndarray): (segments,)
    """
    raise NotImplementedError("Implement this with np.load calls from your saved directory.")

# Load your data here
X_subjects, Y_subjects = load_data()

# Group subjects into 5 folds (subject-independent)
group_names = ['A', 'B', 'C', 'D', 'E']
subject_indices = np.arange(NUM_SUBJECTS)
np.random.shuffle(subject_indices)
group_subjects = {g: subject_indices[i*8:(i+1)*8] for i, g in enumerate(group_names)}


def build_model(input_shape=(INPUT_LENGTH, 1), num_classes=NUM_CLASSES, seed=SEED):
    initializer = tf.keras.initializers.LecunNormal(seed=seed)
    model = Sequential(name="ECG_HAR_1DCNN")

    model.add(Conv1D(32, 9, padding='same', activation='selu', kernel_initializer=initializer, input_shape=input_shape))
    model.add(AvgPool1D(3, strides=2))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    model.add(Conv1D(48, 12, padding='same', activation='selu', kernel_initializer=initializer))
    model.add(AvgPool1D(3, strides=2))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv1D(128, 16, padding='same', activation='selu', kernel_initializer=initializer))
    model.add(AvgPool1D(3, strides=2))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Conv1D(192, 18, padding='same', activation='selu', kernel_initializer=initializer))
    model.add(AvgPool1D(3, strides=2))
    model.add(BatchNormalization())
    model.add(Dropout(0.6))

    model.add(Conv1D(256, 24, padding='same', activation='selu', kernel_initializer=initializer))
    model.add(AvgPool1D(3, strides=2))
    model.add(BatchNormalization())
    model.add(Dropout(0.7))

    model.add(Conv1D(54, 9, padding='same', kernel_initializer=initializer))
    model.add(GlobalAveragePooling1D())
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax', kernel_initializer=initializer))

    model.compile(optimizer=Nadam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# Perform subject-independent 5-fold cross-validation
results = {'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'auc': []}

for fold, test_group in enumerate(group_names, 1):
    print(f"\n[Fold {fold}] Test group: {test_group}")
    test_indices = group_subjects[test_group]
    train_indices = np.concatenate([v for k, v in group_subjects.items() if k != test_group])

    X_train = np.concatenate([X_subjects[i] for i in train_indices], axis=0)
    Y_train = np.concatenate([Y_subjects[i] for i in train_indices], axis=0)
    X_test = np.concatenate([X_subjects[i] for i in test_indices], axis=0)
    Y_test = np.concatenate([Y_subjects[i] for i in test_indices], axis=0)

    # Normalize and reshape
    scaler = RobustScaler()
    X_train = scaler.fit_transform(X_train).reshape(-1, INPUT_LENGTH, 1)
    X_test = scaler.transform(X_test).reshape(-1, INPUT_LENGTH, 1)

    Y_train_cat = to_categorical(Y_train, NUM_CLASSES)
    Y_test_cat = to_categorical(Y_test, NUM_CLASSES)

    # Train model
    model = build_model()
    history = model.fit(
        X_train, Y_train_cat,
        validation_data=(X_test, Y_test_cat),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=0
    )

    # Evaluate
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)

    acc = accuracy_score(Y_test, y_pred_classes)
    prec = precision_score(Y_test, y_pred_classes, average='weighted')
    rec = recall_score(Y_test, y_pred_classes, average='weighted')
    f1 = f1_score(Y_test, y_pred_classes, average='weighted')
    auc = roc_auc_score(Y_test_cat, y_pred, average='weighted', multi_class='ovr')

    results['accuracy'].append(acc)
    results['precision'].append(prec)
    results['recall'].append(rec)
    results['f1'].append(f1)
    results['auc'].append(auc)

    print(f"ACC: {acc:.4f} | PRE: {prec:.4f} | REC: {rec:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")

    # Save confusion matrix and training curves for each fold
    plot_confusion_matrix(
        y_true=Y_test,
        y_pred=y_pred_classes,
        labels=list(range(NUM_CLASSES)),
        class_names=['Sleep', 'Sit', 'Stairs', 'Walk', 'Run'],
        save_path=f'confmat_fold{fold}.png'
    )
    plot_training_curves(
        history,
        acc_path=f'acc_plot_fold{fold}.png',
        loss_path=f'loss_plot_fold{fold}.png'
    )

# Print final results
print("\n===== Final 5-Fold Results =====")
for k, v in results.items():
    print(f"{k.upper():<10}: {np.mean(v):.4f} ± {np.std(v):.4f}")
