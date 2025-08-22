from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from skmultilearn.model_selection import iterative_train_test_split
from config import *
import numpy as np
import torch

"""Dataset creation utilities for noisy learning."""

def prepare_data():
    """Prepare data with combined labeled set for teacher training.
    
    Returns:
        tuple: (
            X_labeled_combined,  # Labeled train + validation combined
            y_labeled_combined,
            X_unlabeled_train,   # Pure unlabeled data
            X_val,              # Validation features (same as in combined)
            y_val,              # Validation targets
            teacher_splits,      # Proper splits for combined data
            X_test,
            y_test
        )
    """
    # Load and reshape data (PEP 8 aligned)
    train_features = np.load(FEATURE_TRAIN_PATH)
    test_features = np.load(FEATURE_TEST_PATH)
    train_labels = np.load(LABEL_TRAIN_PATH)
    test_labels = np.load(LABEL_TEST_PATH)

    x_train = train_features[:, None, :]  # Add channel dim
    x_test = test_features[:, None, :]

    x_train, train_labels = shuffle(
        x_train,
        train_labels,
        random_state=RANDOM_STATE
    )
    # Create validation set from labeled data
    x_train_new, y_train_new, x_val, y_val = iterative_train_test_split(
        x_train,
        train_labels,
        test_size=VALIDATION_SIZE
    )
    # Initial split (labeled + unlabeled)
    x_labeled_train, x_unlabeled_train, y_labeled_train, _ = train_test_split(
        x_train_new, y_train_new,
        train_size=LABELED_RATIO,
        random_state=RANDOM_STATE
    )


    # Combine labeled train + validation for teacher
    x_labeled_combined = np.concatenate([x_labeled_train, x_val])
    y_labeled_combined = np.concatenate([y_labeled_train, y_val])

    # Create splits pointing to original validation in combined data
    train_size = len(x_labeled_train)
    teacher_splits = [
        list(range(train_size)),  # Labeled train indices
        list(range(train_size, len(x_labeled_combined)))  # Val indices
    ]

    # Convert to tensors (PEP 8 aligned)
    def to_tensor(data):
        return torch.tensor(data, dtype=torch.float32).to(DEVICE)

    return (
        to_tensor(x_labeled_combined),
        to_tensor(y_labeled_combined),
        to_tensor(x_unlabeled_train),
        to_tensor(x_val),
        to_tensor(y_val),
        teacher_splits,
        to_tensor(x_test),
        to_tensor(test_labels)
    )