from config import *
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from skmultilearn.model_selection import iterative_train_test_split

"""Dataset creation utilities for simclr learning."""

def prepare_data():
    """Prepare and split data for training and validation.
    
    Args:
        features_train: Training features array
        features_test: Test features array
        labels_train: Training labels array
        
    Returns:
        Tuple containing all prepared data splits in this order:
        (x_train, x_test, x_labeled_train, x_unlabeled_train,
         y_labeled_train, x_val, y_val, x_labeled_combined,
         y_labeled_combined, splits, downstream_splits)
    """
    train_features = np.load(FEATURE_TRAIN_PATH)
    test_features = np.load(FEATURE_TEST_PATH)
    train_labels = np.load(LABEL_TRAIN_PATH)
    test_labels = np.load(LABEL_TEST_PATH)

    # Reshape features to add a new axis
    x_train = train_features[:, None, :]
    x_test = test_features[:, None, :]
    y_test = test_labels
    # Shuffle the training data
    x_train, train_labels = shuffle(
        x_train, train_labels, random_state=RANDOM_STATE
    )

    # Split into new train and validation sets
    x_train_new, y_train_new, x_val, y_val = iterative_train_test_split(
        x_train, train_labels, test_size=VALIDATION_SIZE
    )

    # Split into labeled and unlabeled training data
    x_labeled_train, _, y_labeled_train, _ = train_test_split(
        x_train_new,
        y_train_new,
        train_size=LABELED_RATIO,
        random_state=RANDOM_STATE
    )

    # Combine all data for unified processing
    x_combined = np.concatenate([x_train_new, x_val])
    y_combined = np.concatenate([y_train_new, y_val])

    # Combine labeled train + validation for teacher
    x_labeled_combined = np.concatenate([x_labeled_train, x_val])
    y_labeled_combined = np.concatenate([y_labeled_train, y_val])

    # Create splits for TSDatasets
    train_len = len(x_train_new)
    total_len = len(x_combined)
    splits = (list(range(train_len)), list(range(train_len, total_len)))

    # Create downstream splits
    train_size = len(x_labeled_train)
    downstream_splits = [
        list(range(train_size)),  # Labeled train indices
        list(range(train_size, len(x_labeled_combined)))  # Val indices
    ]

    return (
        x_combined,
        y_combined,
        x_labeled_combined,
        y_labeled_combined,
        x_test,
        y_test,
        splits,
        downstream_splits
    )