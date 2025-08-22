from typing import Tuple
import numpy as np
import torch
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from skmultilearn.model_selection import iterative_train_test_split
from torch.utils.data import TensorDataset
from config import *

"""Dataset creation utilities for combined learning."""

def create_datasets(device: torch.device) -> Tuple[TensorDataset, ...]:
    """
    Create and return processed datasets for semi-supervised learning.

    Args:
        device: Target device for tensor storage (cpu/cuda)

    Returns:
        Tuple containing:
        - Labeled training dataset
        - Unlabeled training dataset
        - Validation dataset
        - Test dataset
    """
    # Load raw data arrays
    train_features = np.load(FEATURE_TRAIN_PATH)
    test_features = np.load(FEATURE_TEST_PATH)
    train_labels = np.load(LABEL_TRAIN_PATH)
    test_labels = np.load(LABEL_TEST_PATH)

    # Add channel dimension (batch_size, channels, features)
    x_train = train_features[:, None, :]
    x_test = test_features[:, None, :]

    # Initial shuffle
    x_train, train_labels = shuffle(
        x_train,
        train_labels,
        random_state=RANDOM_STATE
    )

    # Stratified validation split
    x_train, y_train, x_val, y_val = iterative_train_test_split(
        x_train,
        train_labels,
        test_size=VALIDATION_SIZE
    )

    # Labeled/unlabeled split
    x_labeled, x_unlabeled, y_labeled, _ = train_test_split(
        x_train,
        y_train,
        train_size=LABELED_RATIO,
        random_state=RANDOM_STATE
    )

    # Tensor conversion function
    def to_tensor(data, dtype=torch.float32):
        return torch.tensor(data, dtype=dtype).to(device)

    # Create datasets
    return (
        TensorDataset(to_tensor(x_labeled), to_tensor(y_labeled)),
        TensorDataset(to_tensor(x_unlabeled)),
        TensorDataset(to_tensor(x_val), to_tensor(y_val)),
        TensorDataset(to_tensor(x_test), to_tensor(test_labels))
    )
