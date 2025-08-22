
import torch

"""Configuration for paths and training parameters."""

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(DEVICE)
# Paths
MODEL_SAVE_PATH = "/Users/poorvitiwary/Desktop/project/trained_models"
BASE_PATH = "/Users/poorvitiwary/Desktop/project/"
FEATURE_TRAIN_PATH = "/Users/poorvitiwary/Desktop/project/data/feature_train_mlc_v2.4.npy"
FEATURE_TEST_PATH = "/Users/poorvitiwary/Desktop/project/data/feature_test_mlc_v2.4.npy"
LABEL_TRAIN_PATH = "/Users/poorvitiwary/Desktop/project/data/label_train_mlc_v2.4.npy"
LABEL_TEST_PATH = "/Users/poorvitiwary/Desktop/project/data/label_test_mlc_v2.4.npy"

# Training parameters
RANDOM_STATE = 4
VALIDATION_SIZE = 0.15
# Available labeled ratios (pick one at a time) 
# the ratios belong to these labeled samples in the same order [42,85,170,254,532,658,849]
LABELED_RATIO_OPTIONS = [0.0318, 0.0645,0.129, 0.1928, 0.4039, 0.5, 0.645]

# Manually set the desired ratio (change this value when needed)
LABELED_RATIO = 0.0318  # Ratio of labeled data from training set

# Ratio to model size mapping
RATIO_TO_SIZE = {
    0.0318: 42,
    0.0645: 85,
    0.129: 170,
    0.1928: 254,
    0.4039: 532,
    0.5: 658,
    0.645: 849
}