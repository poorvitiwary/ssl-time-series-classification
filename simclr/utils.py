from config import *
import os
import pickle
import optuna
import torch
from tsai.data.all import TSTimeNoise

def save_checkpoint(
    study,
    trial,
    model,
    trial_id,
    base_path=BASE_PATH
):
    """Save study and model checkpoints.

    Args:
        study: Optuna study object
        trial: Current trial object
        model: Model to save
        trial_id: ID of current trial
        base_path: Base directory for saving files
    """
    # Ensure base directory exists
    os.makedirs(base_path, exist_ok=True)

    # Check if any trial has completed successfully
    completed_trials = [
        t for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
    ]

    if completed_trials:
        # Save the study
        study_path = os.path.join(base_path, 'study_noisy.pkl')
        with open(study_path, 'wb') as f:
            pickle.dump(study, f)

        # Save the model weights for this trial if it's the best trial so far
        if study.best_trial.number == trial.number:
            model_path = os.path.join(
                base_path,
                f'best_model_trial_{trial_id}.pth'
            )
            torch.save(model.state_dict(), model_path)

        print(f"Checkpoint saved for trial {trial_id}")
    else:
        print("No completed trials yet. Checkpoint not saved.")


def load_study(
    study_name,
    storage,
    base_path=BASE_PATH
):
    """Load or create an Optuna study.

    Args:
        study_name: Name of the study
        storage: Database storage URL
        base_path: Base directory for study files

    Returns:
        Optuna study object
    """
    study_path = os.path.join(base_path, 'study_noisy.pkl')

    if os.path.exists(study_path):
        with open(study_path, 'rb') as f:
            study = pickle.load(f)
        print("Resumed study from checkpoint.")
    else:
        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            load_if_exists=True
        )
        print("Created a new study.")

    return study


def create_time_series_augmentations(X, n_augmentations=2):
    """Create pairs of augmented time series for contrastive learning.

    Args:
        X: Input time series data (torch.Tensor)
        n_augmentations: Number of augmentation pairs to create (default: 2)

    Returns:
        Tuple of (aug_data_1, aug_data_2) containing paired augmentations
    """
    aug_data_1 = []
    aug_data_2 = []

    for _ in range(n_augmentations):
        # Create two independent augmentations for each input
        time_noised_1 = TSTimeNoise(magnitude=0.1)(X)
        time_noised_2 = TSTimeNoise(magnitude=0.1)(X)

        aug_data_1.append(time_noised_1)
        aug_data_2.append(time_noised_2)

    # Stack paired augmentations
    return torch.cat(aug_data_1, dim=0), torch.cat(aug_data_2, dim=0)