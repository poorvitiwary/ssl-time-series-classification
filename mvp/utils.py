from config import *
import os
import pickle
import optuna
import torch

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
    base_path='/home/hpc/iwfa/iwfa046h/faps_project/semisupervised/fixmatch'
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