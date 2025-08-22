from config import *
from metrics import *
from best_params import *
from models import *
from data import *
from utils import *
from tsai.all import *
import optuna
import torch
from fastai.callback.tracker import EarlyStoppingCallback

def objective(trial):
    """Optuna objective function for hyperparameter optimization."""
    # Suggest hyperparameters
    nf = trial.suggest_int("nf", 64, 256)
    fc_dropout = trial.suggest_float('fc_dropout', 0.1, 0.5)
    bs = trial.suggest_int("bs", 16, 48)
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    conv_dropout = trial.suggest_float("conv_dropout", 0.0, 0.5)
    ks = trial.suggest_int("ks", 20, 60)
    bottleneck = trial.suggest_categorical("bottleneck", [True, False])
    coord = trial.suggest_categorical("coord", [True, False])
    separable = trial.suggest_categorical("separable", [True, False])
    dilation = trial.suggest_int("dilation", 1, 5)

    # Prepare data
    (x_combined, _,
     _, _,
     _, _,
     splits, _) = prepare_data()

    # Create dataloaders
    udls100 = get_ts_dls(
        x_combined,
        splits=splits,
        bs=bs,
    )

    # Model configuration
    c_in = 1  # Number of input features (time series channels)
    c_out = 8  # Number of classes

    model = InceptionTimePlus(
        c_in=c_in,
        c_out=c_out,
        seq_len=920,
        nf=nf,
        fc_dropout=fc_dropout,
        conv_dropout=conv_dropout,
        ks=ks,
        bottleneck=bottleneck,
        coord=coord,
        separable=separable,
        dilation=dilation
    )

    # Training
    learn = ts_learner(
        udls100,
        model,
        cbs=[MVP(target_dir=MODEL_SAVE_PATH, fname='mvp_pretrain')]
    )
    learn.fit_one_cycle(100, lr_max=lr)

    # Get train loss
    train_loss = learn.recorder.values[-1][0]
    completed_trials = [
        t for t in trial.study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
    ]

    # Save the model if this trial has the lowest training loss
    best_value = (
        min([t.value for t in completed_trials])
        if completed_trials
        else float('inf')
    )
    model_save_path = f"best_model_trial_{trial.number}.pth"
    if train_loss < best_value:
        torch.save(model.state_dict(), model_save_path)
        print(
            f"Saved best model for trial {trial.number} "
            f"with train_loss: {train_loss:.4f}"
        )

    torch.cuda.empty_cache()
    save_checkpoint(trial.study, trial, model, trial.number)
    return train_loss


def run_optimization():
    """Run the Optuna optimization study."""
    study = optuna.create_study(
        storage="sqlite:///db.sqlite3",
        study_name="mvppretrain",
        load_if_exists=True,
        pruner=optuna.pruners.HyperbandPruner(),
        direction="minimize"
    )
    study.optimize(objective, n_trials=150)

    print(f"Best parameters: {study.best_params}")
    print(f"Best value: {study.best_value}")


if __name__ == "__main__":
    run_optimization()