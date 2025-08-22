from config import *
from metrics import *
from best_params import *
from models import *
from callback import *
from data import *
from utils import *
from tsai.all import *
import optuna
import torch
import numpy as np
from torch import nn


def objective(trial):
    """Objective function for Optuna optimization."""
    # Suggest hyperparameters
    nf = trial.suggest_int("nf", 64, 256)
    fc_dropout = trial.suggest_uniform('fc_dropout', 0.1, 0.5)
    bs = trial.suggest_int("bs", 16, 48)
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    conv_dropout = trial.suggest_float("conv_dropout", 0.0, 0.5)
    ks = trial.suggest_int("ks", 20, 60)
    bottleneck = trial.suggest_categorical("bottleneck", [True, False])
    coord = trial.suggest_categorical("coord", [True, False])
    separable = trial.suggest_categorical("separable", [True, False])
    dilation = trial.suggest_int("dilation", 1, 5)
    temperature = trial.suggest_float('temperature', 0.5, 1.0, log=True)
    projection_dim = trial.suggest_categorical('projection_dim',
                                              [128, 256, 512])

    # DataLoader for Unsupervised Pretraining (SimCLR)
    # Prepare data
    (x_combined, _,
     _, _,
     _, _,
     _, _) = prepare_data()
    X = x_combined
    X_aug_1, X_aug_2 = create_time_series_augmentations(torch.tensor(X))

    # Stack the paired views for input to DataLoader
    X_paired = torch.cat([X_aug_1, X_aug_2], dim=0)
    y_dummy = np.zeros(len(X_paired))  # No labels needed for SimCLR
    dls_pretrain = get_ts_dls(
        X_paired.numpy(),
        y_dummy,
        bs=bs,
        shuffle_train=True,
        drop_last=True  # Drop the last incomplete batch
    )

    # DataLoader for Supervised Fine-Tuning
    c_in = 1  # Number of features/channels in the input data
    c_out = 8  # Number of classes (as you have specified)
    seq_len = 920

    # Define Encoder Model
    encoder = InceptionTimePlus(
        c_in=c_in,
        c_out=c_out,  # Assuming the number of classes is 8
        seq_len=seq_len,
        nf=nf,
        fc_dropout=fc_dropout,
        conv_dropout=conv_dropout,
        ks=ks,
        bottleneck=bottleneck,
        coord=coord,
        separable=separable,
        dilation=dilation
    )
    dummy_loss = nn.MSELoss()
    torch.cuda.empty_cache()

    # Initialize Model and Learner for Pretraining
    model = SimCLRModel(
        encoder=encoder,
        projection_dim=projection_dim
    )

    learn = ts_learner(
        dls_pretrain,
        model,
        loss_func=dummy_loss,
        cbs=[SimCLRCallback(temperature=temperature)],
        metrics=None
    )

    # Train with Contrastive Loss (Unsupervised Pretraining)
    learn.fit_one_cycle(n_epoch=100, lr_max=lr)

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

    model_save_path = f"simclr_pretrain.pth"
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
        storage="sqlite:///db.sqlite001",
        study_name="simclrpretrain_001",
        load_if_exists=True,
        pruner=optuna.pruners.HyperbandPruner(),
        direction="minimize"
    )
    study.optimize(objective, n_trials=150)

    print(f"Best parameters: {study.best_params}")
    print(f"Best value: {study.best_value}")


if __name__ == "__main__":
    run_optimization()