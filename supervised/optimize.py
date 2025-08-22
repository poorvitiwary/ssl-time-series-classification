from config import *
from metrics import *
from best_params import *
from models import *
from utils import *
from data import prepare_data
from tsai.all import *
import optuna
from fastai.callback.tracker import EarlyStoppingCallback

def objective(trial):
    """Optimization objective function for hyperparameter tuning.

    Args:
        trial: Optuna trial object

    Returns:
        float: The F1 score 
    """
    # Suggest hyperparameters
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    nf = trial.suggest_int("nf", 64, 256)
    bs = trial.suggest_categorical("bs_callback", [16, 48, 64, 128])
    fc_dropout = trial.suggest_uniform('fc_dropout', 0.1, 0.5)
    conv_dropout = trial.suggest_float('conv_dropout', 0.1, 0.5)
    ks = trial.suggest_int("ks", 20, 60)
    bottleneck = trial.suggest_categorical('bottleneck', [True, False])
    coord = trial.suggest_categorical('coord', [True, False])
    separable = trial.suggest_categorical('separable', [True, False])
    dilation = trial.suggest_int('dilation', 1, 4)

    c_in = 1  # Number of features/channels in the input data
    c_out = 8  # Number of classes (as you have specified)
    seq_len = 920

    # Get data (now with combined labeled set)
    (x_labeled_combined, y_labeled_combined,
     _, _, _,
     splits, _, _) = prepare_data()


    dls = get_ts_dls(
        x_labeled_combined,
        y_labeled_combined,
        splits=splits,
        bs=bs
    )

    # Step 4: Define Encoder Model
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
    torch.cuda.empty_cache()

    # Combine labeled and pseudo-labeled data
    model = InceptionWithSigmoid(base_model=encoder)
    learn = ts_learner(
        dls,
        model,
        metrics=[precision_multi, f1_multi],
        loss_func=BCELossFlat()
    )

    # Train the student model
    learn.fit_one_cycle(
        n_epoch=100,
        lr_max=lr,
        cbs=[EarlyStoppingCallback(patience=40)]
    )

    torch.cuda.empty_cache()
    save_checkpoint(study, trial, student_model, trial.number)
    return learn_student.recorder.values[-1][3]


def run_optimization():
    """Run the hyperparameter optimization study."""
    study = optuna.create_study(
        storage="sqlite:///db.sqlite3",
        study_name="supervised",
        load_if_exists=True,
        pruner=optuna.pruners.HyperbandPruner(),
        direction="maximize"
    )
    study.optimize(objective, n_trials=150)

    print(f"Best parameters: {study.best_params}")
    print(f"Best value: {study.best_value}")


if __name__ == "__main__":
    run_optimization()