from config import *
from callback import *
from models import *
from metrics import *
from best_params import *
from data import *
from utils import *
import optuna
from tsai.all import *
from fastai.data.core import DataLoaders
from fastai.callback.tracker import EarlyStoppingCallback
from tsai.data.preprocessing import TSStandardize
from tsai.data.all import TSTimeNoise, TSRandomSize

def objective(trial):
    """Optimization objective function for hyperparameter tuning.

    Args:
        trial: Optuna trial object

    Returns:
        float: The F1 score 
    """
    # Hyperparameter suggestions
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

    # Datasets
    labeled_dataset, unlabeled_dataset, valid_dataset, _ = create_datasets(DEVICE)
    
    # Define weak and strong augmentations
    weak_tfms = [TSStandardize(), TSTimeNoise(.01)]
    strong_tfms = [TSStandardize(), TSRandomSize(.7)]
    # DataLoaders
    labeled_dl = TfmdDL(labeled_dataset, batch_size=bs, shuffle=True)
    unlabeled_dl = TfmdDL(unlabeled_dataset, batch_size=bs, shuffle=True)
    valid_dl = TfmdDL(valid_dataset, batch_size=bs, shuffle=True)

    # Callback
    fixmatch_cb = FixMatchMultiLabel(
        dl2=unlabeled_dl,
        weak_tfms=weak_tfms,
        strong_tfms=strong_tfms,
        confidence_threshold=0.50,
        lambda_u=0.8,
        verbose=True
    )

    dls = DataLoaders(labeled_dl, valid_dl)

    # Model configuration
    c_in = 1
    c_out = 8
    seq_len = 920

    base_model = InceptionTimePlus(
        c_in=c_in,
        c_out=c_out,
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
    fixmatch_model = InceptionWithSigmoid(base_model=base_model)
    fixmatch_model.to(device)
    # Learner setup
    learn = ts_learner(
        dls,
        fixmatch_model,
        loss_func=BCELossFlat(),
        metrics=[precision_multi, f1_multi],
        cbs=[fixmatch_cb]
    )

    # Training
    learn.fit_one_cycle(
        n_epoch=100,
        lr_max=lr,
        cbs=[EarlyStoppingCallback(patience=40)]
    )

    torch.cuda.empty_cache()
    save_checkpoint(trial, fixmatch_model, trial.number)
    return learn.recorder.values[-1][3]

def run_optimization():
    study = optuna.create_study(
        storage="sqlite:///db.sqlite3",
        study_name="fixmatch",
        load_if_exists=True,
        pruner=optuna.pruners.HyperbandPruner(),
        direction="maximize"
    )
    study.optimize(objective, n_trials=150)

    print(f"Best parameters: {study.best_params}")
    print(f"Best value: {study.best_value}")


if __name__ == "__main__":
    run_optimization()