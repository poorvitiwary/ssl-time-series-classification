from config import *
from metrics import *
from best_params import *
from data import *
from models import *
from utils import *
from tsai.all import *
import optuna
from torch.nn import BCEWithLogitsLoss
from tsai.data.preprocessing import TSStandardize
from tsai.data.all import TSRandomSize

def objective(trial):
    """Optimization objective function for hyperparameter tuning.

    Args:
        trial: Optuna trial object

    Returns:
        float: The F1 score from the last epoch of student model training
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
    l2pl_ratio = trial.suggest_int('l2pl_ratio', 2, 5)

    c_in = 1  # Number of features/channels in the input data
    c_out = 8  # Number of classes (as you have specified)
    seq_len = 920

    # Get data (now with combined labeled set)
    (x_labeled_combined, y_labeled_combined,
     x_unlabeled, x_val, y_val,
     teacher_splits, x_test, y_test) = prepare_data()


    dls_teacher = get_ts_dls(
        x_labeled_combined,
        y_labeled_combined,
        splits=teacher_splits,
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

    # Step 2: Train the Teacher Model for Multilabel
    teacher_model = encoder
    learn = ts_learner(
        dls_teacher,
        teacher_model,
        metrics=[precision_multi, f1_multi],
        loss_func=BCEWithLogitsLoss()
    )
    learn.fit_one_cycle(n_epoch=100, lr_max=lr)

    # Create TSDatasets and TSDataLoader for the unlabeled data without labels
    unlabeled_ds = TSDatasets(X_unlabeledtrain, None)
    udls = TSDataLoader(unlabeled_ds, shuffle=False)

    # Step 3: Generate Pseudo-Labels using the Teacher Model
    # Use the teacher model to predict on the unlabeled data
    pseudo_labels, _ = learn.get_preds(dl=udls)
    pseudo_labels = (pseudo_labels > 0.5).float()

    dsets_pseudo = TSDatasets(X_unlabeledtrain, pseudo_labels)
    dl_pseudo = TSDataLoader(dsets_pseudo, shuffle=True)
    # Fine-tune
    # Initialize the NoisyStudent callback
    noisy_student_cb = NoisyStudent(
        dl_pseudo,
        bs=bs,
        l2pl_ratio=l2pl_ratio,
        verbose=True
    )

    X_labeled_combined = (
        X_labeled_combined.cpu().numpy()
        if isinstance(X_labeled_combined, torch.Tensor)
        else X_labeled_combined
    )

    # Handle y_labeledtrain and pseudo_labels similarly
    y_labeled_combined = (
        y_labeled_combined.cpu().numpy()
        if isinstance(y_labeled_combined, torch.Tensor)
        else y_labeled_combined
    )

    # Step 4: Create DataLoaders for Combined Data
    dls_student = get_ts_dls(x_labeled_combined, 
        y_labeled_combined,
        teacher_splits
    )

    # Combine labeled and pseudo-labeled data
    student_model = Inceptionwithsigmoid(base_model=encoder)
    learn_student = ts_learner(
        dls_student,
        student_model,
        metrics=[precision_multi, f1_multi],
        loss_func=BCELossFlat(),
        cbs=noisy_student_cb,
        batch_tfms=[TSStandardize(), TSRandomSize(.5)]
    )

    # Train the student model
    learn_student.fit_one_cycle(
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
        study_name="noisystudent",
        load_if_exists=True,
        pruner=optuna.pruners.HyperbandPruner(),
        direction="maximize"
    )
    study.optimize(objective, n_trials=150)

    print(f"Best parameters: {study.best_params}")
    print(f"Best value: {study.best_value}")


if __name__ == "__main__":
    run_optimization()