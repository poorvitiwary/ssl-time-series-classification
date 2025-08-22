from config import *
from metrics import *
from best_params import *
from data import *
from models import *
from tsai.all import *
from torch.nn import BCEWithLogitsLoss
import numpy as np
import os
from tsai.data.preprocessing import TSStandardize
from tsai.data.all import TSRandomSize

"""Training module to train with best parameters post optimization."""

def train_model():
    
    best_params = BEST_PARAMS[LABELED_RATIO]
    params = best_params.copy()

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
        bs=params["bs"]
    )

    # Step 4: Define Encoder Model
    encoder = InceptionTimePlus(
        c_in=c_in,
        c_out=c_out,
        seq_len=seq_len,
        nf=params['nf'],
        fc_dropout=params['fc_dropout'],
        conv_dropout=params['conv_dropout'],
        ks=params['ks'],
        bottleneck=params['bottleneck'],
        coord=params['coord'],
        separable=params['separable'],
        dilation=params['dilation']
    )
    torch.cuda.empty_cache()

    # Step 2: Train the Teacher Model for Multilabel
    teacher_model = encoder
    learn = ts_learner(
        dls_teacher,
        teacher_model,
        metrics=[precision_multi, F1_multi],
        loss_func=BCEWithLogitsLoss()
    )
    learn.fit_one_cycle(n_epoch=100, lr_max=params['lr'])

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
        bs=params["bs"],
        l2pl_ratio=params['l2pl_ratio'],
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
        metrics=[precision_multi, F1_multi],
        loss_func=BCELossFlat(),
        cbs=noisy_student_cb,
        batch_tfms=[TSStandardize(), TSRandomSize(.5)]
    )

    # Train the student model
    learn_student.fit_one_cycle(
        n_epoch=100,
        lr_max=params['lr'],
        cbs=[EarlyStoppingCallback(patience=40)])
    
    model_filename = f"noisy_model_{RATIO_TO_SIZE[LABELED_RATIO]}.pth"
    full_model_path = os.path.join(os.path.dirname(MODEL_SAVE_PATH), model_filename)

    torch.save({
        'model_state_dict': learn_student.model.state_dict(),
        'params': params
    }, full_model_path)

    print(f"Model saved to {full_model_path}")


if __name__ == "__main__":
    train_model()