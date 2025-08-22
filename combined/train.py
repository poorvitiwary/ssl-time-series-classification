from best_params import *
from models import *
from callback import *
from config import *
from metrics import *
from best_params import *
from data import *
from tsai.all import *
import os
from fastai.data.core import DataLoaders
from fastai.callback.tracker import EarlyStoppingCallback

"""Training module to train with best parameters post optimization."""

def train_model():
    """
    Train a model using the best parameters from best_params.py

    Args:
        model_save_path: Path to save the trained model

    Returns:
        Trained model and learner objects
    """
    # Use the imported best_params dictionary
    best_params = BEST_PARAMS
    params = best_params.copy()

    # Datasets
    labeled_dataset, unlabeled_dataset, valid_dataset, _ = create_datasets(DEVICE)

    # DataLoaders
    labeled_dl = TfmdDL(labeled_dataset, batch_size=params["bs"], shuffle=True)
    unlabeled_dl = TfmdDL(unlabeled_dataset, batch_size=params["bs"], shuffle=True)
    valid_dl = TfmdDL(valid_dataset, batch_size=params["bs"], shuffle=True)

    # Setup FixMatch callback
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

    # Create model with best parameters
    base_model = InceptionTimePlus(
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
    # Load pretrained weights
    model_filename = "mvp_pretrain.pth"
    full_save_path = os.path.join(MODEL_SAVE_PATH, model_filename)
    checkpoint = torch.load(full_save_path, map_location=DEVICE)
    base_model.backbone.load_state_dict(checkpoint, strict=False)  
    torch.cuda.empty_cache()
    model = InceptionWithSigmoid(base_model=base_model)

    # Create learner
    learn = ts_learner(
        dls,
        model,
        loss_func=BCELossFlat(),
        metrics=[precision_multi, f1_multi],
        cbs=[fixmatch_cb, EarlyStoppingCallback(patience=40)]
    )

    # Train with best learning rate
    learn.fit_one_cycle(
        n_epoch=100,
        lr_max=params['lr']
    )

    model_filename = f"combined_model_{RATIO_TO_SIZE[LABELED_RATIO]}.pth"
    full_model_path = os.path.join(os.path.dirname(MODEL_SAVE_PATH), model_filename)

    torch.save({
        'model_state_dict': learn.model.state_dict(),
        'params': params
    }, full_model_path)

    print(f"Model saved to {full_model_path}")


if __name__ == "__main__":
    train_model()