from config import *
from metrics import *
from best_params import *
from data import *
from models import *
from utils import *
from tsai.all import *
import os
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
    best_params = BEST_PARAMS[LABELED_RATIO]
    params = best_params.copy()

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
        lr_max=params["lr"],
        cbs=[EarlyStoppingCallback(patience=40)]
    )
    
    model_filename = f"supervised_model_{RATIO_TO_SIZE[LABELED_RATIO]}.pth"
    full_model_path = os.path.join(os.path.dirname(MODEL_SAVE_PATH), model_filename)

    torch.save({
        'model_state_dict': learn.model.state_dict(),
        'params': params
    }, full_model_path)

    print(f"Model saved to {full_model_path}")


if __name__ == "__main__":
    train_model()