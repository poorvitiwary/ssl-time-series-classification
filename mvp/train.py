from config import *
from metrics import *
from best_params import *
from data import *
from models import *
import torch
import numpy as np
from tsai.all import *
from IPython.display import clear_output

"""Training module to train the downstream task with best parameters post optimization."""

def train_model():
    """Train the model using optimized hyperparameters and save the best model."""
    params = BEST_PARAMS.copy()

    # Model configuration
    c_in = 1  # Number of features/channels in the input data
    c_out = 8  # Number of classes
    seq_len = 920

    # Get data (now with combined labeled set)
    (_, _,
     x_labeled_combined, y_labeled_combined,
     _, _,
     _, downstream_splits) = prepare_data()
    
    # Create dataloaders
    dls = get_ts_dls(
        x_labeled_combined,
        y_labeled_combined,
        splits=downstream_splits,
        bs=params['bs'],
    )

    # Initialize model
    model = InceptionTimePlus(
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

    model = InceptionWithSigmoid(base_model=model)

    # Training configuration
    n_epochs = 20
    freeze_epochs = 10
    n_tests = 10
    results = []
    best_f1 = 0.0
    model_filename = "mvp_pretrain.pth"
    full_save_path = os.path.join(MODEL_SAVE_PATH, model_filename)
    # Training loop
    for i in range(n_tests):
        clear_output()
        if i > 0:
            print(f'{i}/{n_tests} F1: {np.mean(results):.3f} '
                  f'+/- {np.std(results):.3f}')
        else:
            print(f'{i}/{n_tests}')
        
        # Initialize learner
        learn = ts_learner(
            dls,
            model,
            loss_func=BCELossFlat(),
            pretrained=True,
            weights_path=full_save_path,
            metrics=[precision_multi, f1_multi]
        )
        
        # Fine-tune model
        learn.fine_tune(
            n_epochs,
            base_lr=params['lr'],
            freeze_epochs=freeze_epochs
        )
        
        # Get current F1 score
        current_f1 = learn.recorder.values[-1][-1]
        results.append(current_f1)
        
        # Save model if it's the best so far
        if current_f1 > best_f1:
            best_f1 = current_f1
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"New best model saved with F1: {best_f1:.3f}")

    # Final results
    learn.plot_metrics()
    print(f'\nFinal F1: {np.mean(results):.3f} +/- {np.std(results):.3f} '
          f'in {n_tests} tests')
    print(f'Best model saved to {MODEL_SAVE_PATH} with F1: {best_f1:.3f}')


if __name__ == "__main__":
    train_model()