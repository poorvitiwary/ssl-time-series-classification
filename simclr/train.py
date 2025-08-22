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
        bs=params['bs']

    )
    model_filename = "simclr_pretrain.pth"
    full_save_path = os.path.join(MODEL_SAVE_PATH, model_filename)
    # Load the best pre-trained SimCLR model
    state_dict = torch.load(full_save_path, map_location=DEVICE) 
 # Initialize the encoder with the same architecture used in pretraining
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

    encoder_dict = encoder.state_dict()

    # Strip 'encoder.' from keys
    pretrained_dict = {
        k.replace('encoder.', ''): v
        for k, v in state_dict.items()
        if k.replace('encoder.', '') in encoder_dict
        and v.size() == encoder_dict[k.replace('encoder.', '')].size()
    }

    encoder_dict.update(pretrained_dict)
    encoder.load_state_dict(encoder_dict)

    # Define the supervised model with the pretrained encoder
    model = MultiLabelClassifierWithSigmoid(base_model=encoder)

    torch.cuda.empty_cache()

    # Fine-tuning setup
    n_epochs = 20
    freeze_epochs = 10  # Number of epochs to freeze the encoder
    n_tests = 10  # Number of test runs
    results = []
    best_f1 = 0.0
    for i in range(n_tests):
        clear_output()
        if i > 0:
            print(f'{i}/{n_tests} F1: {np.mean(results):.3f} +/- {np.std(results):.3f}')
        else:
            print(f'{i}/{n_tests}')

        # Initialize the learner
        learn_finetune = Learner(
            dls,
            model,
            loss_func=BCELossFlat(),
            metrics=[precision_multi, f1_multi]
        )

        # Freeze the encoder for the first `freeze_epochs` epochs
        learn_finetune.freeze()  # Freeze the encoder layers
        learn_finetune.fit_one_cycle(freeze_epochs, lr_max=params['lr'])

        # Unfreeze the encoder for the remaining epochs
        learn_finetune.unfreeze()  # Unfreeze the encoder layers
        learn_finetune.fit_one_cycle(n_epochs, lr_max=params['lr'])

        current_f1 = learn_finetune.recorder.values[-1][-1]
        results.append(current_f1)
                # Save model if it's the best so far
        if current_f1 > best_f1:
            best_f1 = current_f1
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"New best model saved with F1: {best_f1:.3f}")

    learn_finetune.plot_metrics()
    print(f'F1: {np.mean(results):.3f} +/- {np.std(results):.3f} in {n_tests} tests')

if __name__ == "__main__":
    train_model()

