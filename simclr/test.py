from config import *
from metrics import *
from best_params import *
from data import *
from models import *
from tsai.all import *
import torch
import numpy as np
import os

"""Testing module to test the saved trained models post optimization."""

def test_model(num_runs=5):
    """Test models matching 'simclr_model_{size}.pth' pattern.
    
    Args:
        num_runs (int): Number of test iterations (default: 5)
    """
    ratio = LABELED_RATIO
    size = RATIO_TO_SIZE.get(ratio)
    if size is None:
        print(f"Error: No size mapping for ratio {ratio}")
        return

    model_name = f"simclr_model_{size}.pth"
    model_path = os.path.join(MODEL_SAVE_PATH, model_name)
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found: {model_name}")
        print("Available simclr models:")
        for f in os.listdir(MODEL_SAVE_PATH):
            if f.startswith("simclr_model_") and f.endswith(".pth"):
                print(f"- {f}")
        return

    # Load best params for this ratio
    params = BEST_PARAMS
    
    # Get test data
    (_, _,
     _, _,
     x_test, y_test,
     _, _) = prepare_data()
    
    # Create test dataloader
    batch_tfms = TSStandardize()
    test_dls = get_ts_dls(
        x_test,
        y_test,
        bs=params['bs'],
        batch_tfms=batch_tfms,
        shuffle=False  # Typically don't shuffle for testing
    )

    # Initialize model
    base_model = InceptionTimePlus(
        c_in=1,
        c_out=8,
        seq_len=920,
        **{k: v for k, v in params.items()
           if k in ['nf', 'fc_dropout', 'conv_dropout', 'ks',
                   'bottleneck', 'coord', 'separable', 'dilation']}
    )
    model = MultiLabelClassifierWithSigmoid(base_model=base_model)
    model.load_state_dict(torch.load(model_path))
    model.to(DEVICE)
    model.eval()

    # Create learner for evaluation
    learn = Learner(
        test_dls,
        model,
        loss_func=BCELossFlat(),
        metrics=[precision_multi, f1_multi]
    )

    # Evaluation loop
    f1_scores = []
    print(f"\nTesting {model_name} (Ratio: {ratio}, Size: {size})")
    print(f"Batch size: {params['bs']}, Filters: {params['nf']}")
    
    for i in range(num_runs):
        preds, targets = learn.get_preds(dl=test_dls.train)
        f1 = f1_multi(preds, targets)
        f1_scores.append(f1.item())
        print(f"Run {i+1}/{num_runs}: F1 = {f1:.4f}")

    # Results
    avg_f1 = np.mean(f1_scores)
    std_f1 = np.std(f1_scores)
    
    print("\nFinal Results:")
    print(f"Avg F1: {avg_f1:.4f}")
    print(f"Std Dev: ±{std_f1:.4f}")
    print(f"Range: [{avg_f1-std_f1:.4f}, {avg_f1+std_f1:.4f}]")


if __name__ == "__main__":
    test_model(num_runs=5)