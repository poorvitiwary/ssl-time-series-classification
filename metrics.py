"""Metrics for model evaluation."""

def precision_multi(inp, targ, thresh=0.5, sigmoid=False):
    """Computes precision when `inp` and `targ` are the same size."""
    inp, targ = flatten_check(inp, targ)
    if sigmoid:
        inp = inp.sigmoid()
    pred = inp > thresh

    correct = pred == targ.bool()
    TP = torch.logical_and(correct, (targ == 1).bool()).sum()
    FP = torch.logical_and(~correct, (targ == 0).bool()).sum()

    precision = TP / (TP + FP + 1e-8)  # Added epsilon for stability
    return precision


def recall_multi(inp, targ, thresh=0.5, sigmoid=False):
    """Computes recall when `inp` and `targ` are the same size."""
    inp, targ = flatten_check(inp, targ)
    if sigmoid:
        inp = inp.sigmoid()
    pred = inp > thresh

    correct = pred == targ.bool()
    TP = torch.logical_and(correct, (targ == 1).bool()).sum()
    FN = torch.logical_and(~correct, (targ == 1).bool()).sum()

    recall = TP / (TP + FN + 1e-8)
    return recall


def specificity_multi(inp, targ, thresh=0.5, sigmoid=False):
    """Computes specificity (true negative rate)."""
    inp, targ = flatten_check(inp, targ)
    if sigmoid:
        inp = inp.sigmoid()
    pred = inp > thresh

    correct = pred == targ.bool()
    TN = torch.logical_and(correct, (targ == 0).bool()).sum()
    FP = torch.logical_and(~correct, (targ == 0).bool()).sum()

    specificity = TN / (TN + FP + 1e-8)
    return specificity


def balanced_accuracy_multi(inp, targ, thresh=0.5, sigmoid=False):
    """Computes balanced accuracy."""
    inp, targ = flatten_check(inp, targ)
    if sigmoid:
        inp = inp.sigmoid()
    pred = inp > thresh

    correct = pred == targ.bool()
    TP = torch.logical_and(correct, (targ == 1).bool()).sum()
    TN = torch.logical_and(correct, (targ == 0).bool()).sum()
    FN = torch.logical_and(~correct, (targ == 1).bool()).sum()
    FP = torch.logical_and(~correct, (targ == 0).bool()).sum()

    TPR = TP / (TP + FN + 1e-8)
    TNR = TN / (TN + FP + 1e-8)
    balanced_accuracy = (TPR + TNR) / 2
    return balanced_accuracy


def fbeta_multi(inp, targ, beta=1.0, thresh=0.5, sigmoid=False):
    """Computes F-beta score (default: F1)."""
    if sigmoid:
        inp = inp.sigmoid()
    pred = (inp > thresh).float()

    def single_class_fbeta(pred, targ):
        TP = (pred * targ).sum()
        FP = (pred * (1 - targ)).sum()
        FN = ((1 - pred) * targ).sum()

        precision = TP / (TP + FP + 1e-8)
        recall = TP / (TP + FN + 1e-8)
        beta_2 = beta ** 2

        return (1 + beta_2) * precision * recall / (beta_2 * precision + recall + 1e-8)

    fbetas = [single_class_fbeta(pred[:, i], targ[:, i]) for i in range(pred.size(1))]
    return sum(fbetas) / len(fbetas)


def f1_multi(*args, **kwargs):
    """Computes F1 score (alias for F-beta with beta=1)."""
    return fbeta_multi(*args, **kwargs)