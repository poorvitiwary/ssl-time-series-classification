from fastai.callback.core import Callback
from torch.utils.data import DataLoader
from tsai.all import *

class FixMatchMultiLabel(Callback):
    """FixMatch callback for multi-label classification tasks with sigmoid + BCELossFlat."""

    def __init__(
        self,
        dl2: DataLoader,
        weak_tfms: list,
        strong_tfms: list,
        confidence_threshold: float = 0.95,
        lambda_u: float = 1.0,
        bs_labeled: int = 64,
        bs_unlabeled: int = 64,
        verbose: bool = False
    ):
        """Initialize FixMatch callback.

        Args:
            dl2: DataLoader for unlabeled data
            weak_tfms: Weak augmentations
            strong_tfms: Strong augmentations
            confidence_threshold: Threshold for pseudo-labeling
            lambda_u: Weight for unsupervised loss
            bs_labeled: Batch size for labeled data
            bs_unlabeled: Batch size for unlabeled data
            verbose: Whether to print debug info
        """
        self.dl2 = dl2
        self.weak_tfms = weak_tfms
        self.strong_tfms = strong_tfms
        self.confidence_threshold = confidence_threshold
        self.lambda_u = lambda_u
        self.bs_labeled = bs_labeled
        self.bs_unlabeled = bs_unlabeled
        self.verbose = verbose

    def before_fit(self):
        """Store original settings before training starts."""
        # Store original batch size and transforms
        self.old_bs = self.dls.train.bs
        self.old_tfms = self.dls.train.after_batch

        # Set batch sizes for labeled and unlabeled DataLoaders
        self.dls.train.bs = self.bs_labeled
        self.dl2.bs = self.bs_unlabeled

        # Initialize unlabeled DataLoader iterator
        self.dl2iter = iter(self.dl2)

        # Store original loss function
        self.old_loss_func = self.learn.loss_func
        self.learn.loss_func = self.loss

    def before_batch(self):
        """Process batch before forward pass."""
        if not self.training:
            return

        # Labeled data (weak augmentation)
        x_labeled, y_labeled = self.x, self.y
        if self.weak_tfms:
            x_labeled = compose_tfms(x_labeled, self.weak_tfms, split_idx=0)

        # Unlabeled data (weak and strong augmentations)
        try:
            x_unlabeled = next(self.dl2iter)
            if isinstance(x_unlabeled, (tuple, list)):
                x_unlabeled = x_unlabeled[0]
        except StopIteration:
            self.dl2iter = iter(self.dl2)
            x_unlabeled = next(self.dl2iter)
            if isinstance(x_unlabeled, (tuple, list)):
                x_unlabeled = x_unlabeled[0]

        # Apply augmentations
        x_unlabeled_weak = compose_tfms(x_unlabeled, self.weak_tfms, split_idx=0)
        x_unlabeled_strong = compose_tfms(x_unlabeled, self.strong_tfms, split_idx=0)

        # Combine labeled and unlabeled data
        self.learn.xb = (torch.cat([x_labeled, x_unlabeled_strong]),)
        self.learn.yb = (y_labeled, x_unlabeled_weak)

    def loss(self, output, *args):
        """Compute combined supervised and unsupervised loss."""
        # Unpack the target arguments
        target = args[0] if len(args) == 1 else args

        if not self.training:
            # Validation mode: Compute only the supervised loss
            y_labeled = target
            labeled_output = output

            if labeled_output.shape != y_labeled.shape:
                raise ValueError(
                    f"Shape mismatch: labeled_output {labeled_output.shape}, "
                    f"y_labeled {y_labeled.shape}"
                )
            return self.old_loss_func(labeled_output, y_labeled)

        # Training mode: Compute both supervised and unsupervised losses
        y_labeled = target[0]
        labeled_output = output[:len(y_labeled)]
        unlabeled_output = output[len(y_labeled):]

        if labeled_output.shape != y_labeled.shape:
            raise ValueError(
                f"Shape mismatch: labeled_output {labeled_output.shape}, "
                f"y_labeled {y_labeled.shape}"
            )

        # Supervised loss (labeled data)
        supervised_loss = self.old_loss_func(labeled_output, y_labeled)

        # Unsupervised loss (unlabeled data)
        y_unlabeled_weak = target[1]
        with torch.no_grad():
            pseudo_probs = unlabeled_output
            pseudo_labels = (pseudo_probs > self.confidence_threshold).float()
            mask = (pseudo_probs > self.confidence_threshold).float()

        if mask.sum() == 0:
            unsupervised_loss = torch.tensor(0.0, device=output.device)
        else:
            unsupervised_loss = (
                self.old_loss_func(unlabeled_output, pseudo_labels) * mask
            ).mean()

        # Combine losses
        return supervised_loss + self.lambda_u * unsupervised_loss

    def after_fit(self):
        """Restore original settings after training."""
        self.dls.train.bs = self.old_bs
        self.dls.train.after_batch = self.old_tfms
        self.learn.loss_func = self.old_loss_func