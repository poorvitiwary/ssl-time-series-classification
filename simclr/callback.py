from tsai.all import *

class SimCLRCallback(Callback):
    """
    A callback to implement SimCLR training with NT-Xent loss for contrastive learning.

    Inspired by the NoisyStudent callback structure, managing the full training loop
    including loss computation.
    """

    def __init__(self, temperature=0.5):
        """Initialize the callback with temperature parameter."""
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def before_fit(self):
        """
        Set up any necessary parameters before fitting the model.

        Stores the original loss function and replaces it with the SimCLR loss.
        """
        # Store the original loss function
        self.old_loss_func = self.learn.loss_func
        # Replace the loss function with the custom SimCLR loss
        self.learn.loss_func = self.loss

    def loss(self, output, *yb):
        """
        Compute the NT-Xent loss for SimCLR contrastive learning.

        Args:
            output: Model outputs containing both augmented views
            *yb: Ignored targets (unused in contrastive learning)

        Returns:
            Computed contrastive loss
        """
        z1, z2 = output.chunk(2, dim=0)  # Split batch into two views

        # Handle mismatched batch sizes by truncating to smallest size
        min_size = min(z1.size(0), z2.size(0))
        z1, z2 = z1[:min_size], z2[:min_size]

        # Skip if we have no valid pairs (shouldn't happen with proper batch sizes)
        if min_size == 0:
            return torch.tensor(0.0, requires_grad=True, device=z1.device)

        # Normalize the embeddings
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        batch_size = z1.size(0)
        # Concatenate both views
        z = torch.cat([z1, z2], dim=0)  # Shape: (2 * batch_size, projection_dim)

        # Compute similarity matrix
        sim_matrix = torch.mm(z, z.T) / self.temperature

        # Mask to exclude self-similarities
        diag_mask = torch.eye(
            2 * batch_size,
            dtype=torch.bool,
            device=sim_matrix.device
        )
        sim_matrix.masked_fill_(diag_mask, -float('inf'))

        # Positive pairs (matching images from different augmentations)
        sim_i_j = sim_matrix[range(batch_size), range(batch_size, 2 * batch_size)]
        sim_j_i = sim_matrix[range(batch_size, 2 * batch_size), range(batch_size)]
        positive_samples = torch.cat([sim_i_j, sim_j_i], dim=0).reshape(2 * batch_size, 1)

        # Negative pairs (all other non-matching images)
        negative_samples = sim_matrix.masked_select(~diag_mask).reshape(
            2 * batch_size, -1
        )

        # Combine positive and negative samples
        logits = torch.cat([positive_samples, negative_samples], dim=1)

        # Labels indicate the positive sample is at index 0
        labels = torch.zeros(
            2 * batch_size,
            dtype=torch.long,
            device=sim_matrix.device
        )

        # Compute and normalize loss
        loss = self.criterion(logits, labels) / (2 * batch_size)
        return loss
    def after_fit(self):
        """Restore the original loss function after training is completed."""
        self.learn.loss_func = self.old_loss_func