from torch import nn

"""Helper models training and optimiztaion."""

class InceptionWithSigmoid(nn.Module):
    """Wrapper model that adds sigmoid activation to base model outputs.

    Args:
        base_model: The base neural network model
    """

    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """Forward pass with sigmoid activation.

        Args:
            x: Input tensor

        Returns:
            Output tensor with sigmoid activation
        """
        return self.sigmoid(self.base_model(x))
    

class MultiLabelClassifierModel(nn.Module):
    """A multi-label classifier model using a given encoder."""

    def __init__(self, encoder, num_classes):
        """Initialize the model with encoder and classifier head.
        
        Args:
            encoder: The encoder model
            num_classes: Number of output classes
        """
        super().__init__()
        self.encoder = encoder
        self.classifier_head = nn.Linear(encoder.c_out, num_classes)

    def forward(self, x):
        """Forward pass of the model.
        
        Args:
            x: Input tensor
            
        Returns:
            Classification logits
        """
        x = self.encoder(x)
        return self.classifier_head(x)


class SimCLRModel(nn.Module):
    """SimCLR model with encoder and projection head."""

    def __init__(self, encoder, projection_dim=128):
        """Initialize the SimCLR model.
        
        Args:
            encoder: The encoder model
            projection_dim: Dimension of the projection head output
        """
        super().__init__()
        self.encoder = encoder

        # Use `c_out` from the encoder directly
        self.projection_head = nn.Sequential(
            nn.Linear(encoder.c_out, 256),  # `c_out` defines the encoder's output size
            nn.ReLU(),
            nn.Linear(256, projection_dim)
        )

    def forward(self, x):
        """Forward pass of the model.
        
        Args:
            x: Input tensor
            
        Returns:
            Projected features
        """
        x = self.encoder(x)  # Pass through the encoder
        return self.projection_head(x)  # Pass through the projection head


class MultiLabelClassifierWithSigmoid(nn.Module):
    """Wrapper for multi-label classifier that adds sigmoid activation."""

    def __init__(self, base_model):
        """Initialize the wrapper model.
        
        Args:
            base_model: The base classification model
        """
        super().__init__()
        self.base_model = base_model
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """Forward pass with sigmoid activation.
        
        Args:
            x: Input tensor
            
        Returns:
            Probability scores for each class
        """
        return self.sigmoid(self.base_model(x))