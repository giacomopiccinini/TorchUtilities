import torch


def count_parameters(model: torch.nn.Module):
    """Count number of parameters in a model.
    Discriminate between trainable and frozen parameters."""

    # Init counter
    trainable = 0
    frozen = 0

    # Loop over parameters
    for p in model.parameters():
        # If it requires gradient, it is trainable
        if p.requires_grad:
            trainable += p.numel()
        # Else it is frozen
        else:
            frozen += p.numel()

    # Compute percentage of trainable parameters
    percentage = trainable / (trainable + frozen) * 100

    # Create dictionary
    parameters = {"trainable": trainable, "frozen": frozen, "percentage": percentage}

    return parameters
