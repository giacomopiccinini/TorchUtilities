import torch
import torch.nn as nn

from typing import Dict, List


def layers_list(
    model: torch.nn.Module, input_size: tuple, device: torch.device = None, dtypes=None
) -> List[Dict]:
    """This function returns the layers of a PyTorch model as a list of dictionaries. In particular they are ordered such that the layers closest to the output are last.
    The layers are organized in a list, where each element of the list a dictionary with the following keys:
    layer: the PyTorch module
    parameters: the number of parameters in the layer
    """

    if not device:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # If dtypes not provided, default to FloatTensor for all input_size elements
    if dtypes == None:
        dtypes = [torch.FloatTensor] * len(input_size)

    # Define a function to register a forward hook on the PyTorch modules
    def register_hook(module):
        """The register_hook() function is applied to all sub-modules in the model using the model.apply() method,
        which ensures the hooks are applied recursively to all sub-modules."""

        # Define the hook function that will be called every forward pass
        def hook(module, input, output):
            # Create layer dict
            layer_dict = {"layer": module}

            # Fill the dictionary
            parameters = 0

            if hasattr(module, "weight"):
                parameters += module.weight.numel()
            if hasattr(module, "bias") and hasattr(module.bias, "numel"):
                parameters += module.bias.numel()

            layer_dict["parameters"] = parameters

            # Append to list
            layers.append(layer_dict)

        # Register the hook function to the module if it's not Sequential or ModuleList
        if not isinstance(module, nn.Sequential) and not isinstance(
            module, nn.ModuleList
        ):
            hooks.append(module.register_forward_hook(hook))

    # Adjust for multiple input_size
    if isinstance(input_size, tuple):
        input_size = [input_size]

    # Prepare dummy inputs for forward pass
    x = [
        torch.rand(2, *in_size).type(dtype)
        for in_size, dtype in zip(input_size, dtypes)
    ].to(device)

    # Initialize summary and hooks list
    layers = []
    hooks = []

    # Apply the register_hook function to all sub-modules
    model.apply(register_hook)

    # Make a forward pass to trigger the hooks
    model(*x)

    # Remove the hooks after use
    for h in hooks:
        h.remove()

    # return summary
    return layers[::-1]
