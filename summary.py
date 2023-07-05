import torch
import pandas as pd
import torch.nn as nn

from collections import OrderedDict


def summary_string(
    model: torch.nn.Module, input_size: tuple, device: torch.device = None, dtypes=None
) -> pd.DataFrame:
    """This function, summary_string(), is designed to provide a comprehensive summary of a given PyTorch neural network model.
    This includes detailed information on each layer's type, input and output shape, and the number of trainable parameters.
    It also provides the total number of parameters, the number of trainable parameters, the size of the input, output,
    and parameters, and an estimate of the total size of the model.

    The function works by registering 'hooks' to each module (layer) of the model.
    In PyTorch, hooks are functions that can be manually registered to be executed when a forward or backward pass is run.
    These hooks allow the developer to interact with and extract information from the module, even deep within a nested module structure.
    In this case, the hooks extract the input and output shape, as well as the number of parameters for each module in the model.
    """

    if not device:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # If dtypes not provided, default to FloatTensor for all input_size elements
    if dtypes == None:
        dtypes = [torch.FloatTensor] * len(input_size)

    # Initialize a string to hold the model summary
    summary_str = ""

    # Define a function to register a forward hook on the PyTorch modules
    def register_hook(module):
        """The register_hook() function is applied to all sub-modules in the model using the model.apply() method,
        which ensures the hooks are applied recursively to all sub-modules."""

        # Define the hook function that will be called every forward pass
        def hook(module, input, output):
            """A forward hook function, hook(module, input, output), is defined within the register_hook() function.
            This hook function extracts the information needed and stores it in the summary dictionary.
            The hook function gets its input and output parameters automatically from PyTorch
            whenever a forward pass is performed on the module that the hook is registered to.
            This is a feature built into PyTorch and is how hooks work in general.

            When you call module.register_forward_hook(hook), you're telling PyTorch:
            "Whenever a forward pass is done on this module, please call the hook function,
            and automatically provide it with the input to and output from the module".
            """

            # Extract the class name of the module
            # This works by extracting the "human readable" name of the module from the module's class type

            """
            Example
            
            # Consider a Conv2d layer as an example
            module = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3)

            # Get the string representation of the module's class
            class_name = str(module.__class__)
            print(class_name)  # Prints: <class 'torch.nn.modules.conv.Conv2d'>

            # Split by period
            parts = class_name.split(".")
            print(parts)  # Prints: ["<class 'torch", 'nn', 'modules', 'conv', "Conv2d'>"]

            # Select the last part
            last_part = parts[-1]
            print(last_part)  # Prints: "Conv2d'>"

            # Split by single quote and select the first part
            final_class_name = last_part.split("'")[0]
            print(final_class_name)  # Prints: Conv2d

            """
            class_name = str(module.__class__).split(".")[-1].split("'")[0]

            # Get the module index based on the current length of summary
            module_idx = len(summary)

            # Create a unique key for the current module
            m_key = f"{class_name}-{module_idx+1}"

            # Initialize an OrderedDict for this module
            summary[m_key] = OrderedDict()

            # Init
            summary[m_key]["params"] = 0
            summary[m_key]["trainable"] = False

            if hasattr(module, "weight"):
                summary[m_key]["params"] += module.weight.numel()
                summary[m_key]["trainable"] = module.weight.requires_grad
            if hasattr(module, "bias") and hasattr(module.bias, "numel"):
                summary[m_key]["params"] += module.bias.numel()

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
    summary = OrderedDict()
    hooks = []

    # Apply the register_hook function to all sub-modules
    model.apply(register_hook)

    # Make a forward pass to trigger the hooks
    model(*x)

    # Remove the hooks after use
    for h in hooks:
        h.remove()

    # Heading
    summary_str += (
        "----------------------------------------------------------------" + "\n"
    )
    summary_str += f"{'Layer':<20}  {'Parameters':<25} {'Trainable':<15}\n"
    summary_str += (
        "================================================================" + "\n"
    )

    # Init counter
    total_params = 0
    total_output = 0
    trainable_params = 0

    # Loop over layers in summary
    for layer in summary:
        # Add new line
        summary_str += f"{layer:<20}  {str(summary[layer]['params']):<25}  {str(summary[layer]['trainable']):<15}\n"

        # Add params to total
        total_params += summary[layer]["params"]

        # If layer is trainable
        if summary[layer]["trainable"]:
            trainable_params += summary[layer]["params"]

    # Append final summary details
    summary_str += (
        "================================================================" + "\n"
    )
    summary_str += f"Total params: {total_params}\n"
    summary_str += f"Trainable params: {trainable_params}\n"
    summary_str += f"Non-trainable params: {total_params - trainable_params}\n"
    summary_str += "----------------------------------------------------------------"

    print(summary_str)

    # return summary
    return pd.DataFrame().from_dict(summary, orient="index").reset_index(names="layer")
