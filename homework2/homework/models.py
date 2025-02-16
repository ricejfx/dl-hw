"""
Implement the following models for classification.

Feel free to modify the arguments for each of model's __init__ function.
This will be useful for tuning model hyperparameters such as hidden_dim, num_layers, etc,
but remember that the grader will assume the default constructor!
"""

from pathlib import Path

import torch
import torch.nn as nn


class ClassificationLoss(nn.Module):
    def forward(self, logits: torch.Tensor, target: torch.LongTensor) -> torch.Tensor:
        """
        Multi-class classification loss
        Hint: simple one-liner

        Args:
            logits: tensor (b, c) logits, where c is the number of classes
            target: tensor (b,) labels

        Returns:
            tensor, scalar loss
        """

        #raise NotImplementedError("ClassificationLoss.forward() is not implemented")
        return torch.nn.functional.cross_entropy(logits, target)


class LinearClassifier(nn.Module):
    def __init__(
        self,
        h: int = 64,
        w: int = 64,
        num_classes: int = 6,
    ):
        """
        Args:
            h: int, height of the input image
            w: int, width of the input image
            num_classes: int, number of classes
        """
        super().__init__()
        layers = []
        layers.append(torch.nn.Flatten()) 
        layers.append(torch.nn.Linear(h*w*3, num_classes)) # assumuing 3 channels based on the forward and Readme

        self.model = torch.nn.Sequential(*layers)
        
        return None
        #raise NotImplementedError("LinearClassifier.__init__() is not implemented")


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor (b, 3, H, W) image

        Returns:
            tensor (b, num_classes) logits
        """
        return self.model(x)
        #raise NotImplementedError("LinearClassifier.forward() is not implemented")


class MLPClassifier(nn.Module):
    def __init__(
        self,
        h: int = 64,
        w: int = 64,
        num_classes: int = 6,
        hidden_dim: int = 96 
    ):
        """
        An MLP with a single hidden layer

        Args:
            h: int, height of the input image
            w: int, width of the input image
            num_classes: int, number of classes
        """
        super().__init__()
        layers = []
        layers.append(torch.nn.Flatten())
    
        layers.append(torch.nn.Linear(h*w*3, hidden_dim))
        layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Linear(hidden_dim, num_classes))

        #raise NotImplementedError("MLPClassifier.__init__() is not implemented")
        self.model = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor (b, 3, H, W) image

        Returns:
            tensor (b, num_classes) logits
        """
        return self.model(x)
        #raise NotImplementedError("MLPClassifier.forward() is not implemented")


class MLPClassifierDeep(nn.Module):
    def __init__(
        self,
        h: int = 64,
        w: int = 64,
        num_classes: int = 6,
        first_layer_dim: int = 128,
        hidden_dim: list = [128, 64, 32]#,
        # optimizer: torch.optim.Optimizer = None,
        # regularizer: float = None
    ):
        """
        An MLP with multiple hidden layers

        Args:
            h: int, height of image
            w: int, width of image
            num_classes: int

        Hint - you can add more arguments to the constructor such as:
            hidden_dim: int, size of hidden layers
            num_layers: int, number of hidden layers
        """
        super().__init__()
        layers = []
        layers.append(torch.nn.Flatten())
        layers.append(torch.nn.Linear(h*w*3, first_layer_dim))
        for i in range(len(hidden_dim)):
            layers.append(torch.nn.ReLU())
            #layers.append(torch.nn.Linear(hidden_dim, num_classes if (i+1)==num_layers else hidden_dim))
            layers.append(torch.nn.Linear(first_layer_dim if i==0 else hidden_dim[i],
                                          num_classes if (i+1)==len(hidden_dim) else hidden_dim[i+1]))

        self.model = torch.nn.Sequential(*layers)
        #raise NotImplementedError("MLPClassifierDeep.__init__() is not implemented")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor (b, 3, H, W) image

        Returns:
            tensor (b, num_classes) logits
        """
        return self.model(x)
        #raise NotImplementedError("MLPClassifierDeep.forward() is not implemented")


class MLPClassifierDeepResidual(nn.Module):
    if 1:
        class MLPCDRBlock(nn.Module):
            def __init__(self, in_channels: int, out_channels: int):
                super(MLPClassifierDeepResidual.MLPCDRBlock, self).__init__()
                
                self.model = torch.nn.Sequential(
                    torch.nn.Linear(in_channels, out_channels),
                    torch.nn.LayerNorm(out_channels),
                    torch.nn.ReLU(),
                    torch.nn.Linear(out_channels, out_channels),
                    torch.nn.LayerNorm(out_channels),
                    torch.nn.ReLU()
                )

                if in_channels != out_channels:
                    self.skip = torch.nn.Linear(in_channels, out_channels)
                else:
                    self.skip = torch.nn.Identity()

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.model(x) + self.skip(x)
    
    def __init__(
        self,
        h: int = 64,
        w: int = 64,
        num_classes: int = 6,
        first_layer_dim: int = 128,
        hidden_dim: list = [512, 128, 64]#, [64, 64, 64],
        #regularizer: float = None,
        #optimizer: torch.optim.Optimizer = None
    ):
        """
        Args:
            h: int, height of image
            w: int, width of image
            num_classes: int

        Hint - you can add more arguments to the constructor such as:
            hidden_dim: int, size of hidden layers
            num_layers: int, number of hidden layers
        """
        super().__init__()

        layers = []
        layers.append(torch.nn.Flatten())
        init_c = first_layer_dim
        layers.append(torch.nn.Linear(h*w*3, init_c, bias=False))
        #c = hidden_dim[0]
        c = init_c
        
        #for s in hidden_dim[1:]:
        for s in hidden_dim:
            layers.append(MLPClassifierDeepResidual.MLPCDRBlock(c, s))
            c = s
        
        layers.append(torch.nn.Linear(c, num_classes, bias=False))

        self.model = torch.nn.Sequential(*layers)
        #raise NotImplementedError("MLPClassifierDeepResidual.__init__() is not implemented")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor (b, 3, H, W) image

        Returns:
            tensor (b, num_classes) logits
        """
        return self.model(x)
        #raise NotImplementedError("MLPClassifierDeepResidual.forward() is not implemented")


model_factory = {
    "linear": LinearClassifier,
    "mlp": MLPClassifier,
    "mlp_deep": MLPClassifierDeep,
    "mlp_deep_residual": MLPClassifierDeepResidual,
}


def calculate_model_size_mb(model: torch.nn.Module) -> float:
    """
    Args:
        model: torch.nn.Module

    Returns:
        float, size in megabytes
    """
    return sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024


def save_model(model):
    """
    Use this function to save your model in train.py
    """
    for n, m in model_factory.items():
        if isinstance(model, m):
            return torch.save(model.state_dict(), Path(__file__).resolve().parent / f"{n}.th")
    raise ValueError(f"Model type '{str(type(model))}' not supported")


def load_model(model_name: str, with_weights: bool = False, **model_kwargs):
    """
    Called by the grader to load a pre-trained model by name
    """
    r = model_factory[model_name](**model_kwargs)
    if with_weights:
        model_path = Path(__file__).resolve().parent / f"{model_name}.th"
        assert model_path.exists(), f"{model_path.name} not found"
        try:
            r.load_state_dict(torch.load(model_path, map_location="cpu"))
        except RuntimeError as e:
            raise AssertionError(
                f"Failed to load {model_path.name}, make sure the default model arguments are set correctly"
            ) from e

    # Limit model sizes since they will be zipped and submitted
    model_size_mb = calculate_model_size_mb(r)
    if model_size_mb > 10:
        raise AssertionError(f"{model_name} is too large: {model_size_mb:.2f} MB")
    print(f"Model size: {model_size_mb:.2f} MB")

    return r
