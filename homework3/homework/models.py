from pathlib import Path

import torch
import torch.nn as nn

HOMEWORK_DIR = Path(__file__).resolve().parent
INPUT_MEAN = [0.2788, 0.2657, 0.2629]
INPUT_STD = [0.2064, 0.1944, 0.2252]

class Classifier(nn.Module):
    class DownBlock(nn.Module):
        def __init__(self, in_channels : int = 3, out_channels : int = 6, ks : int = 3):
            super().__init__()
            self.model = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=ks, stride=2, padding=1),
                nn.GroupNorm(1, out_channels),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=ks, padding=1),
                nn.GroupNorm(1, out_channels),
                nn.ReLU()
            )
        
        def forward(self, x):
            return self.model(x)

    class UpBlock(nn.Module):
        def __init__(self, in_channels : int, out_channels : int, ks : int = 3):
            super().__init__()
            self.model = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=ks, stride=2, padding=1, output_padding=1),
                nn.GroupNorm(1, out_channels),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=ks, padding=1),
                nn.GroupNorm(1, out_channels),
                nn.ReLU(),
                nn.Dropout(p=0.1)
            )
        
        def forward(self, x):
            return self.model(x)

    def __init__(self, in_channels: int = 3, num_classes: int = 6, ks: int = 3):
        super().__init__()

        # Downs
        self.down1 = self.DownBlock(in_channels, 32)
        self.down2 = self.DownBlock(32, 64)
        self.down3 = self.DownBlock(64, 128)

        # Ups
        self.up3 = self.UpBlock(128, 128)
        self.up2 = self.UpBlock(128 + 64, 64)
        self.up1 = self.UpBlock(64 + 32, 32)

        # Put together the image processing block
        # 32 channels, matching what I have below
        self.img_proc = nn.Sequential(
            nn.Conv2d(32 + in_channels, 32, kernel_size=ks, padding=1),
            nn.GroupNorm(1, 32),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Conv2d(32, num_classes, kernel_size=1)
        )

        # Input normalization
        self.register_buffer("input_mean", torch.tensor(INPUT_MEAN))
        self.register_buffer("input_std", torch.tensor(INPUT_STD))

        return None 

    def forward(self, x):
        z = (x - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]
        # netowrk step            # expected tensor shape
        d1 = self.down1(z)        # (B, 32, H/2, W/2)
        d2 = self.down2(d1)       # (B, 64, H/4, W/4)
        d3 = self.down3(d2)       # (B, 128, H/8, W/8)

        # using cat since this give me more flexibility with the node size

        u3 = self.up3(d3)                      # (B, 128, H/4, W/4)
        u3 = torch.cat([u3, d2], dim=1)

        u2 = self.up2(u3)                     # (B, 64, H/2, W/2)
        u2 = torch.cat([u2, d1], dim=1)

        u1 = self.up1(u2)                     # (B, 32, H, W)
        u1 = torch.cat([u1, z], dim=1)

        return self.img_proc(u1).mean(dim=-1).mean(dim=-1)
    
    def predict(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Used for inference, takes an image and returns class labels and normalized depth.
        This is what the metrics use as input (this is what the grader will use!).

        Args:
            x (torch.FloatTensor): image with shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            tuple of (torch.LongTensor, torch.FloatTensor):
                - pred: class labels {0, 1, 2} with shape (b, h, w)
                - depth: normalized depth [0, 1] with shape (b, h, w)
        """
        logits = self(x)
        pred = logits.argmax(dim=1)

        return pred

class Detector(nn.Module):
    class DownBlock(nn.Module):
        def __init__(self, in_channels : int, out_channels : int, ks : int = 3):
            super().__init__()
            self.model = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=ks, stride=2, padding=1),
                nn.GroupNorm(1, out_channels),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=ks, padding=1),
                nn.GroupNorm(1, out_channels),
                nn.ReLU()
            )
        
        def forward(self, x):
            return self.model(x)

    class UpBlock(nn.Module):
        def __init__(self, in_channels : int, out_channels : int, ks : int = 3):
            super().__init__()
            self.model = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=ks, stride=2, padding=1, output_padding=1),
                nn.GroupNorm(1, out_channels),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=ks, padding=1),
                nn.GroupNorm(1, out_channels),
                nn.ReLU(),
                nn.Dropout(p=0.1)
            )
        
        def forward(self, x):
            return self.model(x)

    def __init__(self, in_channels: int = 3, num_classes: int = 3, ks: int = 3):
        super().__init__()

        # Downs
        self.down1 = self.DownBlock(in_channels, 32)
        self.down2 = self.DownBlock(32, 64)
        self.down3 = self.DownBlock(64, 128)

        # Ups
        self.up3 = self.UpBlock(128, 128)
        self.up2 = self.UpBlock(128 + 64, 64)
        self.up1 = self.UpBlock(64 + 32, 32)

        # Put together the image processing block
        # 32 channels, matching what I have below
        self.img_proc = nn.Sequential(
            nn.Conv2d(32 + in_channels, 32, kernel_size=3, padding=1),
            nn.GroupNorm(1, 32),
            nn.ReLU(),
            nn.Dropout(p=0.1)
        )

        self.segmentation = nn.Conv2d(32, num_classes, kernel_size=1)
        self.depth = nn.Conv2d(32, 1, kernel_size=1)

        # Input normalization
        self.register_buffer("input_mean", torch.tensor(INPUT_MEAN))
        self.register_buffer("input_std", torch.tensor(INPUT_STD))

    def forward(self, x):
        z = (x - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]
        # netowrk step            # expected tensor shape
        d1 = self.down1(z)        # (B, 32, H/2, W/2)
        d2 = self.down2(d1)       # (B, 64, H/4, W/4)
        d3 = self.down3(d2)       # (B, 128, H/8, W/8)

        # using cat since this give me more flexibility with the node size

        u3 = self.up3(d3)                      # (B, 128, H/4, W/4)
        u3 = torch.cat([u3, d2], dim=1)

        u2 = self.up2(u3)                     # (B, 64, H/2, W/2)
        u2 = torch.cat([u2, d1], dim=1)

        u1 = self.up1(u2)                     # (B, 32, H, W)
        u1 = torch.cat([u1, z], dim=1)

        img = self.img_proc(u1)               # (B, 32, H, W)

        logits = self.segmentation(img)         # (B, num_classes, H, W)
        depth = self.depth(img).squeeze(1)      # (B, H, W)

        return logits, depth
    
    def predict(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Used for inference, takes an image and returns class labels and normalized depth.
        This is what the metrics use as input (this is what the grader will use!).

        Args:
            x (torch.FloatTensor): image with shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            tuple of (torch.LongTensor, torch.FloatTensor):
                - pred: class labels {0, 1, 2} with shape (b, h, w)
                - depth: normalized depth [0, 1] with shape (b, h, w)
        """
        logits, raw_depth = self(x)
        pred = logits.argmax(dim=1)

        # Optional additional post-processing for depth only if needed
        depth = raw_depth

        return pred, depth

MODEL_FACTORY = {
    "classifier": Classifier,
    "detector": Detector,
}

def load_model(
    model_name: str,
    with_weights: bool = False,
    **model_kwargs,
) -> torch.nn.Module:
    """
    Called by the grader to load a pre-trained model by name
    """
    m = MODEL_FACTORY[model_name](**model_kwargs)

    if with_weights:
        model_path = HOMEWORK_DIR / f"{model_name}.th"
        assert model_path.exists(), f"{model_path.name} not found"

        try:
            m.load_state_dict(torch.load(model_path, map_location="cpu"))
        except RuntimeError as e:
            raise AssertionError(
                f"Failed to load {model_path.name}, make sure the default model arguments are set correctly"
            ) from e

    # limit model sizes since they will be zipped and submitted
    model_size_mb = calculate_model_size_mb(m)

    if model_size_mb > 20:
        raise AssertionError(f"{model_name} is too large: {model_size_mb:.2f} MB")

    return m


def save_model(model: torch.nn.Module) -> str:
    """
    Use this function to save your model in train.py
    """
    model_name = None

    for n, m in MODEL_FACTORY.items():
        if type(model) is m:
            model_name = n

    if model_name is None:
        raise ValueError(f"Model type '{str(type(model))}' not supported")

    output_path = HOMEWORK_DIR / f"{model_name}.th"
    torch.save(model.state_dict(), output_path)

    return output_path


def calculate_model_size_mb(model: torch.nn.Module) -> float:
    """
    Args:
        model: torch.nn.Module

    Returns:
        float, size in megabytes
    """
    return sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024


def debug_model(batch_size: int = 1):
    """
    Test your model implementation

    Feel free to add additional checks to this function -
    this function is NOT used for grading
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sample_batch = torch.rand(batch_size, 3, 64, 64).to(device)

    print(f"Input shape: {sample_batch.shape}")

    model = load_model("classifier", in_channels=3, num_classes=6).to(device)
    output = model(sample_batch)

    # should output logits (b, num_classes)
    print(f"Output shape: {output.shape}")


if __name__ == "__main__":
    debug_model()
