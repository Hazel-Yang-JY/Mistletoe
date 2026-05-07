import random
import numpy as np
import torch
import torchvision.transforms.functional as TF
import torchvision.transforms as T
from torchvision import transforms


# =========================================================
# Perturbation
# =========================================================
@torch.no_grad()
def perturb(x, strength=0.05):
    """
    Apply random perturbation to an image tensor.

    Args:
        x: torch.Tensor with shape [B, C, H, W] or [C, H, W],
           expected to be in range [-1, 1].
        strength: perturbation strength.

    Returns:
        torch.Tensor with the same shape as input, in range [-1, 1].
    """
    if not isinstance(x, torch.Tensor):
        raise TypeError("perturb expects a torch.Tensor input.")

    single = (x.dim() == 3)

    if single:
        x = x.unsqueeze(0)

    if x.dim() != 4:
        raise ValueError(
            f"Expected input shape [B, C, H, W] or [C, H, W], got {tuple(x.shape)}"
        )

    x = x.clone()
    batch_size = x.size(0)

    for i in range(batch_size):
        mode = random.choice(["gauss", "lowfreq"])

        if mode == "gauss":
            noise = torch.randn_like(x[i]) * strength
            x[i] = (x[i] + noise).clamp(-1, 1)

        elif mode == "lowfreq":
            noise = torch.randn_like(x[i]) * strength
            blur = TF.gaussian_blur(
                noise,
                kernel_size=11,
                sigma=(1.0, 2.0)
            )
            x[i] = (x[i] + blur).clamp(-1, 1)

    return x.squeeze(0) if single else x


# =========================================================
# Augmentation
# =========================================================
augment_transform = T.Compose([
    T.ToPILImage(),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomRotation(
        degrees=15,
        interpolation=transforms.InterpolationMode.BILINEAR
    ),
    T.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.1
    ),
    T.ToTensor()
])


@torch.no_grad()
def augment(x):
    """
    Apply random data augmentation.

    Supported inputs:
        1. numpy.ndarray with shape [H, W, C] or [H, W], uint8 or float
        2. torch.Tensor with shape [C, H, W]

    For torch.Tensor input:
        - If values are in [-1, 1], the output will also be in [-1, 1].
        - If values are in [0, 1], the output will also be in [0, 1].

    For numpy.ndarray input:
        - The output will be a torch.Tensor in range [0, 1].

    Returns:
        torch.Tensor with shape [C, H, W].
    """
    if isinstance(x, np.ndarray):
        if x.ndim == 2:
            x = np.expand_dims(x, axis=-1)

        if x.ndim != 3:
            raise ValueError(
                f"Expected numpy input with shape [H, W, C] or [H, W], got {x.shape}"
            )

        return augment_transform(x)

    if isinstance(x, torch.Tensor):
        if x.ndim != 3:
            raise ValueError(
                f"Expected tensor input with shape [C, H, W], got {tuple(x.shape)}"
            )

        if x.shape[0] not in [1, 3]:
            raise ValueError(
                f"Expected tensor channel dimension to be 1 or 3, got {x.shape[0]}"
            )

        x_min = float(x.min())
        x_max = float(x.max())

        input_is_minus_one_to_one = x_min < 0.0

        if input_is_minus_one_to_one:
            # Convert from [-1, 1] to [0, 1] for torchvision transforms
            x_in = ((x + 1.0) / 2.0).clamp(0, 1)
        else:
            # Assume input is already in [0, 1]
            x_in = x.clamp(0, 1)

        x_aug = augment_transform(x_in)

        if input_is_minus_one_to_one:
            # Convert back to [-1, 1]
            x_aug = x_aug * 2.0 - 1.0

        return x_aug

    raise TypeError(
        "augment expects either a numpy.ndarray or a torch.Tensor input."
    )