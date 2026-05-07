import os
import cv2
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2


# =========================================================
# Configuration
# =========================================================
config = {
    'lr': 1e-3,
    'wd': 1e-2,
    'bs': 256,
    'img_size': 512,
    'epochs': 100,
    'seed': 1000
}


# =========================================================
# Seed
# =========================================================
def seed_everything(seed=42):
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


seed_everything(seed=config['seed'])


# =========================================================
# Transforms
# =========================================================
def get_train_transforms():
    return A.Compose([
        A.Resize(config['img_size'], config['img_size']),
        A.Normalize(),
        ToTensorV2(p=1.0)
    ])


# =========================================================
# Dataset
# =========================================================
class ImageNetDataset(Dataset):

    def __init__(self, paths, augmentations):
        self.paths = paths
        self.augmentations = augmentations

    def __getitem__(self, idx):
        path = self.paths[idx]

        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.augmentations:
            augmented = self.augmentations(image=image)
            image = augmented['image']

        return image

    def __len__(self):
        return len(self.paths)


# =========================================================
# Vector Quantization Module
# =========================================================
class VQ(nn.Module):

    def __init__(
        self,
        num_embeddings=512,
        embedding_dim=64,
        commitment_cost=0.25
    ):
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost

        self.embeddings = nn.Embedding(
            self.num_embeddings,
            self.embedding_dim
        )

        self.embeddings.weight.data.uniform_(
            -1 / self.num_embeddings,
            1 / self.num_embeddings
        )

    def forward(self, inputs):
        inputs = inputs.permute(0, 2, 3, 1).contiguous()

        input_shape = inputs.shape

        flat_inputs = inputs.view(-1, self.embedding_dim)

        distances = torch.cdist(
            flat_inputs,
            self.embeddings.weight
        )

        encoding_index = torch.argmin(distances, dim=1)

        quantized = torch.index_select(
            self.embeddings.weight,
            0,
            encoding_index
        ).view(input_shape)

        e_latent_loss = F.mse_loss(
            quantized.detach(),
            inputs
        )

        q_latent_loss = F.mse_loss(
            quantized,
            inputs.detach()
        )

        commitment_loss = (
            q_latent_loss +
            self.commitment_cost * e_latent_loss
        )

        quantized = inputs + (quantized - inputs).detach()

        quantized = quantized.permute(0, 3, 1, 2).contiguous()

        return commitment_loss, quantized


# =========================================================
# Residual Blocks
# =========================================================
class ResidualBlock(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_channels
    ):
        super(ResidualBlock, self).__init__()

        self.resblock = nn.Sequential(
            nn.ReLU(inplace=True),

            nn.Conv2d(
                in_channels,
                hidden_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            ),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                hidden_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                bias=False
            )
        )

    def forward(self, x):
        return x + self.resblock(x)


class ResidualStack(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_channels,
        num_res_layers
    ):
        super(ResidualStack, self).__init__()

        self.num_res_layers = num_res_layers

        self.layers = nn.ModuleList([
            ResidualBlock(
                in_channels,
                out_channels,
                hidden_channels
            )
            for _ in range(num_res_layers)
        ])

    def forward(self, x):
        for i in range(self.num_res_layers):
            x = self.layers[i](x)

        return F.relu(x)


# =========================================================
# VQ-VAE Model
# =========================================================
class Model(nn.Module):

    def __init__(
        self,
        num_embeddings=512,
        embedding_dim=64,
        commitment_cost=0.25
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost

        # Encoder
        self.conv1 = nn.Conv2d(
            3,
            64,
            kernel_size=4,
            stride=2,
            padding=1
        )

        self.conv2 = nn.Conv2d(
            64,
            128,
            kernel_size=4,
            stride=2,
            padding=1
        )

        self.conv3 = nn.Conv2d(
            128,
            128,
            kernel_size=3,
            stride=1,
            padding=1
        )

        self.resblock1 = ResidualStack(
            128,
            128,
            64,
            3
        )

        # Vector quantization
        self.vq_conv = nn.Conv2d(
            128,
            self.embedding_dim,
            kernel_size=1,
            stride=1
        )

        self.vq = VQ(
            self.num_embeddings,
            self.embedding_dim,
            self.commitment_cost
        )

        # Decoder
        self.conv4 = nn.Conv2d(
            self.embedding_dim,
            64,
            kernel_size=3,
            stride=1,
            padding=1
        )

        self.resblock2 = ResidualStack(
            64,
            64,
            32,
            3
        )

        self.conv5 = nn.ConvTranspose2d(
            64,
            32,
            kernel_size=4,
            stride=2,
            padding=1
        )

        self.conv6 = nn.ConvTranspose2d(
            32,
            3,
            kernel_size=4,
            stride=2,
            padding=1
        )

    def encode(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        x = self.conv3(x)

        x = self.resblock1(x)

        return x

    def decode(self, quantized):
        x = self.conv4(quantized)

        x = self.resblock2(x)

        x = F.relu(self.conv5(x))

        x = self.conv6(x)

        return x

    def forward(self, inputs):
        x = self.encode(inputs)

        commitment_loss, quantized = self.vq(
            self.vq_conv(x)
        )

        outputs = self.decode(quantized)

        reconstruction_loss = F.mse_loss(
            outputs,
            inputs
        )

        loss = reconstruction_loss + commitment_loss

        return loss, outputs, reconstruction_loss


# =========================================================
# Load Pretrained VQ-VAE
# =========================================================
model = Model()

model.load_state_dict(
    torch.load(
        'vqvae.bin',
        map_location='cpu'
    )
)


# =========================================================
# Reconstruction + Latent Perturbation
# =========================================================
def get_rec_image(
    filename,
    output_path,
    perturb_dim=None,
    epsilon=0.01
):
    """
    Generate reconstructed image with optional latent perturbation.

    Args:
        filename:
            Input image path.

        output_path:
            Output image path.

        perturb_dim:
            Latent channel index to perturb.

        epsilon:
            Perturbation magnitude.

    Returns:
        Reconstructed image as numpy.ndarray.
    """

    # Read image
    image = cv2.imread(filename)

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    # Preprocess
    augmented = get_train_transforms()(image=image)

    image = augmented['image'].unsqueeze(0)

    with torch.no_grad():

        # Encode
        x = model.encode(image)

        # Quantize
        _, quantized = model.vq(
            model.vq_conv(x)
        )

        # Optional latent perturbation
        if (
            perturb_dim is not None and
            0 <= perturb_dim < quantized.shape[1]
        ):
            quantized = quantized.clone()

            quantized[:, perturb_dim, :, :] += epsilon

        # Decode
        predictions = model.decode(quantized)

    # Convert back to uint8 image
    predictions = (
        predictions.squeeze(0)
        .detach()
        .cpu() * 255.0
    )

    rec_image = predictions.permute(
        1,
        2,
        0
    ).numpy()

    rec_image = np.clip(
        rec_image,
        0,
        255
    ).astype(np.uint8)

    rec_image = cv2.cvtColor(
        rec_image,
        cv2.COLOR_RGB2BGR
    )

    # Ensure output directory exists
    os.makedirs(
        os.path.dirname(output_path),
        exist_ok=True
    )

    cv2.imwrite(output_path, rec_image)

    return rec_image
