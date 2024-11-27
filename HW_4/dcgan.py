import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import torchvision.utils as utils
import os
from scipy import linalg
from torch.utils.data import Subset
import torchvision.models as models
import torch.nn.functional as F
from IPython.display import HTML

# Set random seed and device
torch.manual_seed(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Constants for model training
LEARNING_RATE = 5e-5
BATCH_SIZE = 128
IMAGE_SIZE = 64
CHANNELS_IMG = 3  
NOISE_DIM = 100
NUM_EPOCHS = 40
FEATURES_DISC = 64 
FEATURES_GEN = 64 
BETA1 = 0.5

# Directory setup for results and models
os.makedirs('Results', exist_ok=True)
os.makedirs('Models', exist_ok=True)

# Load CIFAR-10 dataset and preprocess it
dataset = datasets.CIFAR10(
    root="./dataset/CIFAR10data",
    download=True,
    transform=transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize((0.5,) * 3, (0.5,) * 3)
    ])
)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

# Display a batch of training images
sample_batch = next(iter(dataloader))
plt.figure(figsize=(10, 10))
plt.axis("off")
plt.title("Training Images")
plt.imshow(np.transpose(utils.make_grid(sample_batch[0][:32], padding=2, normalize=True).cpu(), (1, 2, 0)))

# Discriminator network definition
class Discriminator(nn.Module):
    def __init__(self, input_channels, feature_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(input_channels, feature_dim, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            self._discriminator_block(feature_dim, feature_dim * 2),
            self._discriminator_block(feature_dim * 2, feature_dim * 4),
            self._discriminator_block(feature_dim * 4, feature_dim * 8),
            nn.Conv2d(feature_dim * 8, 1, kernel_size=4, stride=2, padding=0),
            nn.Sigmoid()
        )

    def _discriminator_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        return self.model(x)

# Generator network definition
class Generator(nn.Module):
    def __init__(self, noise_dim, output_channels, feature_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            self._generator_block(noise_dim, feature_dim * 16, 4, 1, 0),
            self._generator_block(feature_dim * 16, feature_dim * 8, 4, 2, 1),
            self._generator_block(feature_dim * 8, feature_dim * 4, 4, 2, 1),
            self._generator_block(feature_dim * 4, feature_dim * 2, 4, 2, 1),
            nn.ConvTranspose2d(feature_dim * 2, output_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def _generator_block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.model(x)

# Initialize weights for both models
def initialize_weights(model):
    for layer in model.modules():
        if isinstance(layer, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(layer.weight.data, 0.0, 0.02)

# Instantiate models, optimizers, and loss function
gen = Generator(NOISE_DIM, CHANNELS_IMG, FEATURES_GEN).to(device)
disc = Discriminator(CHANNELS_IMG, FEATURES_DISC).to(device)
initialize_weights(gen)
initialize_weights(disc)
optimizer_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(BETA1, 0.999))
optimizer_disc = optim.Adam(disc.parameters(), lr=LEARNING_RATE, betas=(BETA1, 0.999))
loss_fn = nn.BCELoss()
fixed_noise = torch.randn(32, NOISE_DIM, 1, 1).to(device)

# Pre-trained InceptionV3 model for FID calculation
class InceptionV3(nn.Module):
    DEFAULT_BLOCK_INDEX = 3
    BLOCK_INDEX_BY_DIM = {
        64: 0,
        192: 1,
        768: 2,
        2048: 3
    }

    def __init__(self, output_blocks=None, resize_input=True, normalize_input=True, requires_grad=False):
        super(InceptionV3, self).__init__()
        self.resize_input = resize_input
        self.normalize_input = normalize_input
        self.output_blocks = output_blocks or [self.DEFAULT_BLOCK_INDEX]
        self.last_needed_block = max(self.output_blocks)
        self.blocks = nn.ModuleList()

        inception = models.inception_v3(pretrained=True)
        self._build_blocks(inception)

        for param in self.parameters():
            param.requires_grad = requires_grad

    def _build_blocks(self, inception):
        self.blocks.append(nn.Sequential(
            inception.Conv2d_1a_3x3,
            inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2)
        ))
        if self.last_needed_block >= 1:
            self.blocks.append(nn.Sequential(
                inception.Conv2d_3b_1x1,
                inception.Conv2d_4a_3x3,
                nn.MaxPool2d(kernel_size=3, stride=2)
            ))
        if self.last_needed_block >= 2:
            self.blocks.append(nn.Sequential(
                inception.Mixed_5b, inception.Mixed_5c, inception.Mixed_5d,
                inception.Mixed_6a, inception.Mixed_6b, inception.Mixed_6c,
                inception.Mixed_6d, inception.Mixed_6e
            ))
        if self.last_needed_block >= 3:
            self.blocks.append(nn.Sequential(
                inception.Mixed_7a, inception.Mixed_7b, inception.Mixed_7c,
                nn.AdaptiveAvgPool2d(output_size=(1, 1))
            ))

    def forward(self, x):
        if self.resize_input:
            x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
        if self.normalize_input:
            x = 2 * x - 1
        outputs = []
        for idx, block in enumerate(self.blocks):
            x = block(x)
            if idx in self.output_blocks:
                outputs.append(x)
            if idx == self.last_needed_block:
                break
        return outputs

# Function to compute FID
def calculate_fid(real_images, fake_images, model):
    mu_real, sigma_real = calculate_activation_statistics(real_images, model, device=device)
    mu_fake, sigma_fake = calculate_activation_statistics(fake_images, model, device=device)
    return calculate_frechet_distance(mu_real, sigma_real, mu_fake, sigma_fake)
