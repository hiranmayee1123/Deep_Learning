import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
image_size = 64
channels_img = 3
z_dim = 100
batch_size = 128
learning_rate = 1e-4
num_epochs = 200
lambda_gp = 10  # Gradient penalty coefficient
num_critic = 5  # Number of critic steps per generator step

# Data preparation
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * channels_img, [0.5] * channels_img),
])

dataset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Visualize a batch of images
real_batch = next(iter(dataloader))
plt.figure(figsize=(8, 8))
plt.axis("off")
plt.title("Training Images")
plt.imshow(np.transpose(torchvision.utils.make_grid(real_batch[0][:64], padding=2, normalize=True).cpu(), (1, 2, 0)))
plt.show()

# Initialize weights
def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)

# Generator
class Generator(nn.Module):
    def __init__(self, z_dim, channels_img):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            self._block(z_dim, 512, 4, 1, 0),  # 4x4
            self._block(512, 256, 4, 2, 1),   # 8x8
            self._block(256, 128, 4, 2, 1),   # 16x16
            self._block(128, 64, 4, 2, 1),    # 32x32
            nn.ConvTranspose2d(64, channels_img, 4, 2, 1),  # 64x64
            nn.Tanh(),
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.gen(x)

# Critic
class Critic(nn.Module):
    def __init__(self, channels_img):
        super(Critic, self).__init__()
        self.critic = nn.Sequential(
            self._block(channels_img, 64, 4, 2, 1),  # 32x32
            self._block(64, 128, 4, 2, 1),           # 16x16
            self._block(128, 256, 4, 2, 1),          # 8x8
            self._block(256, 512, 4, 2, 1),          # 4x4
            nn.Conv2d(512, 1, 4, 1, 0),              # 1x1
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.critic(x)

# Gradient penalty
def gradient_penalty(critic, real, fake, device):
    batch_size, c, h, w = real.shape
    alpha = torch.rand((batch_size, 1, 1, 1), device=device).repeat(1, c, h, w)
    interpolated_images = real * alpha + fake * (1 - alpha)
    interpolated_images.requires_grad_(True)

    mixed_scores = critic(interpolated_images)
    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores, device=device),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradient = gradient.view(gradient.size(0), -1)
    gradient_norm = gradient.norm(2, dim=1)
    penalty = torch.mean((gradient_norm - 1) ** 2)
    return penalty

# Initialize models
gen = Generator(z_dim, channels_img).to(device)
critic = Critic(channels_img).to(device)
initialize_weights(gen)
initialize_weights(critic)

# Optimizers
opt_gen = optim.Adam(gen.parameters(), lr=learning_rate, betas=(0.0, 0.9))
opt_critic = optim.Adam(critic.parameters(), lr=learning_rate, betas=(0.0, 0.9))

# Training loop
fixed_noise = torch.randn(32, z_dim, 1, 1).to(device)

for epoch in range(num_epochs):
    for batch_idx, (real, _) in enumerate(dataloader):
        real = real.to(device)
        cur_batch_size = real.size(0)
        noise = torch.randn(cur_batch_size, z_dim, 1, 1).to(device)
        fake = gen(noise)

        # Train Critic
        for _ in range(num_critic):
            critic_real = critic(real).reshape(-1)
            critic_fake = critic(fake.detach()).reshape(-1)
            gp = gradient_penalty(critic, real, fake, device)
            loss_critic = -(torch.mean(critic_real) - torch.mean(critic_fake)) + lambda_gp * gp
            critic.zero_grad()
            loss_critic.backward()
            opt_critic.step()

        # Train Generator
        gen_fake = critic(fake).reshape(-1)
        loss_gen = -torch.mean(gen_fake)
        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()

    # Logging
    print(f"Epoch [{epoch+1}/{num_epochs}] Loss D: {loss_critic:.4f}, loss G: {loss_gen:.4f}")

    # Save progress
    if epoch % 10 == 0:
        fake_images = gen(fixed_noise)
        save_image(fake_images, f"generated_{epoch}.png", normalize=True)
