# Importing necessary libraries
import torch
import torchvision
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.utils as utils
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import models
from scipy import linalg
import os

# Create directories to store results and models
os.makedirs('Results', exist_ok=True)
os.makedirs('Models', exist_ok=True)

# Set random seed and device
torch.manual_seed(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Device:', device)

# Define hyperparameters
LEARNING_RATE = 5e-5
BATCH_SIZE = 64
IMAGE_SIZE = 64
CHANNELS_IMG = 3
NOISE_DIM = 100
NUM_EPOCHS = 40
FEATURES_DISC = 64
FEATURES_GEN = 64
NUM_CLASSES = 10
EMBED_SIZE = 100
CRITIC_ITERS = 5

# Load and preprocess the CIFAR-10 dataset
dataset = datasets.CIFAR10(
    root="./dataset/CIFAR10data",
    download=True,
    transform=transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

# Display a batch of training images
real_batch = next(iter(dataloader))
plt.figure(figsize=(10, 10))
plt.axis("off")
plt.title("Training Images")
plt.imshow(np.transpose(utils.make_grid(real_batch[0][:BATCH_SIZE], padding=2, normalize=True).cpu(), (1, 2, 0)))

# Define the Generator class
class Generator(nn.Module):
    def __init__(self, num_classes, img_size, embed_size, channels_noise, channels_img, features_g):
        super(Generator, self).__init__()
        self.img_size = img_size
        self.net = nn.Sequential(
            nn.ConvTranspose2d(NOISE_DIM + embed_size, features_g * 16, 4, 1, 0, bias=False),
            self._g_block(features_g * 16, features_g * 8, 4, 2, 1),
            self._g_block(features_g * 8, features_g * 4, 4, 2, 1),
            self._g_block(features_g * 4, features_g * 2, 4, 2, 1),
            nn.ConvTranspose2d(features_g * 2, channels_img, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )
        self.embed = nn.Embedding(num_classes, embed_size)

    def _g_block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels, momentum=0.9),
            nn.ReLU()
        )

    def forward(self, x, labels):
        embedding = self.embed(labels).unsqueeze(2).unsqueeze(3)
        x = torch.cat([x, embedding], dim=1)
        return self.net(x)

# Define the Discriminator class
class Discriminator(nn.Module):
    def __init__(self, num_classes, img_size, channels_img, features_d):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            nn.Conv2d(channels_img, features_d, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            self._d_block(features_d, features_d * 2, 4, 2, 1),
            self._d_block(features_d * 2, features_d * 4, 4, 2, 1),
            self._d_block(features_d * 4, features_d * 8, 4, 2, 1)
        )
        self.validity_layer = nn.Sequential(nn.Conv2d(512, 1, 4, 1, 0, bias=False), nn.Sigmoid())
        self.labels_layer = nn.Sequential(nn.Conv2d(512, num_classes + 1, 4, 1, 0, bias=False), nn.LogSoftmax(dim=1))
        self.embed = nn.Embedding(num_classes, img_size * img_size)

    def _d_block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        x = self.disc(x)
        valid = self.validity_layer(x).view(-1)
        label = self.labels_layer(x).view(-1, 11)
        return valid, label

# Initialize weights
def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)

# Instantiate and initialize the Generator and Discriminator
gen = Generator(NUM_CLASSES, IMAGE_SIZE, EMBED_SIZE, NOISE_DIM, CHANNELS_IMG, FEATURES_GEN).to(device)
disc = Discriminator(NUM_CLASSES, IMAGE_SIZE, CHANNELS_IMG, FEATURES_DISC).to(device)
initialize_weights(gen)
initialize_weights(disc)
gen.train()
disc.train()

# Define the InceptionV3 class for feature extraction
class InceptionV3(nn.Module):
    DEFAULT_BLOCK_INDEX = 3
    BLOCK_INDEX_BY_DIM = {64: 0, 192: 1, 768: 2, 2048: 3}

    def __init__(self, output_blocks=[DEFAULT_BLOCK_INDEX], resize_input=True, normalize_input=True, requires_grad=False):
        super(InceptionV3, self).__init__()
        self.resize_input = resize_input
        self.normalize_input = normalize_input
        self.output_blocks = sorted(output_blocks)
        self.last_needed_block = max(output_blocks)

        inception = models.inception_v3(pretrained=True)
        self.blocks = nn.ModuleList([
            nn.Sequential(inception.Conv2d_1a_3x3, inception.Conv2d_2a_3x3, inception.Conv2d_2b_3x3, nn.MaxPool2d(kernel_size=3, stride=2)),
            nn.Sequential(inception.Conv2d_3b_1x1, inception.Conv2d_4a_3x3, nn.MaxPool2d(kernel_size=3, stride=2)) if self.last_needed_block >= 1 else None,
            nn.Sequential(inception.Mixed_5b, inception.Mixed_5c, inception.Mixed_5d, inception.Mixed_6a, inception.Mixed_6b, inception.Mixed_6c, inception.Mixed_6d, inception.Mixed_6e) if self.last_needed_block >= 2 else None,
            nn.Sequential(inception.Mixed_7a, inception.Mixed_7b, inception.Mixed_7c, nn.AdaptiveAvgPool2d(output_size=(1, 1))) if self.last_needed_block >= 3 else None
        ])

        for param in self.parameters():
            param.requires_grad = requires_grad

    def forward(self, x):
        outp = []
        if self.resize_input:
            x = nn.functional.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
        if self.normalize_input:
            x = 2 * x - 1
        for idx, block in enumerate(self.blocks):
            if block:
                x = block(x)
                if idx in self.output_blocks:
                    outp.append(x)
        return outp

# Instantiate the InceptionV3 model for evaluation
block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
inception_model = InceptionV3([block_idx]).to(device)
def compute_activation_stats(images, model, batch_size=128, dims=2048, use_cuda=False):
    """
    Computes the activation statistics (mean and covariance) for a set of images.
    """
    model.eval()
    activations = np.empty((len(images), dims))
    
    # Transfer images to CUDA if enabled
    if use_cuda:
        images_batch = images.cuda()
    else:
        images_batch = images
    
    predictions = model(images_batch)[0]

    # Apply global average pooling if the model output is not scalar
    if predictions.size(2) != 1 or predictions.size(3) != 1:
        predictions = adaptive_avg_pool2d(predictions, output_size=(1, 1))
    
    activations = predictions.cpu().data.numpy().reshape(predictions.size(0), -1)
    mean = np.mean(activations, axis=0)
    covariance = np.cov(activations, rowvar=False)
    return mean, covariance

def compute_frechet_distance(mean1, cov1, mean2, cov2, epsilon=1e-6):
    """
    Computes the Frechet Distance between two multivariate Gaussians.
    """
    mean1, mean2 = np.atleast_1d(mean1), np.atleast_1d(mean2)
    cov1, cov2 = np.atleast_2d(cov1), np.atleast_2d(cov2)
    
    # Ensure matching dimensions
    assert mean1.shape == mean2.shape, "Mean vectors must have the same dimensions"
    assert cov1.shape == cov2.shape, "Covariance matrices must have the same dimensions"
    
    diff = mean1 - mean2
    sqrt_cov, _ = linalg.sqrtm(cov1.dot(cov2), disp=False)

    # Handle potential numerical instabilities
    if not np.isfinite(sqrt_cov).all():
        offset = np.eye(cov1.shape[0]) * epsilon
        sqrt_cov = linalg.sqrtm((cov1 + offset).dot(cov2 + offset))
    
    if np.iscomplexobj(sqrt_cov):
        sqrt_cov = sqrt_cov.real
    
    trace_sqrt_cov = np.trace(sqrt_cov)
    return diff.dot(diff) + np.trace(cov1) + np.trace(cov2) - 2 * trace_sqrt_cov

def calculate_fid(real_images, fake_images, model):
    """
    Computes the Frechet Inception Distance (FID) for real and generated images.
    """
    mean_real, cov_real = compute_activation_stats(real_images, model, use_cuda=True)
    mean_fake, cov_fake = compute_activation_stats(fake_images, model, use_cuda=True)
    return compute_frechet_distance(mean_real, cov_real, mean_fake, cov_fake)

# Training setup
fixed_noise = torch.randn(32, NOISE_DIM, 1, 1).to(device)
step = 0
discriminator_optimizer = optim.Adam(disc.parameters(), LEARNING_RATE, betas=(0.5, 0.999))
generator_optimizer = optim.Adam(gen.parameters(), LEARNING_RATE, betas=(0.5, 0.999))
loss_function = nn.BCELoss()

# Training statistics
gen_loss_list, disc_loss_list, fid_list, img_list = [], [], [], []
gen_loss, disc_loss = [], []

print("Training started...")
start_time = time.time()

for epoch in range(1, NUM_EPOCHS + 1):
    for batch_idx, (real_images, labels) in enumerate(dataloader):
        real_images, labels = real_images.to(device), labels.to(device)
        batch_size = real_images.size(0)
        fake_class_labels = 10 * torch.ones(batch_size, dtype=torch.long).to(device)

        # Train Discriminator
        disc.zero_grad()
        
        # Real image loss
        real_labels = torch.full((batch_size,), 1.0, dtype=torch.float).to(device)
        disc_real_output, real_class_preds = disc(real_images)
        real_image_loss = loss_function(disc_real_output, real_labels)
        real_label_loss = F.nll_loss(real_class_preds, labels)
        total_real_loss = real_image_loss + real_label_loss
        total_real_loss.backward()

        # Fake image loss
        noise = torch.randn(batch_size, NOISE_DIM, 1, 1).to(device)
        sample_labels = torch.randint(0, 10, (batch_size,), dtype=torch.long).to(device)
        fake_images = gen(noise, sample_labels)
        fake_labels = torch.full((batch_size,), 0.0, dtype=torch.float).to(device)
        disc_fake_output, fake_class_preds = disc(fake_images.detach())
        fake_image_loss = loss_function(disc_fake_output, fake_labels)
        fake_label_loss = F.nll_loss(fake_class_preds, fake_class_labels)
        total_fake_loss = fake_image_loss + fake_label_loss
        total_fake_loss.backward()

        # Update Discriminator
        discriminator_loss = (total_real_loss + total_fake_loss) / 2
        discriminator_optimizer.step()

        # Train Generator
        gen.zero_grad()
        noise = torch.randn(batch_size, NOISE_DIM, 1, 1).to(device)
        generated_images = gen(noise, sample_labels)
        gen_output, gen_class_preds = disc(generated_images)
        generator_loss = loss_function(gen_output, real_labels) + F.nll_loss(gen_class_preds, sample_labels)
        generator_loss.backward()
        generator_optimizer.step()

        # Record losses
        gen_loss.append(generator_loss.item())
        disc_loss.append(discriminator_loss.item())
        
        # Save generated images periodically
        if step % 500 == 0:
            with torch.no_grad():
                fake_images = gen(noise, sample_labels).detach().cpu()
            img_list.append(utils.make_grid(fake_images, normalize=True))
        step += 1

    # Calculate FID score
    fid_score = calculate_fid(real_images, fake_images, model)
    fid_list.append(fid_score)

    # Logging and saving progress
    if epoch % 5 == 0:
        print(f"Epoch {epoch}/{NUM_EPOCHS} | Discriminator Loss: {discriminator_loss:.4f} | Generator Loss: {generator_loss:.4f} | FID: {fid_score:.4f}")

    # Save images per epoch
    os.makedirs('Results/ACGAN_FAKE', exist_ok=True)
    os.makedirs('Results/ACGAN_REAL', exist_ok=True)
    utils.save_image(fake_images, f'Results/ACGAN_FAKE/epoch_{epoch:03d}.png', normalize=True)
    utils.save_image(real_images, f'Results/ACGAN_REAL/epoch_{epoch:03d}.png', normalize=True)

# Save training duration
end_time = time.time()
print(f"Training completed in {end_time - start_time:.2f} seconds.")

