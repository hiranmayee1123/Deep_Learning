# Importing necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from google.colab import drive

# Mounting Google Drive
drive.mount('/content/drive')

# Define the path to your files
path = '/content/drive/MyDrive/CPSC-8430-Deep-Learning-001/HW4/'

# Loading FID scores from numpy files
fid_dcgan = np.load(path + 'Results/FID_score/DC_FID.npy')
fid_wgangp = np.load(path + 'Results/FID_score/WGANGP_FID.npy')
fid_acgan = np.load(path + 'Results/FID_score/ACGAN_FID.npy')

# Plotting the FID scores for comparison
plt.figure(figsize=(10, 5))
plt.title("FID Score Comparisons: DCGAN vs WGAN-GP vs ACGAN")
plt.plot(fid_dcgan, label="DCGAN", color='red')
plt.plot(fid_wgangp, label="WGAN-GP", color='green')
plt.plot(fid_acgan, label="ACGAN", color='blue')
plt.xlabel("Epochs")
plt.ylabel("FID Score")
plt.legend()

# Saving the plot as a JPEG image
output_path = path + 'Results/FID_summary.jpg'
plt.savefig(output_path, format='jpeg', dpi=100, bbox_inches='tight')

# Display the plot
plt.show()

# Displaying key statistics for WGAN-GP FID scores
min_fid = np.min(fid_wgangp)
mean_fid = np.mean(fid_wgangp)
fid_epoch_40 = fid_wgangp[39]  # Note: Python uses 0-based indexing
min_fid, mean_fid, fid_epoch_40
