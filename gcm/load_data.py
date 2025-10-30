import scipy.io
import numpy as np

# Load the hyperspectral image
data = scipy.io.loadmat('data/Indian_pines_corrected.mat')
image_cube = data['indian_pines_corrected']

# Load the ground truth labels
gt_data = scipy.io.loadmat('data/Indian_pines_gt.mat')
labels = gt_data['indian_pines_gt']

print("✅ Data loaded successfully!")
print("Hyperspectral image shape:", image_cube.shape)
print("Labels shape:", labels.shape)
print("Unique label values:", np.unique(labels))
