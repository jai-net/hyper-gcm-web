import numpy as np
import matplotlib.pyplot as plt

data = np.load("data/preprocessed_data.npz")
mask = data["mask"]
if mask.ndim == 1:
    mask = mask.reshape(145, 145)
test_idx = data["test_idx"]

preds = np.load("data/predictions.npy")

# Initialize full prediction map
prediction_map = np.zeros(mask.shape, dtype=np.uint8)

# Create flattened mask index array
flat_mask = mask.flatten()

# Get absolute indices (0–21024) of labeled pixels
labeled_indices = np.where(flat_mask)[0]

# Get absolute indices of only test pixels
test_indices_absolute = labeled_indices[test_idx]

# Flatten the prediction_map to assign predictions
prediction_map_flat = prediction_map.flatten()
prediction_map_flat[test_indices_absolute] = preds + 1  # +1 to separate from background

# Reshape for final visualization
prediction_map = prediction_map_flat.reshape(mask.shape)

# Plot
plt.figure(figsize=(6, 6))
plt.imshow(prediction_map, cmap='tab20')
plt.title("GCN + SVM Test Predictions")
plt.axis('off')
plt.show()
