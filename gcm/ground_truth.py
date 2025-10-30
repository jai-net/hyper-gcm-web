import numpy as np
import torch
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
import joblib
import matplotlib.pyplot as plt
import scipy.io
from matplotlib.patches import Patch

# ----------------------------------------------------------
# Load scaler and PCA models
# ----------------------------------------------------------
scaler = joblib.load("data/scaler.pkl")
pca = joblib.load("data/pca.pkl")

# ----------------------------------------------------------
# Load hyperspectral data and labels
# ----------------------------------------------------------
mat_data = scipy.io.loadmat("data/Indian_pines_corrected.mat")
mat_labels = scipy.io.loadmat("data/Indian_pines_gt.mat")

data = mat_data["indian_pines_corrected"]
labels = mat_labels["indian_pines_gt"]

h, w, b = data.shape
flat_data = data.reshape(-1, b)
flat_labels = labels.flatten()
mask_labeled = flat_labels > 0

# ----------------------------------------------------------
# Standardize and reduce dimensionality
# ----------------------------------------------------------
print("✅ Standardizing data...")
X_std = scaler.transform(flat_data)

print("✅ Applying PCA...")
X_pca = pca.transform(X_std)

# ----------------------------------------------------------
# Define GCN
# ----------------------------------------------------------
class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

# ----------------------------------------------------------
# Load model
# ----------------------------------------------------------
graph_data = torch.load("data/train_graph.pt", weights_only=False)
model = GCN(
    input_dim=graph_data.num_features,
    hidden_dim=64,
    num_classes=16
)
model.load_state_dict(torch.load("data/gcn_model.pth"))
model.eval()
model = model.to("cpu")

# ----------------------------------------------------------
# Extract embeddings
# ----------------------------------------------------------
with torch.no_grad():
    X_tensor = torch.tensor(X_pca, dtype=torch.float)
    embeddings_all = model.conv1(
        X_tensor, torch.empty((2, 0), dtype=torch.long)
    ).relu().cpu().numpy()

print("✅ Embeddings shape:", embeddings_all.shape)

# ----------------------------------------------------------
# Predict
# ----------------------------------------------------------
svm = joblib.load("data/svm.pkl")

print("✅ Predicting...")
preds_flat = np.zeros(flat_data.shape[0], dtype=int)
preds_flat[mask_labeled] = svm.predict(embeddings_all[mask_labeled])

# ----------------------------------------------------------
# Prepare maps
# ----------------------------------------------------------
prediction_map = preds_flat.reshape(h, w)
ground_truth_map = labels

# ----------------------------------------------------------
# Labels and colors
# ----------------------------------------------------------
class_labels = [
    "Alfalfa",
    "Corn-notill",
    "Corn-mintill",
    "Corn",
    "Grass-pasture",
    "Grass-trees",
    "Grass-pasture-mowed",
    "Hay-windrowed",
    "Oats",
    "Soybean-notill",
    "Soybean-mintill",
    "Soybean-clean",
    "Wheat",
    "Woods",
    "Buildings-Grass-Trees-Drives",
    "Stone-Steel-Towers"
]
cmap = plt.cm.get_cmap("nipy_spectral", 16)
cmap.set_bad(color="white")

# ----------------------------------------------------------
# Visualization side by side
# ----------------------------------------------------------
plt.figure(figsize=(16, 8))

# Ground Truth
plt.subplot(1, 2, 1)
masked_gt = np.ma.masked_where(ground_truth_map == 0, ground_truth_map - 1)
plt.imshow(masked_gt, cmap=cmap, vmin=0, vmax=15)
plt.title("Ground Truth Labels")
plt.axis("off")

# Predicted
plt.subplot(1, 2, 2)
masked_pred = np.ma.masked_where(prediction_map == 0, prediction_map - 1)
plt.imshow(masked_pred, cmap=cmap, vmin=0, vmax=15)
plt.title("Predicted Classification Map")
plt.axis("off")

# Legend
legend_elements = [
    Patch(facecolor=cmap(i / 15.0), edgecolor="black", label=f"{i+1}: {class_labels[i]}")
    for i in range(16)
]
plt.legend(
    handles=legend_elements,
    bbox_to_anchor=(1.05, 1),
    loc="upper left",
    borderaxespad=0.,
    title="Classes"
)

plt.tight_layout()
plt.savefig("ground_vs_predicted.png", dpi=300, bbox_inches="tight")
plt.show()

print("🎯 Comparison image saved as ground_vs_predicted.png")
