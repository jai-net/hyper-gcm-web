import numpy as np
import torch
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from torch_geometric.utils import dense_to_sparse
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import kneighbors_graph
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import scipy.io
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
# =======================
# 1. Load raw data
# =======================
print("✅ Loading Salinas data...")
data = scipy.io.loadmat("data/Salinas_corrected.mat")["salinas_corrected"]
labels = scipy.io.loadmat("data/Salinas_gt.mat")["salinas_gt"]

h, w, b = data.shape
flat_data = data.reshape(-1, b)
flat_labels = labels.flatten()
mask = flat_labels > 0

X = flat_data[mask]
y = flat_labels[mask]

print("Data shape:", data.shape)
print("Labeled pixels:", X.shape[0])

# =======================
# 2. Standardize + PCA
# =======================
print("✅ Preprocessing...")
scaler = StandardScaler()
X_std = scaler.fit_transform(X)
pca = PCA(n_components=30, whiten=True, random_state=42)
X_pca = pca.fit_transform(X_std)

# =======================
# 3. Train/test split
# =======================
X_train, X_test, y_train, y_test = train_test_split(
    X_pca, y, test_size=0.2, stratify=y, random_state=42
)

# =======================
# 4. Graph Construction
# =======================
print("✅ Building k-NN graph...")
adj = kneighbors_graph(X_train, n_neighbors=10, mode="connectivity", include_self=False)
edge_index = torch.tensor(np.vstack((adj.nonzero()[0], adj.nonzero()[1])), dtype=torch.long)

# =======================
# 5. Define GCN
# =======================
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
    def embed(self, x, edge_index):
        x = self.conv1(x, edge_index)
        return F.relu(x).detach().cpu().numpy()

# =======================
# 6. Train GCN
# =======================
print("✅ Training GCN...")
graph_x = torch.tensor(X_train, dtype=torch.float)
graph_y = torch.tensor(y_train - 1, dtype=torch.long)
model = GCN(input_dim=30, hidden_dim=64, num_classes=16)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

model.train()
for epoch in range(200):
    optimizer.zero_grad()
    out = model(graph_x, edge_index)
    loss = F.cross_entropy(out, graph_y)
    loss.backward()
    optimizer.step()
    if epoch % 20 == 0:
        pred = out.argmax(dim=1).numpy()
        acc = (pred == graph_y.numpy()).mean()
        print(f"Epoch {epoch:03d}: Loss {loss:.4f} | Train Acc {acc:.4f}")

# =======================
# 7. Extract embeddings
# =======================
model.eval()
embeddings_train = model.embed(graph_x, edge_index)

# =======================
# 8. Train SVM
# =======================
print("✅ Training SVM...")
svm = SVC(C=1.0, kernel="rbf")
svm.fit(embeddings_train, y_train)

# =======================
# 9. Test embeddings
# =======================
with torch.no_grad():
    test_x = torch.tensor(X_test, dtype=torch.float)
    emb_test = model.conv1(test_x, torch.empty((2,0), dtype=torch.long)).relu().numpy()

# =======================
# 10. Evaluate
# =======================
y_pred = svm.predict(emb_test)
acc = accuracy_score(y_test, y_pred)
print(f"🎯 Test Accuracy: {acc:.4f}")

# =======================
# 11. Predict full map
# =======================
print("✅ Predicting full image...")
X_std_full = scaler.transform(flat_data)
X_pca_full = pca.transform(X_std_full)

with torch.no_grad():
    all_x = torch.tensor(X_pca_full, dtype=torch.float)
    emb_all = model.conv1(all_x, torch.empty((2,0), dtype=torch.long)).relu().numpy()

pred_full = np.zeros(flat_data.shape[0], dtype=int)
pred_full[mask] = svm.predict(emb_all[mask])
pred_img = pred_full.reshape(h, w)

# =======================
# 12. Visualize
# =======================
base_cmap = plt.cm.get_cmap("tab20", 16)  # 16 classes (1–16)
colors = [(0, 0, 0)] + list(base_cmap.colors)  # black + tab20 colors
cmap = ListedColormap(colors)

# Get unique labels (excluding background for legend)
unique_labels = np.unique(labels)
unique_labels = unique_labels[unique_labels != 0]

class_names = {
    1: "Brocoli_green_weeds_1",
    2: "Brocoli_green_weeds_2",
    3: "Fallow",
    4: "Fallow_rough_plow",
    5: "Fallow_smooth",
    6: "Stubble",
    7: "Celery",
    8: "Grapes_untrained",
    9: "Soil_vinyard_develop",
    10: "Corn_senesced_green_weeds",
    11: "Lettuce_romaine_4wk",
    12: "Lettuce_romaine_5wk",
    13: "Lettuce_romaine_6wk",
    14: "Lettuce_romaine_7wk",
    15: "Vinyard_untrained",
    16: "Vinyard_vertical_trellis"
}

# Build legend patches (skip background label 0)
legend_patches = []
for label in unique_labels:
    patch = mpatches.Patch(
        color=cmap(label),  # label maps to correct color in ListedColormap
        label=class_names.get(label, f"Class {label}")
    )
    legend_patches.append(patch)

plt.figure(figsize=(14, 6))

# Ground Truth
plt.subplot(1, 2, 1)
plt.imshow(labels, cmap=cmap, vmin=0, vmax=16)  # vmax=16 to include all classes
plt.title("Ground Truth")
plt.axis("off")

# Predicted
plt.subplot(1, 2, 2)
plt.imshow(pred_img, cmap=cmap, vmin=0, vmax=16)
plt.title("Predicted Classification")
plt.axis("off")

# Add legend
plt.legend(
    handles=legend_patches,
    loc="center left",
    bbox_to_anchor=(1.02, 0.5),
    title="Salinas Classes",
    fontsize=8,
    title_fontsize=9,
    frameon=False
)

plt.tight_layout()
plt.savefig("salinas_result_with_class_names.png", dpi=300, bbox_inches="tight")
plt.show()

