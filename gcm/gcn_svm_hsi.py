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
import tkinter as tk
from tkinter import filedialog

# =======================
# 0. Select MAT files
# =======================
print("📂 Please select the hyperspectral data and label MAT files")

root = tk.Tk()
root.withdraw()

data_path = filedialog.askopenfilename(title="Select Hyperspectral Data (.mat)")
label_path = filedialog.askopenfilename(title="Select Ground Truth Labels (.mat)")

if not data_path or not label_path:
    raise FileNotFoundError("❌ You must select both .mat files!")

print(f"✅ Selected Data File: {data_path}")
print(f"✅ Selected Label File: {label_path}")

# =======================
# 1. Load raw data
# =======================
print("✅ Loading data...")

def load_mat_auto(path):
    mat = scipy.io.loadmat(path)
    for key in mat.keys():
        if not key.startswith("__"):
            return mat[key]
    raise ValueError(f"No valid array found in {path}")

data = load_mat_auto(data_path)
labels = load_mat_auto(label_path)

if data.ndim != 3:
    raise ValueError("❌ Data file must be 3D (H, W, Bands)")
if labels.ndim != 2:
    raise ValueError("❌ Label file must be 2D (H, W)")

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
num_classes = len(np.unique(y_train))
model = GCN(input_dim=30, hidden_dim=64, num_classes=num_classes)
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
base_cmap = plt.cm.get_cmap("tab20", num_classes)
colors = [(0, 0, 0)] + list(base_cmap.colors)
cmap = ListedColormap(colors)

plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.imshow(labels, cmap=cmap, vmin=0, vmax=num_classes)
plt.title("Ground Truth")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(pred_img, cmap=cmap, vmin=0, vmax=num_classes)
plt.title("Predicted Classification")
plt.axis("off")

plt.tight_layout()
plt.savefig("classification_result.png", dpi=300, bbox_inches="tight")
plt.show()
