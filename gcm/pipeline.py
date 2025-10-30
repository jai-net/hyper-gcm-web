# pipeline.py
import numpy as np
import torch
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import kneighbors_graph
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import scipy.io
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import io

# -------------------------------
# GCN Model
# -------------------------------
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

# -------------------------------
# Utility to auto-load .mat file
# -------------------------------
def load_mat_auto(path_or_bytes):
    mat = scipy.io.loadmat(path_or_bytes)
    for key in mat.keys():
        if not key.startswith("__"):
            return mat[key]
    raise ValueError("No valid variable found in .mat file")

# -------------------------------
# Main pipeline
# -------------------------------
def run_pipeline(data_bytes, gt_bytes):
    data = load_mat_auto(io.BytesIO(data_bytes))
    labels = load_mat_auto(io.BytesIO(gt_bytes))

    if data.ndim != 3:
        raise ValueError("Data file must be 3D (H, W, Bands)")
    if labels.ndim != 2:
        raise ValueError("Label file must be 2D (H, W)")

    h, w, b = data.shape
    flat_data = data.reshape(-1, b)
    flat_labels = labels.flatten()
    mask = flat_labels > 0

    X = flat_data[mask]
    y = flat_labels[mask]

    # Standardization + PCA
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)
    pca = PCA(n_components=30, whiten=True, random_state=42)
    X_pca = pca.fit_transform(X_std)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_pca, y, test_size=0.2, stratify=y, random_state=42
    )

    # Graph
    adj = kneighbors_graph(X_train, n_neighbors=10, mode="connectivity", include_self=False)
    edge_index = torch.tensor(np.vstack((adj.nonzero()[0], adj.nonzero()[1])), dtype=torch.long)

    # Train GCN
    num_classes = len(np.unique(y_train))
    model = GCN(input_dim=30, hidden_dim=64, num_classes=num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    graph_x = torch.tensor(X_train, dtype=torch.float)
    graph_y = torch.tensor(y_train - 1, dtype=torch.long)

    model.train()
    for epoch in range(100):
        optimizer.zero_grad()
        out = model(graph_x, edge_index)
        loss = F.cross_entropy(out, graph_y)
        loss.backward()
        optimizer.step()

    model.eval()
    embeddings_train = model.embed(graph_x, edge_index)

    # Train SVM
    svm = SVC(C=1.0, kernel="rbf")
    svm.fit(embeddings_train, y_train)

    # Test
    with torch.no_grad():
        test_x = torch.tensor(X_test, dtype=torch.float)
        emb_test = model.conv1(test_x, torch.empty((2,0), dtype=torch.long)).relu().numpy()
    y_pred = svm.predict(emb_test)
    acc = accuracy_score(y_test, y_pred)

    # Predict full image
    X_std_full = scaler.transform(flat_data)
    X_pca_full = pca.transform(X_std_full)
    with torch.no_grad():
        all_x = torch.tensor(X_pca_full, dtype=torch.float)
        emb_all = model.conv1(all_x, torch.empty((2,0), dtype=torch.long)).relu().numpy()
    pred_full = np.zeros(flat_data.shape[0], dtype=int)
    pred_full[mask] = svm.predict(emb_all[mask])
    pred_img = pred_full.reshape(h, w)

    # Visualization
    cmap = ListedColormap([(0,0,0)] + list(plt.cm.get_cmap("tab20", num_classes).colors))
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].imshow(labels, cmap=cmap, vmin=0, vmax=num_classes)
    ax[0].set_title("Ground Truth"); ax[0].axis("off")
    ax[1].imshow(pred_img, cmap=cmap, vmin=0, vmax=num_classes)
    ax[1].set_title("Predicted"); ax[1].axis("off")

    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png', bbox_inches="tight")
    buf.seek(0)
    plt.close()

    return acc, buf
