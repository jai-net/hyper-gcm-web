# =======================
# hsi_classifier_app_optimized.py
# =======================
import torch
import torch.nn.functional as F
import joblib
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from tkinter import Tk, filedialog, messagebox

# =======================
# 1. Define GCN model
# =======================
from torch_geometric.nn import GCNConv

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

# =======================
# 2. GUI file selection
# =======================
root = Tk()
root.withdraw()
file_path = filedialog.askopenfilename(title="Select Salinas HSI .mat file")
if not file_path:
    messagebox.showinfo("HSI Classifier", "No file selected. Exiting.")
    exit()

# =======================
# 3. Load HSI data
# =======================
data = scipy.io.loadmat(file_path)["salinas_corrected"]
h, w, b = data.shape
flat_data = data.reshape(-1, b)

# =======================
# 4. Load trained models
# =======================
scaler = joblib.load("scaler.pkl")
pca = joblib.load("pca.pkl")
svm = joblib.load("svm_model.pkl")

model = GCN(input_dim=30, hidden_dim=64, num_classes=16)
model.load_state_dict(torch.load("gcn_model.pt"))
model.eval()

# =======================
# 5. Preprocess
# =======================
print("✅ Preprocessing...")
X_std = scaler.transform(flat_data)
X_pca = pca.transform(X_std)
all_x = torch.tensor(X_pca, dtype=torch.float32)

# =======================
# 6. Batch processing for GCN embeddings
# =======================
print("✅ Generating embeddings in batches...")
batch_size = 5000  # adjust depending on RAM
emb_list = []

for i in range(0, all_x.shape[0], batch_size):
    batch = all_x[i:i+batch_size]
    with torch.no_grad():
        emb_batch = model.conv1(batch, torch.empty((2,0), dtype=torch.long)).relu().numpy()
    emb_list.append(emb_batch)

emb_all = np.vstack(emb_list)

# =======================
# 7. SVM prediction
# =======================
print("✅ Running SVM prediction...")
pred_full = svm.predict(emb_all)
pred_img = pred_full.reshape(h, w)

# =======================
# 8. Visualization & save
# =======================
print("✅ Visualizing and saving classification map...")
base_cmap = plt.cm.get_cmap("tab20", 16)
colors = [(0,0,0)] + list(base_cmap.colors)
cmap = ListedColormap(colors)

plt.figure(figsize=(8, 8))
plt.imshow(pred_img, cmap=cmap, vmin=0, vmax=16)
plt.axis("off")
plt.title("Predicted Classification")
plt.tight_layout()
plt.savefig("classification_map.png", dpi=300)
plt.show(block=False)  # Non-blocking visualization

messagebox.showinfo("HSI Classifier", "Classification complete!\nSaved as classification_map.png")
