import numpy as np
import torch
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
import joblib
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load preprocessed train/test data
data_npz = np.load("data/preprocessed_data.npz")
X_train = data_npz["X_train"]
y_train = data_npz["y_train"]
X_test = data_npz["X_test"]
y_test = data_npz["y_test"]

print("✅ Data loaded.")
print("Train samples:", X_train.shape)
print("Test samples:", X_test.shape)

# Already PCA-transformed
X_test_pca = X_test
print("✅ Test data ready (already PCA-transformed).")

# Define the GCN model architecture
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
        # Return the hidden layer embedding
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        return x.detach().cpu().numpy()

# Load graph data and model weights
graph_data = torch.load("data/train_graph.pt", weights_only=False)

model = GCN(
    input_dim=graph_data.num_features,
    hidden_dim=64,
    num_classes=16
)

model.load_state_dict(torch.load("data/gcn_model.pth"))
model.eval()

# Move everything to CPU
device = torch.device("cpu")
graph_data = graph_data.to(device)
model = model.to(device)

# Extract embeddings for training nodes
with torch.no_grad():
    embeddings_train = model.embed(graph_data.x, graph_data.edge_index)

print("✅ Train embeddings shape:", embeddings_train.shape)

# Train SVM classifier on embeddings
svm = SVC(C=1.0, kernel="rbf", gamma="scale")
svm.fit(embeddings_train, y_train)
joblib.dump(svm, "data/svm.pkl")
print("✅ SVM model saved to data/svm.pkl.")
print("✅ SVM trained on GCN embeddings.")

# For test data: approximate embeddings without aggregation (empty edge_index)
with torch.no_grad():
    X_test_tensor = torch.tensor(X_test_pca, dtype=torch.float)
    embeddings_test = model.conv1(
        X_test_tensor,
        torch.empty((2, 0), dtype=torch.long)
    ).relu()
    embeddings_test = embeddings_test.detach().cpu().numpy()

print("✅ Test embeddings shape:", embeddings_test.shape)

# Predict and evaluate
y_pred = svm.predict(embeddings_test)
accuracy = accuracy_score(y_test, y_pred)
np.save("data/predictions.npy", y_pred)

print("🎯 Test Accuracy: {:.4f}".format(accuracy))
