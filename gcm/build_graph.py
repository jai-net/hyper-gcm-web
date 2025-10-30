import numpy as np
from sklearn.neighbors import kneighbors_graph
import torch
from torch_geometric.data import Data

# Load preprocessed data
data = np.load("data/preprocessed_data.npz")
X_train = data["X_train"]
y_train = data["y_train"]

print("✅ Preprocessed data loaded:")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

# Build kNN graph (symmetric)
k = 10
knn_matrix = kneighbors_graph(X_train, k, mode='connectivity', include_self=False)
adj = knn_matrix.tocoo()

# Convert to PyTorch Geometric format
edge_index = torch.tensor([adj.row, adj.col], dtype=torch.long)
x = torch.tensor(X_train, dtype=torch.float)
y = torch.tensor(y_train, dtype=torch.long)

# Create the data object
data = Data(x=x, edge_index=edge_index, y=y)

print("✅ Graph created:")
print("Number of nodes:", data.num_nodes)
print("Number of edges:", data.num_edges)
print("Feature size:", data.num_features)
print("Class count:", len(torch.unique(data.y)))

# Save the graph object
torch.save(data, "data/train_graph.pt")
print("✅ Graph saved to data/train_graph.pt")
