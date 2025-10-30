import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

# Load the graph data
data = data = torch.load("data/train_graph.pt", weights_only=False)


print("✅ Graph data loaded.")
print("Nodes:", data.num_nodes)
print("Edges:", data.num_edges)
print("Classes:", len(torch.unique(data.y)))

# Define a 2-layer GCN
class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x

# Instantiate the model
model = GCN(input_dim=data.num_features, hidden_dim=64, num_classes=len(torch.unique(data.y)))
print("✅ Model initialized.")

# Use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = data.to(device)
model = model.to(device)

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

# Training loop
model.train()
for epoch in range(1, 201):
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out, data.y - 1)  # labels need to start from 0
    loss.backward()
    optimizer.step()

    if epoch % 20 == 0:
        pred = out.argmax(dim=1)
        correct = (pred == (data.y - 1)).sum()
        acc = int(correct) / data.num_nodes
        print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | Train Accuracy: {acc:.4f}")

# Save the trained model
torch.save(model.state_dict(), "data/gcn_model.pth")
print("✅ GCN training complete and model saved.")
