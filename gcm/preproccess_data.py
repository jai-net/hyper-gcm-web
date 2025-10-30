import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import scipy.io
import joblib

# Load .mat files
data = scipy.io.loadmat("data/Indian_pines_corrected.mat")["indian_pines_corrected"]
labels = scipy.io.loadmat("data/Indian_pines_gt.mat")["indian_pines_gt"]

print("✅ Raw data loaded:")
print("Data shape:", data.shape)   # (145,145,200)
print("Labels shape:", labels.shape)

# Flatten the spatial dimensions
h, w, b = data.shape
flat_data = data.reshape(-1, b)        # (145*145, 200)
flat_labels = labels.reshape(-1)       # (145*145,)

# Mask out unlabeled pixels (label==0)
mask = flat_labels > 0
X = flat_data[mask]
y = flat_labels[mask]

print("✅ After masking unlabeled pixels:")
print("X shape:", X.shape)
print("y shape:", y.shape)

# Standardize spectra (mean=0, std=1)
scaler = StandardScaler()
X_std = scaler.fit_transform(X)
print("✅ Data standardized.")

# Apply PCA (reduce to 30 components)
pca = PCA(n_components=30, whiten=True, random_state=42)
X_pca = pca.fit_transform(X_std)
print("✅ PCA completed.")
print("PCA shape:", X_pca.shape)

# Generate indices and split into train/test (e.g., 80/20)
indices = np.arange(X.shape[0])
train_idx, test_idx = train_test_split(
    indices, test_size=0.2, stratify=y, random_state=42
)

# Use indices to split data
X_train = X_pca[train_idx]
X_test = X_pca[test_idx]
y_train = y[train_idx]
y_test = y[test_idx]
print("✅ Data split complete.")
print("Train size:", X_train.shape[0])
print("Test size:", X_test.shape[0])

# Save processed data
np.savez("data/preprocessed_data.npz",
         X_train=X_train,
         X_test=X_test,
         y_train=y_train,
         y_test=y_test,
         train_idx=train_idx,
         test_idx=test_idx,
         mask=mask)

# Save scaler and PCA model
joblib.dump(scaler, "data/scaler.pkl")
joblib.dump(pca, "data/pca.pkl")

print("✅ Preprocessed data saved to data/preprocessed_data.npz.")
