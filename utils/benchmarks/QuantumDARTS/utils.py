from qiskit.circuit.library import RXGate, RZGate, CXGate
from qiskit.quantum_info import SparsePauliOp

import numpy as np
from sklearn.datasets import fetch_openml, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

from utils.data.preprocessing import data_pipeline

# Define the candidate gate pool
GATE_POOL = [
    {"name": "ID", "gate": None, "n_qubits": 1, "n_params": 0},
    {"name": "RX", "gate": RXGate, "n_qubits": 1, "n_params": 1},
    {"name": "RZ", "gate": RZGate, "n_qubits": 1, "n_params": 1},
    {"name": "CNOT", "gate": CXGate, "n_qubits": 2, "n_params": 0},
]


def load_mnist_data(n_features=8, random_state=42):
    """
    Loads and prepares the MNIST dataset for binary classification (0 vs 1).

    Args:
        n_features (int): The number of dimensions to reduce the data to using PCA.
        random_state (int): Seed for reproducibility.

    Returns:
        tuple: (X_train, y_train, X_test, y_test)
    """
    print("Loading MNIST dataset...")
    # Fetch MNIST from openml, which is more reliable than the old mldata.org
    mnist = fetch_openml('mnist_784', version=1, cache=True, parser='auto')
    X = mnist.data.astype('float32')
    y = mnist.target.astype('int64')
    print("Dataset loaded.")

    # Filter for digits 0 and 1
    mask = (y == 0) | (y == 1)
    X_filtered = X[mask]
    # FIX: Convert the pandas Series to a NumPy array
    y_filtered = y[mask].to_numpy()

    print(f"Filtered for digits 0 and 1. Found {len(y_filtered)} samples.")

    # Dimension reduction with PCA
    print(f"Performing PCA to reduce dimensions to {n_features}...")
    pca = PCA(n_components=n_features)
    X_pca = pca.fit_transform(X_filtered)

    # Scale features to be in the range [0, pi] for angle encoding
    scaler = MinMaxScaler(feature_range=(0, np.pi))
    X_scaled = scaler.fit_transform(X_pca)
    print("Data preprocessed and scaled.")

    # Split data according to paper's specified test size
    # Total samples for 0/1 is ~14780. Test size of 2115 is ~14.3%

    test_size = 0.15  # 15% for testing
    val_size = 0.15   # 15% for validation

    # Step 1: Split into training and a temporary set (val + test)
    # The size of the temporary set will be val_size + test_size = 30%
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_scaled, y_filtered, 
        test_size=(val_size + test_size), 
        random_state=random_state, 
        stratify=y_filtered
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, 
        test_size=(test_size / (val_size + test_size)), 
        random_state=random_state, 
        stratify=y_temp
    )

    print(f"Train shape: {X_train.shape}")
    print(f"Validation shape: {X_val.shape}")
    print(f"Test shape: {X_test.shape}")

    X_train = X_train[:100]
    y_train = y_train[:100]
    X_val = X_val[:100]
    y_val = y_val[:100]
    X_test = X_test[:100]
    y_test = y_test[:100]
    
    print(f"Data split into {len(y_train)} training and {len(y_test)} test samples.")
    return X_train, y_train, X_val, y_val, X_test, y_test


def load_multiclass_data(n_features=8, n_classes=4, n_samples=1000, random_state=42):
    """
    Generates a synthetic multiclass dataset.
    """
    print(f"Generating synthetic data: {n_samples} samples, {n_features} features, {n_classes} classes.")
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_features, # Make all features useful
        n_redundant=0,
        n_classes=n_classes,
        n_clusters_per_class=1,
        random_state=random_state
    )

    # Scale features to be in the range [0, pi] for angle encoding
    scaler = MinMaxScaler(feature_range=(0, np.pi))
    X_scaled = scaler.fit_transform(X)
    
    test_size = 0.15  # 15% for testing
    val_size = 0.15   # 15% for validation

    # Step 1: Split into training and a temporary set (val + test)
    # The size of the temporary set will be val_size + test_size = 30%
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_scaled, y, 
        test_size=(val_size + test_size), 
        random_state=random_state, 
        stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, 
        test_size=(test_size / (val_size + test_size)), 
        random_state=random_state, 
        stratify=y_temp
    )
    X_train = X_train[:100]
    y_train = y_train[:100]
    X_val = X_val[:100]
    y_val = y_val[:100]
    X_test = X_test[:100]
    y_test = y_test[:100]
    
    return X_train, y_train, X_val, y_val, X_test, y_test


def load_openml_data(openml_id, n_features=8, seed=42):
    print(f"[env] Loading OpenML data for dataset ID: {openml_id}")
    _, quantum_data, _, _, _ = data_pipeline(
        openml_dataset_id=int(openml_id),
        n_components=n_features,
        do_pca=True, # Assuming PCA is desired.
        seed=seed
    )
    (x_train_q, x_val_q, x_test_q, y_train_q, y_val_q, y_test_q, _, _, _) = quantum_data

    # Convert tensors to numpy and ensure labels are {-1, 1} for consistency
    X_train = x_train_q.numpy()
    X_val = x_val_q.numpy()
    X_test = x_test_q.numpy()
    y_train = y_train_q.numpy().astype(int)
    y_val = y_val_q.numpy().astype(int)
    y_test = y_test_q.numpy().astype(int)

    return X_train, y_train, X_val, y_val, X_test, y_test
