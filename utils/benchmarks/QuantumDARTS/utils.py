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



def make_simple_multiclass_data(n_samples=1000, n_features=4, n_classes=4, random_state=42, cluster_std=0.5, separation_scale=3.0):
    """
    Generates a simple synthetic multiclass classification dataset with well-separated clusters.

    This function creates n_classes clusters by distributing their centers along the axes 
    of the feature space, ensuring they are distinct and separated.

    Args:
        n_samples (int): Total number of samples to generate.
        n_features (int): The number of features for each sample.
        n_classes (int): The number of distinct classes (clusters) to generate.
        random_state (int): Seed for the random number generator for reproducibility.
        cluster_std (float): The standard deviation of the Gaussian clusters. Smaller values 
                             lead to tighter, more easily separable clusters.
        separation_scale (float): A factor controlling the distance of cluster centers from the origin.
                                  Larger values result in greater separation between classes.
    """
    
    # This method of creating centers works for up to 2 * n_features classes.
    if n_classes > 2 * n_features:
        raise ValueError(
            f"Cannot create {n_classes} unique centers for {n_features} features "
            f"with this method. Maximum supported classes is {2 * n_features}."
        )

    # Use a modern random number generator for better practice
    rng = np.random.default_rng(random_state)
    
    all_X_parts = []
    all_y_parts = []

    # Distribute samples evenly among classes, handling remainders
    samples_per_class = [n_samples // n_classes] * n_classes
    for i in range(n_samples % n_classes):
        samples_per_class[i] += 1

    for i in range(n_classes):
        # --- Create a unique center for each class ---
        center = np.zeros(n_features)
        # Cycle through feature dimensions to place the center
        feature_idx = i % n_features
        # Use positive and negative directions along the axis
        sign = (-1)**(i // n_features)
        center[feature_idx] = sign * separation_scale
        
        # Generate Gaussian data points around the center
        n_class_samples = samples_per_class[i]
        X_class = (rng.standard_normal((n_class_samples, n_features)) * cluster_std) + center
        y_class = np.full(n_class_samples, i) # Label is the class index

        all_X_parts.append(X_class)
        all_y_parts.append(y_class)

    # Combine the data from all classes into single arrays
    X = np.vstack(all_X_parts)
    y = np.hstack(all_y_parts)
    
    # Shuffle the entire dataset to mix the classes
    shuffle_idx = rng.permutation(n_samples)
    X, y = X[shuffle_idx], y[shuffle_idx]

    # Scale features to a range suitable for angle encoding in quantum circuits
    scaler = MinMaxScaler(feature_range=(0, np.pi))
    X_scaled = scaler.fit_transform(X)

    # --- Split data into training, validation, and test sets ---
    test_size = 0.15
    val_size = 0.15

    # First split to separate training data from a temporary set (validation + test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_scaled, y,
        test_size=(val_size + test_size),
        random_state=random_state,
        stratify=y  # Stratify to ensure class proportions are maintained
    )
    
    # Second split to separate the temporary set into validation and test sets
    relative_test_size = test_size / (val_size + test_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=relative_test_size,
        random_state=random_state,
        stratify=y_temp
    )
    
    return X_train, y_train, X_val, y_val, X_test, y_test



from qiskit.circuit import QuantumCircuit, Parameter


def qiskit_to_matrix(circuit: QuantumCircuit) -> list[list[str]]:
    """
    Converts a Qiskit QuantumCircuit into a matrix-like list of strings,
    preserving the parallel "layered" structure seen in the circuit diagram.

    This function understands that gates on different qubits can occur in the
    same time step (layer) and structures the matrix accordingly. This is
    essential for accurately representing and reconstructing circuits.

    Args:
        circuit: The input Qiskit QuantumCircuit object.

    Returns:
        A list of lists of strings representing the layered circuit structure.
    """
    num_qubits = circuit.num_qubits
    if num_qubits == 0:
        return []

    qubit_map = {qubit: i for i, qubit in enumerate(circuit.qubits)}
    
    # Initialize matrix with one empty list per qubit
    matrix = [[] for _ in range(num_qubits)]
    
    # Tracks the number of columns filled for each qubit's timeline
    # This tells us when each qubit is next available.
    qubit_available_at_col = [0] * num_qubits
    
    multi_qubit_gate_counter = 1
    gate_name_map = {'r': 'rx', 'u1': 'rz', 'x': 'x', 'rz': 'rz', 'rx': 'rx'}

    for instruction in circuit.data:
        op = instruction.operation
        q_args = instruction.qubits

        if not q_args or op.name in ['barrier', 'measure']:
            continue

        op_qubit_indices = [qubit_map[q] for q in q_args]

        # Determine the column where this new gate should start.
        # It's the first column where all its qubits are available.
        start_col = 0
        for idx in op_qubit_indices:
            start_col = max(start_col, qubit_available_at_col[idx])

        # --- Pad all rows with empty strings up to the start column ---
        for i in range(num_qubits):
            padding_needed = start_col - len(matrix[i])
            if padding_needed > 0:
                matrix[i].extend([''] * padding_needed)
        
        # --- Place the gate strings in the correct rows at start_col ---
        if op.num_qubits > 1:
            if op.name == 'cx':
                control_idx, target_idx = op_qubit_indices
                matrix[control_idx].append(f'cx_{multi_qubit_gate_counter}c')
                matrix[target_idx].append(f'cx_{multi_qubit_gate_counter}t')
                multi_qubit_gate_counter += 1
            else:
                # Handle other multi-qubit gates if necessary
                print(f"Warning: Unsupported multi-qubit gate '{op.name}' found.")
                for idx in op_qubit_indices:
                    matrix[idx].append(f'{op.name}_{idx}') # Generic placeholder

        elif op.num_qubits == 1:
            qubit_index = op_qubit_indices[0]
            gate_name = gate_name_map.get(op.name, op.name)
            matrix[qubit_index].append(gate_name)

        # --- Update qubit availability and fill gaps for non-participating qubits ---
        for idx in op_qubit_indices:
            qubit_available_at_col[idx] = start_col + 1
        
        # Ensure all rows are of the same length after adding the gate
        max_len = max(len(row) for row in matrix)
        for i, row in enumerate(matrix):
            if len(row) < max_len:
                row.append('')
                qubit_available_at_col[i] = max_len


    # Final padding to make the matrix rectangular
    max_len = max(len(row) for row in matrix)
    for row in matrix:
        padding_needed = max_len - len(row)
        if padding_needed > 0:
            row.extend([''] * padding_needed)
            
    return matrix

import json

def write_matrix_to_json(matrix: list[list[str]], filename: str):
    """Writes the circuit matrix to a JSON file."""
    with open(filename, 'w') as f:
        json.dump(matrix, f, indent=2) # indent for readability
    print(f"Matrix successfully written to {filename}")

def read_matrix_from_json(filename: str) -> list[list[str]]:
    """Reads a circuit matrix from a JSON file."""
    with open(filename, 'r') as f:
        matrix = json.load(f)
    print(f"Matrix successfully read from {filename}")
    return matrix
