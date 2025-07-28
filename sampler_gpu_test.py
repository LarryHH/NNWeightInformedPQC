import os
import json
import time
import random
import numpy as np
import pandas as pd
import openml
from pathlib import Path
from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns

import qiskit
from qiskit import QuantumCircuit
from qiskit.circuit.library import TwoLocal, ZZFeatureMap
from qiskit.primitives import StatevectorSampler, Sampler
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.connectors import TorchConnector
from qiskit_machine_learning.gradients import (
    ParamShiftSamplerGradient,
    SPSASamplerGradient,
)
from qiskit.transpiler import PassManager

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
)

from utils.data.preprocessing import data_pipeline

from utils.nn.ClassicalNN import ClassicalNN, FlexibleNN
from utils.nn.QuantumNN import QuantumNN

import utils.ansatze.AnsatzExtractor as AnsatzExtractor

from utils.ansatze.HEA import HEA
from utils.ansatze.RealAmplitudes import RealAmplitudeAnsatz
from utils.ansatze.RandomPQC import RandomPQCAnsatz
from utils.ansatze.MPS import MPSAnsatz
from utils.ansatze.TTN import TTNAnsatz
from utils.ansatze.WIPQC import WeightInformedPQCAnsatz

# --- Configuration ---
SEED = 0
BATCH_SIZE = 32
GPU_OPTIONS = [False, True]
QUBIT_COUNTS = [2, 4, 6, 8]
GRADIENT_METHODS = ["spsa", "guided_spsa", "param_shift"]
OUTPUT_CSV_FILE = "results/qnn_benchmark_results.csv"

def set_seed(seed):
    """Set the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Set seed at the beginning
set_seed(SEED)

# --- Placeholder Classes (to make the script self-contained) ---
# NOTE: These are simplified versions based on your original code's structure.

# --- Main Benchmark Logic ---
results = []

# Check for CUDA availability once
cuda_available = torch.cuda.is_available()
if not cuda_available:
    print("[WARNING] CUDA is not available. GPU tests will be skipped.")

print(f"\n[INFO] Starting QNN performance benchmark...")
print(f"Configurations to test: {len(GPU_OPTIONS) * len(QUBIT_COUNTS) * len(GRADIENT_METHODS)}")
print("-" * 50)

# Create a progress bar for the outer loop
for use_gpu in tqdm(GPU_OPTIONS, desc="Device (GPU/CPU)"):
    if use_gpu and not cuda_available:
        continue

    for n_qubits in tqdm(QUBIT_COUNTS, desc=f"Qubits (GPU={use_gpu})", leave=False):
        for gradient_method in GRADIENT_METHODS:
            print(f"\n[TESTING] Qubits: {n_qubits}, Gradient: {gradient_method}, Use GPU: {use_gpu}")

            try:
                # 1. Instantiate the Ansatz and the QNN model
                hea_ansatz = HEA(n_qubits=n_qubits, depth=2)
                
                # We need a dummy value for num_classes, it doesn't affect the raw pass speed.
                num_classes = 2 
                
                # This object is just a wrapper, the raw QNN is inside
                model_wrapper = QuantumNN(
                    ansatz=hea_ansatz.get_ansatz(),
                    n_qubits=n_qubits,
                    num_classes=num_classes,
                    gradient_method=gradient_method,
                )
                
                # 2. Extract the raw Qiskit SamplerQNN object
                qnn = model_wrapper.model._neural_network
                
                # 3. Create sample data matching the QNN's requirements
                # The number of inputs for the QNN corresponds to n_qubits
                sample_features = np.random.rand(BATCH_SIZE, qnn.num_inputs)
                sample_weights = np.random.rand(qnn.num_weights)

                # 4. Time the raw forward pass
                t0 = time.time()
                _ = qnn.forward(sample_features, sample_weights)
                t1 = time.time()
                forward_time = t1 - t0
                print(f"  Raw Forward Pass ({BATCH_SIZE} samples): {forward_time:.4f} seconds")
                
                # 5. Time the raw backward pass (gradient calculation)
                t0 = time.time()
                _, _ = qnn.backward(sample_features, sample_weights)
                t1 = time.time()
                backward_time = t1 - t0
                print(f"  Raw Backward Pass ({BATCH_SIZE} samples): {backward_time:.4f} seconds")

                # Store results
                results.append({
                    "use_gpu": use_gpu,
                    "n_qubits": n_qubits,
                    "gradient_method": gradient_method,
                    "forward_time_s": forward_time,
                    "backward_time_s": backward_time,
                    "total_time_s": forward_time + backward_time,
                    "status": "Success"
                })

            except Exception as e:
                print(f"  [ERROR] Test failed for this configuration: {e}")
                results.append({
                    "use_gpu": use_gpu,
                    "n_qubits": n_qubits,
                    "gradient_method": gradient_method,
                    "forward_time_s": np.nan,
                    "backward_time_s": np.nan,
                    "status": f"Failed: {e}"
                })

# --- Save Results to CSV ---
print("\n" + "="*50)
print("Benchmark complete. Saving results...")

# Convert results list to a pandas DataFrame
results_df = pd.DataFrame(results)

# Save the DataFrame to a CSV file
results_df.to_csv(OUTPUT_CSV_FILE, index=False)

print(f"\nResults successfully saved to '{OUTPUT_CSV_FILE}'")
print("\nFinal Results Summary:")
print(results_df)
print("="*50)