# import os
# import json
# import time
# import random
# import numpy as np
# import pandas as pd
# import openml
# from pathlib import Path
# from tqdm import tqdm

# import matplotlib.pyplot as plt
# import seaborn as sns

# import qiskit
# from qiskit import QuantumCircuit
# from qiskit.circuit.library import TwoLocal, ZZFeatureMap
# from qiskit.primitives import StatevectorSampler, Sampler
# from qiskit_machine_learning.neural_networks import SamplerQNN
# from qiskit_machine_learning.connectors import TorchConnector
# from qiskit_machine_learning.gradients import (
#     ParamShiftSamplerGradient,
#     SPSASamplerGradient,
# )
# from qiskit.transpiler import PassManager

# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F
# from torch.utils.data import TensorDataset, DataLoader

# from sklearn.decomposition import PCA
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
# from sklearn.metrics import (
#     accuracy_score,
#     f1_score,
#     precision_score,
#     recall_score,
#     confusion_matrix,
#     classification_report,
# )

# from utils.data.preprocessing import data_pipeline

# from utils.nn.ClassicalNN import ClassicalNN, FlexibleNN
# from utils.nn.QuantumNN import QuantumNN

# import utils.ansatze.AnsatzExtractor as AnsatzExtractor

# from utils.ansatze.HEA import HEA
# from utils.ansatze.RealAmplitudes import RealAmplitudeAnsatz
# from utils.ansatze.RandomPQC import RandomPQCAnsatz
# from utils.ansatze.MPS import MPSAnsatz
# from utils.ansatze.TTN import TTNAnsatz
# from utils.ansatze.WIPQC import WeightInformedPQCAnsatz

# # --- Configuration ---
# SEED = 0
# NUM_BENCHMARK_RUNS = 10
# BATCH_SIZE = 32
# GPU_OPTIONS = [False]  # False for CPU, True for GPU
# QUBIT_COUNTS = [2, 4, 6, 8]  # Number of qubits to test
# GRADIENT_METHODS = ["spsa", "guided_spsa", "param_shift"]
# OUTPUT_CSV_FILE = "results/qnn_benchmark_results.csv"

# def set_seed(seed):
#     """Set the random seed for reproducibility."""
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(seed)

# # Set seed at the beginning
# set_seed(SEED)

# # --- Placeholder Classes (to make the script self-contained) ---
# # NOTE: These are simplified versions based on your original code's structure.

# # --- Main Benchmark Logic ---
# results = []

# # Check for CUDA availability once
# cuda_available = torch.cuda.is_available()
# if not cuda_available:
#     print("[WARNING] CUDA is not available. GPU tests will be skipped.")

# print(f"\n[INFO] Starting QNN performance benchmark...")
# print(f"Configurations to test: {len(GPU_OPTIONS) * len(QUBIT_COUNTS) * len(GRADIENT_METHODS)}")
# print("-" * 50)


# # Create a progress bar for the outer loop
# for use_gpu in tqdm(GPU_OPTIONS, desc="Device (GPU/CPU)"):
#     if use_gpu and not cuda_available:
#         continue

#     for n_qubits in tqdm(QUBIT_COUNTS, desc=f"Qubits (GPU={use_gpu})", leave=False):

#         print("\n[INFO] Loading data...")
#         _, quantum_data, input_dim, output_dim, is_multiclass = (
#             data_pipeline(
#                 187,
#                 batch_size=32,
#                 do_pca=True,
#                 use_gpu=use_gpu,
#                 n_components=n_qubits,
#                 seed=SEED,
#             )
#         )
#         (
#             x_train_q,
#             x_val_q,
#             x_test_q,
#             y_train_q,
#             y_val_q,
#             y_test_q,
#             train_loader_q,
#             val_loader_q,
#             test_loader_q,
#         ) = quantum_data

#         for gradient_method in GRADIENT_METHODS:
#             print(f"\n[TESTING] Qubits: {n_qubits}, Gradient: {gradient_method}, Use GPU: {use_gpu}")


#             try:
#                 # 1. Instantiate the Ansatz and the QNN model
#                 hea_ansatz = HEA(n_qubits=n_qubits, depth=2)
                
#                 # We need a dummy value for num_classes, it doesn't affect the raw pass speed.
#                 num_classes = 2 
                
#                 # This object is just a wrapper, the raw QNN is inside
#                 model_wrapper = QuantumNN(
#                     ansatz=hea_ansatz.get_ansatz(),
#                     n_qubits=n_qubits,
#                     num_classes=num_classes,
#                     gradient_method=gradient_method,
#                     use_gpu=use_gpu,
#                     seed=SEED,
#                     default_shots=1024,
#                 )
                
#                 # 2. Extract the raw Qiskit SamplerQNN object
#                 qnn = model_wrapper.model._neural_network
                
#                 # 3. Create sample data matching the QNN's requirements
#                 # The number of inputs for the QNN corresponds to n_qubits
#                 # sample_features = np.random.rand(BATCH_SIZE, qnn.num_inputs)
#                 # sample_weights = np.random.rand(qnn.num_weights)

#                 # 3. Use a real data batch for the benchmark
#                 # Extract one batch of features from the training dataloader
#                 real_features_tensor, _ = next(iter(train_loader_q))

#                 # Convert the PyTorch tensor to a NumPy array for the QNN
#                 # If the tensor is on the GPU, it must be moved to the CPU first
#                 sample_features = real_features_tensor.cpu().numpy()

#                 # Initialize model weights randomly (this is correct for a raw pass benchmark)
#                 sample_weights = np.random.rand(qnn.num_weights)

#                 print(f"  Using one batch of {sample_features.shape[0]} samples from the actual dataset.")

#                 # 4. Time the forward and backward passes over multiple runs
#                 forward_times = []
#                 backward_times = []

#                 print(f"  Running benchmark for {NUM_BENCHMARK_RUNS} passes...")
#                 for _ in range(NUM_BENCHMARK_RUNS):
#                     # Time the forward pass for one run
#                     t0 = time.time()
#                     _ = qnn.forward(sample_features, sample_weights)
#                     t1 = time.time()
#                     forward_times.append(t1 - t0)

#                     # Time the backward pass for one run
#                     t0 = time.time()
#                     _, _ = qnn.backward(sample_features, sample_weights)
#                     t1 = time.time()
#                     backward_times.append(t1 - t0)

#                 # Calculate the average and standard deviation from all the runs
#                 avg_forward_time = np.mean(forward_times)
#                 std_forward_time = np.std(forward_times)
#                 avg_backward_time = np.mean(backward_times)
#                 std_backward_time = np.std(backward_times)

#                 print(f"  Avg Forward Pass : {avg_forward_time:.4f} ± {std_forward_time:.4f} seconds")
#                 print(f"  Avg Backward Pass: {avg_backward_time:.4f} ± {std_backward_time:.4f} seconds")

#                 # Store the aggregated results
#                 results.append({
#                     "use_gpu": use_gpu,
#                     "n_qubits": n_qubits,
#                     "gradient_method": gradient_method,
#                     "avg_forward_s": avg_forward_time,
#                     "std_forward_s": std_forward_time,
#                     "avg_backward_s": avg_backward_time,
#                     "std_backward_s": std_backward_time,
#                     "total_avg_time_s": avg_forward_time + avg_backward_time,
#                     "status": "Success"
#                 })

#             except Exception as e:
#                 print(f"  [ERROR] Test failed for this configuration: {e}")
#                 results.append({
#                     "use_gpu": use_gpu,
#                     "n_qubits": n_qubits,
#                     "gradient_method": gradient_method,
#                     "forward_time_s": np.nan,
#                     "backward_time_s": np.nan,
#                     "status": f"Failed: {e}"
#                 })

# # --- Save Results to CSV ---
# print("\n" + "="*50)
# print("Benchmark complete. Saving results...")

# # Convert results list to a pandas DataFrame
# results_df = pd.DataFrame(results)

# # Save the DataFrame to a CSV file
# results_df.to_csv(OUTPUT_CSV_FILE, index=False)

# print(f"\nResults successfully saved to '{OUTPUT_CSV_FILE}'")
# print("\nFinal Results Summary:")
# print(results_df)
# print("="*50)



import os
import time
import random
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.optim as optim

# --- Local/Custom imports ---
# Assuming these custom modules are in your project structure
from utils.data.preprocessing import data_pipeline
from utils.nn.QuantumNN import QuantumNN
from utils.ansatze.HEA import HEA


# --- Configuration ---
SEED = 0
NUM_FIT_EPOCHS = 5  # Number of epochs to run in the fit() benchmark
BATCH_SIZE = 32
GPU_OPTIONS = [False, True]
QUBIT_COUNTS = [2, 4, 6, 8]
GRADIENT_METHODS = ["spsa", "guided_spsa", "param_shift"]
OUTPUT_CSV_FILE = "results/qnn_fit_benchmark_results.csv"

def set_seed(seed):
    """Set the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Set seed at the beginning
set_seed(SEED)


# --- Main Benchmark Logic ---
results = []

# Check for CUDA availability once
cuda_available = torch.cuda.is_available()
if not cuda_available:
    print("[WARNING] CUDA is not available. GPU tests will be skipped.")

print(f"\n[INFO] Starting QNN .fit() performance benchmark for {NUM_FIT_EPOCHS} epochs...")
print("-" * 50)


# Create a progress bar for the outer loop
for use_gpu in tqdm(GPU_OPTIONS, desc="Device (GPU/CPU)"):
    if use_gpu and not cuda_available:
        continue

    for n_qubits in tqdm(QUBIT_COUNTS, desc=f"Qubits (GPU={use_gpu})", leave=False):
        # Load data once per qubit count configuration
        print(f"\n[INFO] Loading data for {n_qubits} qubits...")
        _, quantum_data, input_dim, output_dim, is_multiclass = (
            data_pipeline(
                187,
                batch_size=BATCH_SIZE,
                do_pca=True,
                use_gpu=use_gpu,
                n_components=n_qubits,
                seed=SEED,
            )
        )
        (
            x_train_q,
            x_val_q,
            x_test_q,
            y_train_q,
            y_val_q,
            y_test_q,
            train_loader_q,
            val_loader_q,
            test_loader_q,
        ) = quantum_data

        for gradient_method in GRADIENT_METHODS:
            print(f"\n[TESTING] Qubits: {n_qubits}, Gradient: {gradient_method}, Use GPU: {use_gpu}")

            try:
                # 1. Instantiate the full model from your custom class and an optimizer
                model = QuantumNN(
                    ansatz=HEA(n_qubits=n_qubits, depth=2).get_ansatz(),
                    n_qubits=n_qubits,
                    num_classes=output_dim,
                    gradient_method=gradient_method,
                    use_gpu=use_gpu,
                    seed=SEED,
                    default_shots=1024,
                )
                optimizer = optim.Adam(model.parameters(), lr=0.01)

                # 2. Time the entire fit() call for a few epochs
                print(f"  Running .fit() for {NUM_FIT_EPOCHS} epochs...")
                t0 = time.time()
                model.fit(
                    train_loader=train_loader_q,
                    val_loader=val_loader_q,
                    epochs=NUM_FIT_EPOCHS,
                    optimizer=optimizer,
                    verbose=False,  # Disable inner tqdm for cleaner benchmark logs
                    eval_every=1    # Evaluate every epoch to include its cost
                )
                t1 = time.time()
                total_fit_time = t1 - t0
                time_per_epoch = total_fit_time / NUM_FIT_EPOCHS

                print(f"  Total Time for {NUM_FIT_EPOCHS} epochs: {total_fit_time:.4f}s ({time_per_epoch:.4f}s/epoch)")

                # 3. Store the aggregated results
                results.append({
                    "use_gpu": use_gpu,
                    "n_qubits": n_qubits,
                    "gradient_method": gradient_method,
                    "num_epochs": NUM_FIT_EPOCHS,
                    "total_fit_time_s": total_fit_time,
                    "time_per_epoch_s": time_per_epoch,
                    "status": "Success"
                })

            except Exception as e:
                print(f"  [ERROR] Test failed for this configuration: {e}")
                results.append({
                    "use_gpu": use_gpu,
                    "n_qubits": n_qubits,
                    "gradient_method": gradient_method,
                    "num_epochs": NUM_FIT_EPOCHS,
                    "total_fit_time_s": np.nan,
                    "time_per_epoch_s": np.nan,
                    "status": f"Failed: {e}"
                })

# --- Save Results to CSV ---
print("\n" + "="*50)
print("Benchmark complete. Saving results...")

# Ensure the results directory exists
output_dir = os.path.dirname(OUTPUT_CSV_FILE)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

results_df = pd.DataFrame(results)
results_df.to_csv(OUTPUT_CSV_FILE, index=False)

print(f"\nResults successfully saved to '{OUTPUT_CSV_FILE}'")
print("\nFinal Results Summary:")
print(results_df)
print("="*50)