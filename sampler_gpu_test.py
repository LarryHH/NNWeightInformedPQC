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

SEED = 0
N_COMPONENTS = 6
N_EPOCHS = 50

USE_GPU = False

# OPENML_DATASET_IDS = {
#     "iris": (61, 4),
#     "wine": (187, 13),
#     "diabetes": (37, 8),
# }


def set_seed(seed):
    """Set the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


print("\n[INFO] Loading data...")
classical_data, quantum_data, input_dim, output_dim, is_multiclass = (
    data_pipeline(
        187,
        batch_size=32,
        do_pca=True,
        use_gpu=USE_GPU,
        n_components=N_COMPONENTS,
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

hea_ansatz = HEA(n_qubits=input_dim, depth=2)
hea_ansatz.draw()


# 1. Instantiate your old QuantumNN class
model_hea = QuantumNN(
    ansatz=hea_ansatz.get_ansatz(),
    n_qubits=input_dim,
    num_classes=output_dim,
    seed=SEED,
    default_shots=1024,
    gradient_method="guided_spsa" # or "guided_spsa"
)

print("\n" + "="*50)
print("RUNNING DIRECT QNN PERFORMANCE TEST (for old SamplerQNN)...")

# 2. Extract the raw Qiskit QNN object from the Torch wrapper
# The path is simpler here because .model is a direct attribute
qnn = model_hea.model._neural_network

# 3. Create sample data for one batch
batch_size = 32
sample_features = np.random.rand(batch_size, qnn.num_inputs)
sample_weights = np.random.rand(qnn.num_weights)

# 4. Time the raw forward pass
t0 = time.time()
forward_output = qnn.forward(sample_features, sample_weights)
t1 = time.time()
print(f"Raw Forward Pass ({batch_size} samples): {t1 - t0:.4f} seconds")

# 5. Time the raw backward pass (gradient calculation)
t0 = time.time()
input_grads, weight_grads = qnn.backward(sample_features, sample_weights)
t1 = time.time()
print(f"Raw Backward Pass ({batch_size} samples): {t1 - t0:.4f} seconds")
print("="*50 + "\n")
