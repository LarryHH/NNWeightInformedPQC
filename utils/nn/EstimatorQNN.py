import torch
import torch.nn as nn
import numpy as np
import os

import qiskit
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit import transpile

# Imports for the EstimatorQNN Path
from qiskit_aer.primitives import Estimator as AerEstimator
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector
from qiskit_machine_learning.gradients import ParamShiftEstimatorGradient, SPSAEstimatorGradient
from qiskit_machine_learning.utils import algorithm_globals

# Your custom imports
from utils.ansatze.GenericAnsatz import twolocal_nontranspiled, zzfeaturemap_nontranspiled
try:
    # This is a placeholder for your base NN class structure
    from .NN import NN
except ImportError:
    class NN(nn.Module):
        def __init__(self, num_classes, use_gpu):
            super().__init__()
            self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        def _train_batch(self, *args, **kwargs): pass
        def _evaluate_batch_loss_and_logits(self, *args, **kwargs): pass


class QuantumNN(nn.Module):
    """
    Acts as a quantum feature extractor, outputting a SINGLE expectation value.
    """
    def __init__(
        self,
        ansatz,
        n_qubits: int,
        initial_point: np.ndarray = None,
        seed: int = None,
        gradient_method: str = "param_shift",
        spsa_epsilon: float = 0.05,
    ):
        super().__init__()
        
        feature_map = zzfeaturemap_nontranspiled(n_qubits, reps=2)
        qc = QuantumCircuit(n_qubits)
        qc.compose(feature_map, inplace=True)
        qc.compose(ansatz, inplace=True)

        backend = AerSimulator(method="statevector")
        transpiled_qc = transpile(qc, backend=backend, optimization_level=1)

        self.estimator = AerEstimator()
        if gradient_method == "param_shift":
            self.gradient = ParamShiftEstimatorGradient(self.estimator)
        else: # spsa
            self.gradient = SPSAEstimatorGradient(self.estimator, epsilon=spsa_epsilon)

        # --- KEY CHANGE 1: DEFINE ONLY ONE OBSERVABLE ---
        # The QNN will now only compute the expectation value of Z on the first qubit.
        # This reduces the quantum workload to the absolute minimum.
        self.observable = SparsePauliOp("Z" + "I"*(n_qubits-1))
        
        qnn = EstimatorQNN(
            circuit=transpiled_qc,
            estimator=self.estimator,
            input_params=feature_map.parameters,
            weight_params=ansatz.parameters,
            observables=self.observable, # Pass the single observable
            gradient=self.gradient,
        )

        if initial_point is None:
            initial_weights = 0.1 * np.random.randn(qnn.num_weights)
        else:
            initial_weights = np.asarray(initial_point)

        self.model = TorchConnector(qnn, initial_weights=initial_weights)

    def forward(self, x: torch.Tensor):
        return self.model(x)

class HybridModel(NN):
    """
    This is the main model class you will use in your training script.
    It combines the quantum feature extractor (QuantumNN) with a classical linear layer.
    """
    def __init__(self, n_qubits, num_classes, ansatz, use_gpu=False, **kwargs):
        super().__init__(num_classes=num_classes, use_gpu=use_gpu)
        
        self.quantum_layer = QuantumNN(n_qubits=n_qubits, ansatz=ansatz, **kwargs)
        
        self.classical_layer = nn.Linear(1, num_classes)
        
        self.criterion = nn.NLLLoss()
        self.to(self.device)

    def forward(self, x: torch.Tensor):
        x = x.to(self.device)
        
        # 1. Get the single expectation value from the quantum layer. Shape: [batch_size, 1]
        quantum_feature = self.quantum_layer(x)
        
        # 2. Pass this single feature to the classical layer to get final logits
        logits = self.classical_layer(quantum_feature)
        
        # 3. Convert to log-probabilities
        return torch.log_softmax(logits, dim=1)

    def _prepare_targets_for_loss(self, yb):
        return yb.to(self.device).long().view(-1)

    def _prepare_targets_for_comparison(self, yb):
        return yb.to(self.device).long().view(-1)

    def _train_batch(self, xb, yb, optimizer):
        optimizer.zero_grad()
        log_probs = self(xb)
        yb_processed = self._prepare_targets_for_loss(yb)
        loss = self.criterion(log_probs, yb_processed)
        loss.backward()
        optimizer.step()
        return loss

    def _evaluate_batch_loss_and_logits(self, xb, yb_original):
        log_probs = self(xb)
        yb_processed = self._prepare_targets_for_loss(yb_original)
        loss = self.criterion(log_probs, yb_processed)
        return log_probs, loss