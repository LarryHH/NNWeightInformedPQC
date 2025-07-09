import torch
import torch.nn as nn
import numpy as np

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
from utils.optimizers.GuidedSPSA import GuidedSPSASamplerGradient
from qiskit.transpiler import PassManager
from qiskit import transpile
from qiskit_machine_learning.utils import algorithm_globals

try:
    from .NN import NN
except ImportError:
    from NN import NN


class QuantumNN(NN):  # Renamed from SamplerQNNTorchModel
    """
    A Quantum Neural Network model using Qiskit's SamplerQNN and TorchConnector,
    refactored from SamplerQNNTorchModel.
    """

    def __init__(
        self,
        ansatz,
        n_qubits=2,
        num_classes=2,
        initial_point=None,
        seed=None,
        use_gpu: bool = False,
        default_shots: int = 1024,
        gradient_method: str = "param_shift",  # Default gradient method
        spsa_epsilon: float = 0.05,  # Default epsilon for SPSA
    ):
        """
        Initializes the QuantumNN model.

        Args:
            ansatz (qiskit.QuantumCircuit): The parameterized quantum circuit (ansatz).
            n_qubits (int, optional): Number of qubits. Defaults to 2.
            num_classes (int, optional): Number of output classes. Defaults to 2.
            initial_point (np.ndarray, optional): Initial values for the ansatz parameters.
                                                 If None, random values are used. Defaults to None.
        """
        super().__init__(num_classes=num_classes, use_gpu=use_gpu)
        self.n_qubits = n_qubits
        self.ansatz = ansatz
        self.initial_point = initial_point
        self.model = None
        self.criterion = nn.NLLLoss()
        self.seed = seed if seed is not None else algorithm_globals.random_seed

        feature_map = ZZFeatureMap(feature_dimension=n_qubits, reps=2)

        qc = QuantumCircuit(n_qubits)
        qc.compose(feature_map, inplace=True)
        qc.compose(self.ansatz, inplace=True)

        def interpret(x):
            return x % self.num_classes

        # Select the sampler and gradient method based on the device
        if use_gpu:
            self.sampler = self.select_sampler(sampler_device="gpu", default_shots=default_shots)
        else:
            self.sampler = self.select_sampler(sampler_device="cpu", default_shots=default_shots)
        
        self.inspect_sampler(self.sampler)
        self.gradient = self.select_gradient(gradient_method, spsa_epsilon)
        
        qnn = SamplerQNN(
            circuit=qc,
            input_params=feature_map.parameters,  # Parameters of the feature map
            weight_params=self.ansatz.parameters,  # Parameters of the ansatz (trainable weights)
            interpret=interpret,  # Maps measurement outcomes to class indices
            output_shape=self.num_classes,  # QNN outputs a probability vector of this size
            sampler=self.sampler,
            gradient=self.gradient,
            input_gradients=False,  # Default, but can be explicit
            pass_manager=PassManager(),  # Added as in original code
        )

        num_ansatz_params = 0
        if self.ansatz.parameters:  # Check if ansatz has parameters
            num_ansatz_params = len(self.ansatz.parameters)

        current_initial_point = None
        if self.initial_point is None:
            current_initial_point = 0.1 * np.random.randn(num_ansatz_params)
        else:
            current_initial_point = np.asarray(self.initial_point)

        self.model = TorchConnector(
            qnn,
            initial_weights=current_initial_point if num_ansatz_params > 0 else None,
        )
        self.model.to(self.device)

    def select_sampler(self, sampler_device: str = "cpu", default_shots=1024):
        """
        Selects the sampler for the QuantumNN model.
        Currently supports 'gpu' for AerSampler and 'cpu' for Qiskit Sampler.
        """
        sampler = None
        if sampler_device == "gpu":
            if torch.cuda.is_available():
                from qiskit_aer.primitives import SamplerV2 as AerSampler
                sampler = AerSampler(
                    default_shots=default_shots, 
                    seed=self.seed,
                    options={
                        "backend_options": {"method": "automatic", "device": "GPU", "batched_shots_gpu": True}
                    },
                )
            else:
                print("Warning: CUDA not available. Falling back to CPU Sampler.")
                from qiskit.primitives import Sampler
                sampler = Sampler(options={"shots": default_shots, "seed": self.seed})
        elif sampler_device == "cpu":
            from qiskit.primitives import Sampler
            sampler = Sampler(options={"shots": default_shots, "seed": self.seed})
        else:
            raise ValueError(f"Unsupported sampler device: {sampler_device}. Use 'gpu' or 'cpu'.")
        return sampler
        
    @staticmethod
    def inspect_sampler(sampler):
        """Print run-time & backend information for any Qiskit Sampler."""
        print("== Sampler info ==")
        print("class:", sampler.__module__, sampler.__class__.__name__)
        print("run options:", sampler.options)             # always exists

        # Aer ≥0.17  →  sampler.backend
        # Aer ≤0.16  →  sampler._backend   (private, but fall back if needed)
        backend = getattr(sampler, "backend", None) or getattr(sampler, "_backend", None)
        if backend is None:
            print("backend: reference CPU sampler (no backend object).")
            return

        print("backend:", backend.__class__.__name__)
        # backend.options is public in current Aer releases
        if hasattr(backend, "options"):
            print("backend options:", backend.options)

        # optional GPU/CPU device list (may not exist on CPU backends)
        if hasattr(backend, "available_devices"):
            try:
                print("available devices:", backend.available_devices())
            except Exception as err:
                print("available_devices():", err)


    def select_gradient(self, gradient_method: str = "param_shift", spsa_epsilon: float = 0.05):
        """
        Selects the gradient method for the QuantumNN model.
        Currently supports 'param_shift' and 'spsa'.
        """
        gradient = None
        if gradient_method == "param_shift":
            gradient = ParamShiftSamplerGradient(self.sampler)
        elif gradient_method == "spsa":
            gradient = SPSASamplerGradient(self.sampler, epsilon=spsa_epsilon)
        elif gradient_method == "guided_spsa":
            gradient = GuidedSPSASamplerGradient(
                self.sampler,
                N_epochs=100,  # Total number of epochs
                tau=0.5,      # Fraction of batch for param-shift
                epsilon=0.8,  # SPSA damping constant
                k_min_ratio=0.10,  # k_min = θ_len × k_min_ratio
                k_max_factor=1.5,  # k_max = θ_len × min(1, k_max_factor-tau)
                seed=self.seed,
            )
        else:
            raise ValueError(f"Unsupported gradient type: {gradient_method}. Use 'param_shift', 'spsa', or 'guided_spsa'.")
        return gradient

    def forward(self, x):
        """
        Forward pass for the QuantumNN.

        Args:
            x (torch.Tensor): Input tensor. Expected to be FloatTensor.

        Returns:
            torch.Tensor: Output log-probabilities.
        """
        # Ensure input is float, as TorchConnector and quantum circuits expect float features
        # self.model is the TorchConnector instance
        p = self.model(x.float())
        # Add a small epsilon for numerical stability before log for NLLLoss, as in original
        log_probs = torch.log(p + 1e-12)
        return log_probs

    def _prepare_targets_for_loss(self, yb):
        """
        Prepares target labels for NLLLoss.
        Ensures targets are Long type and have shape (N).
        """
        return yb.long().view(-1)

    def _prepare_targets_for_comparison(self, yb):
        """
        Prepares target labels for comparison.
        Ensures targets are Long type and have shape (N).
        """
        return yb.long().view(-1)

    def _train_batch(self, xb, yb, optimizer):
        """
        Performs a single training step for the QuantumNN.
        """
        optimizer.zero_grad()
        log_probs = self(xb)  # self.forward(xb) which returns log_probabilities
        yb_processed = self._prepare_targets_for_loss(yb)
        loss = self.criterion(log_probs, yb_processed)
        loss.backward()
        optimizer.step()
        return loss

    def _evaluate_batch_loss_and_logits(self, xb, yb_original):
        """
        Evaluates a batch and returns log-probabilities (as logits for NLLLoss) and loss.
        """
        log_probs = self(xb)  # self.forward(xb) returns log_probabilities
        yb_processed = self._prepare_targets_for_loss(yb_original)
        loss = self.criterion(log_probs, yb_processed)
        # For NLLLoss, the input (log_probs) is effectively the "logits"
        return log_probs, loss

    def end_epoch(self):
        """
        Advance the guided-SPSA scheduler once per epoch.
        Safe to call even if another gradient is selected.
        """
        if isinstance(self.gradient, GuidedSPSASamplerGradient):
            self.gradient.step_epoch()