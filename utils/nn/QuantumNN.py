import torch
import torch.nn as nn
import numpy as np

import qiskit
from qiskit import QuantumCircuit
from qiskit.circuit.library import TwoLocal, ZZFeatureMap
from qiskit.primitives import StatevectorSampler, Sampler
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.connectors import TorchConnector
from qiskit_machine_learning.gradients import ParamShiftSamplerGradient, SPSASamplerGradient
from qiskit.transpiler import PassManager
from qiskit import transpile
from qiskit_machine_learning.utils import algorithm_globals

try:
    from .NN import NN
except ImportError:
    from NN import NN

class QuantumNN(NN): # Renamed from SamplerQNNTorchModel
    """
    A Quantum Neural Network model using Qiskit's SamplerQNN and TorchConnector,
    refactored from SamplerQNNTorchModel.
    """
    def __init__(self, ansatz, n_qubits=2, num_classes=2, initial_point=None, seed=None):
        """
        Initializes the QuantumNN model.

        Args:
            ansatz (qiskit.QuantumCircuit): The parameterized quantum circuit (ansatz).
            n_qubits (int, optional): Number of qubits. Defaults to 2.
            num_classes (int, optional): Number of output classes. Defaults to 2.
            initial_point (np.ndarray, optional): Initial values for the ansatz parameters.
                                                 If None, random values are used. Defaults to None.
        """
        super().__init__(num_classes=num_classes)
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

        try:
            if torch.cuda.is_available():
                from qiskit_aer.primitives import SamplerV2 as AerSampler
                print("Qiskit Sampler (Aer backend) configured to use GPU.")
                sampler = AerSampler(
                    default_shots=1024,  # Default number of shots
                    seed=self.seed,
                    options={"backend_options": {"method": "statevector", "device": "GPU"}}
                )
                gradient = ParamShiftSamplerGradient(sampler) # for param shift rule (exact, slow)
                qc = transpile(qc, seed_transpiler=self.seed, optimization_level=3)
            else:
                raise ImportError # Force CPU fallback if CUDA not available (no GPU, no Aer-GPU)
        except (ImportError, qiskit.exceptions.QiskitError) as e:
            print(f"Warning: Could not set up Qiskit Aer Sampler for GPU ({e}). Falling back to CPU Sampler.")
            from qiskit.primitives import Sampler
            sampler = Sampler(options={"shots": 1024, "seed": self.seed})
            gradient = ParamShiftSamplerGradient(sampler) # for param shift rule (exact, slow)
        
            # sampler = StatevectorSampler() # for statevector simulation (exact, slow)
            # gradient = SPSASamplerGradient(sampler, epsilon=0.05) # stochastic, uses only 2 circuit evals (fast for many params), but needs careful tuning for epsilon and more epochs

        qnn = SamplerQNN(
            circuit=qc,
            input_params=feature_map.parameters,    # Parameters of the feature map
            weight_params=self.ansatz.parameters,   # Parameters of the ansatz (trainable weights)
            interpret=interpret,       # Maps measurement outcomes to class indices
            output_shape=self.num_classes,          # QNN outputs a probability vector of this size
            sampler=sampler,
            gradient=gradient,
            input_gradients=False, # Default, but can be explicit
            pass_manager=PassManager() # Added as in original code
        )

        num_ansatz_params = 0
        if self.ansatz.parameters: # Check if ansatz has parameters
            num_ansatz_params = len(self.ansatz.parameters)

        current_initial_point = None
        if self.initial_point is None:
            current_initial_point = 0.1 * np.random.randn(num_ansatz_params)
        else:
            current_initial_point = np.asarray(self.initial_point)


        self.model = TorchConnector(qnn, initial_weights=current_initial_point if num_ansatz_params > 0 else None)
        self.model.to(self.device)

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
        log_probs = self(xb) # self.forward(xb) which returns log_probabilities
        yb_processed = self._prepare_targets_for_loss(yb)
        loss = self.criterion(log_probs, yb_processed)
        loss.backward()
        optimizer.step()
        return loss

    def _evaluate_batch_loss_and_logits(self, xb, yb_original):
        """
        Evaluates a batch and returns log-probabilities (as logits for NLLLoss) and loss.
        """
        log_probs = self(xb) # self.forward(xb) returns log_probabilities
        yb_processed = self._prepare_targets_for_loss(yb_original)
        loss = self.criterion(log_probs, yb_processed)
        # For NLLLoss, the input (log_probs) is effectively the "logits"
        return log_probs, loss
