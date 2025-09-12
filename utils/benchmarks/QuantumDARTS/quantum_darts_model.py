import torch
import torch.nn as nn
import torch.nn.functional as F
from .differentiable_quantum_layer import DifferentiableQuantumCircuit

class QuantumDARTSModel(nn.Module):
    """
    PyTorch module for QuantumDARTS, now with a more stable loss function and
    cleaner architecture fixing.
    """
    def __init__(self, n_qubits, n_layers, gate_pool, n_classes, k_prime=4):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.gate_pool = gate_pool
        self.n_gates = len(gate_pool)
        self.n_classes = n_classes
        # MODIFIED: Use NLLLoss, which is the correct function for log-probabilities.
        self.loss_fn = nn.NLLLoss()
        self.tau = 1.0

        self.P = nn.Parameter(torch.randn(self.n_layers, self.n_qubits, 1, k_prime))
        self.Q = nn.Parameter(torch.randn(self.n_layers, self.n_qubits, k_prime, self.n_gates))
        
        max_rot_params = self.n_layers * self.n_qubits
        self.rotation_params = nn.Parameter(torch.rand(max_rot_params) * 2 * torch.pi)

        self.last_sample_one_hot = None  # To store the last sampled architecture

    @property
    def architecture_logits(self):
        return (self.P @ self.Q).squeeze(2)

    def forward(self, x):
        # The forward pass now handles the architecture sampling.
        # If the architecture has been fixed for evaluation, use the fixed one-hot tensor.
        if hasattr(self, 'fixed_one_hot'):
            gate_select_one_hot = self.fixed_one_hot
        else:
            # Otherwise, sample using Gumbel-Softmax.
            gate_select_one_hot = F.gumbel_softmax(self.architecture_logits, tau=self.tau, hard=True, dim=-1)
        self.last_sample_one_hot = gate_select_one_hot.detach()

        # The differentiable layer now only handles the quantum simulation and its gradients.
        probabilities = DifferentiableQuantumCircuit.apply(
            x,
            gate_select_one_hot, # Pass the one-hot tensor directly
            self.rotation_params,
            self.n_qubits,
            self.n_layers,
            self.gate_pool,
            self.n_classes
        )
        return probabilities
    
    def compute_loss(self, x, y):
        """Computes the Negative Log Likelihood loss."""
        probabilities = self.forward(x)
        # MODIFIED: Take the log of probabilities before passing to NLLLoss.
        # Add a small epsilon for numerical stability to prevent log(0).
        log_probabilities = (probabilities + 1e-12).log()
        return self.loss_fn(log_probabilities, y)

    def set_tau(self, new_tau):
        self.tau = new_tau

    def fix_architecture_(self):
        """
        Discretizes the architecture by taking the argmax and stores the
        resulting one-hot tensor as a fixed buffer. The forward pass will
        then use this fixed buffer instead of sampling.
        """
        with torch.no_grad():
            argmax_indices = self.architecture_logits.argmax(dim=-1)
            one_hot = F.one_hot(argmax_indices, num_classes=self.n_gates).float()
        # Register as a buffer so it's part of the model's state and moves to the correct device.
        self.register_buffer("fixed_one_hot", one_hot)

