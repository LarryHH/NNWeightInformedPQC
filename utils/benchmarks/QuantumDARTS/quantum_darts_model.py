import torch
import torch.nn as nn
import torch.nn.functional as F
from .differentiable_quantum_layer import DifferentiableQuantumCircuit

from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment
import math

class QuantumDARTSModel(nn.Module):
    def __init__(self, n_qubits, n_layers, gate_pool, n_classes, k_prime=4):
        super().__init__()
        self.n_qubits  = n_qubits
        self.n_layers  = n_layers
        self.gate_pool = gate_pool
        self.n_gates   = len(gate_pool)
        self.n_classes = n_classes
        self.tau       = 1.0

        self.P = nn.Parameter(torch.randn(self.n_layers, self.n_qubits, 1, k_prime))
        self.Q = nn.Parameter(torch.randn(self.n_layers, self.n_qubits, k_prime, self.n_gates))
        
        max_rot_params = self.n_layers * self.n_qubits
        self.rotation_params = nn.Parameter(torch.rand(max_rot_params) * 2 * torch.pi)

        self.last_sample_one_hot = None
        self.register_buffer("P_align", None, persistent=False)
        self.P_align = None

        M = 2 ** self.n_qubits
        counts = torch.tensor(
            [M // self.n_classes + (1 if c < (M % self.n_classes) else 0)
             for c in range(self.n_classes)],
            dtype=torch.float32
        )
        weights = 1.0 / counts
        self.register_buffer("class_weights", weights)
        self.loss_fn = nn.NLLLoss(weight=self.class_weights)  # was nn.NLLLoss()


    @property
    def architecture_logits(self):
        return (self.P @ self.Q).squeeze(2)  # [L, Q, |pool|]
    
    def architecture_negative_entropy(self):
        # logits: [L, Q, |G|]
        p = torch.softmax(self.architecture_logits, dim=-1).clamp_min(1e-12)
        H = -(p * p.log()).sum(dim=-1).mean()   # mean over layers & qubits
        return -H  # minimizing (-H) == maximizing entropy


    def set_perm(self, P: torch.Tensor):
        if P is None:
            self.P_align = None
        else:
            # ensure tensor, dtype/device match model params
            dev = next(self.parameters()).device
            dt  = next(self.parameters()).dtype
            self.P_align = P.detach().to(device=dev, dtype=dt)

    def forward(self, x):
        # fixed architecture (after discretization) or sampled one-hot via Gumbel-Softmax
        if hasattr(self, "fixed_one_hot"):
            gate_select_one_hot = self.fixed_one_hot
        else:
            gate_select_one_hot = F.gumbel_softmax(
                self.architecture_logits, tau=self.tau, hard=True, dim=-1
            )
        self.last_sample_one_hot = gate_select_one_hot.detach()

        # Differentiable quantum layer returns class probabilities (sum=1)
        probabilities = DifferentiableQuantumCircuit.apply(
            x,
            gate_select_one_hot,
            self.rotation_params,
            self.n_qubits,
            self.n_layers,
            self.gate_pool,
            self.n_classes,
        )
        return probabilities  # (N, C)

    def compute_loss(self, x, y):
        probs = self.forward(x)      # (N, C), mod-C head
        log_prob = (probs + 1e-12).log()
        return self.loss_fn(log_prob, y)


    def set_tau(self, new_tau):
        self.tau = new_tau

    def fix_architecture_(self):
        """
        Discretize the architecture by argmax and store a fixed one-hot buffer.
        """
        with torch.no_grad():
            argmax_indices = self.architecture_logits.argmax(dim=-1)  # [L, Q]
            one_hot = F.one_hot(argmax_indices, num_classes=self.n_gates).float()
        self.register_buffer("fixed_one_hot", one_hot)
