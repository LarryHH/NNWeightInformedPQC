import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict

try:
    from .NN import NN
except ImportError:
    from NN import NN

class ClassicalNN(NN):
    """
    A classical neural network model.
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        """
        Initializes the ClassicalNN model.

        Args:
            input_dim (int): Dimension of the input features.
            hidden_dim (int): Dimension of the hidden layers.
            output_dim (int): Dimension of the output (number of classes).
        """
        super().__init__(num_classes=output_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim) # Output layer

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        """
        Forward pass for the classical neural network.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output logits.
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x) # Raw logits for CrossEntropyLoss
        return x
    
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
        Performs a single training step for the ClassicalNN.
        """
        optimizer.zero_grad()
        pred_logits = self(xb) # Get raw logits from forward pass
        yb_processed = self._prepare_targets_for_loss(yb)
        loss = self.criterion(pred_logits, yb_processed)
        loss.backward()
        optimizer.step()
        return loss

    def _evaluate_batch_loss_and_logits(self, xb, yb_original):
        """
        Evaluates a batch and returns logits and loss for ClassicalNN.
        """
        logits = self(xb) # Get raw logits
        yb_processed = self._prepare_targets_for_loss(yb_original)
        loss = self.criterion(logits, yb_processed)
        return logits, loss

    @staticmethod
    def get_weights_and_biases(model_instance):
        """
        Retrieves weights and biases from the linear layers of the model.

        Args:
            model_instance (ClassicalNN): An instance of the ClassicalNN model.

        Returns:
            dict: A dictionary containing weights and biases.
        """
        weights_and_biases = {}
        for name, module in model_instance.named_modules():
            if isinstance(module, torch.nn.Linear):
                weights_and_biases[f"{name}.weight"] = module.weight.detach().cpu().clone()
                if module.bias is not None:
                    weights_and_biases[f"{name}.bias"] = module.bias.detach().cpu().clone()
        return weights_and_biases


class FlexibleNN(NN):
    """
    Same coding style as ClassicalNN but with:
      • arbitrary hidden layer depths
      • optional condition-number regulariser
      • optional weight-to-angle scaling on export
    """
    _ACT = {"relu": nn.ReLU, "tanh": nn.Tanh, "gelu": nn.GELU}

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],   # e.g. [64,64,32]
        output_dim: int,
        *,
        act: str = "relu",
        condition_number: float = 0.0,      # 0 ⇒ no κ penalty
        scale_on_export: bool = False,
    ):
        super().__init__(num_classes=output_dim)
        self.condition_number = condition_number
        self.scale_on_export = scale_on_export
        layers, in_d = [], input_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_d, h), self._ACT[act]()]
            in_d = h
        layers.append(nn.Linear(in_d, output_dim))
        self.net = nn.Sequential(*layers)
        self.net.to(self.device) 
        self.criterion = nn.CrossEntropyLoss()
        self.criterion.to(self.device)

    def forward(self, x):
        return self.net(x)

    def _prepare_targets_for_loss(self, yb):         # unchanged helpers
        return yb.long().view(-1)

    def _train_batch(self, xb, yb, optimizer):
        optimizer.zero_grad()
        logits = self(xb)
        loss = self.criterion(logits, self._prepare_targets_for_loss(yb))
        if self.condition_number > 0:
            loss = loss + self.condition_number * self._cond_penalty()
        loss.backward()
        optimizer.step()
        return loss

    def _evaluate_batch_loss_and_logits(self, xb, yb):
        logits = self(xb)
        loss = self.criterion(logits, self._prepare_targets_for_loss(yb))
        return logits, loss

    @staticmethod
    def get_weights_and_biases(model_instance) -> Dict[str, torch.Tensor]:
        wb = {}
        for n, m in model_instance.named_modules():
            if isinstance(m, nn.Linear):
                wb[f"{n}.weight"] = m.weight.detach().cpu().clone()
                if m.bias is not None:
                    wb[f"{n}.bias"] = m.bias.detach().cpu().clone()
        return wb

    def export_weights(self, angle_bound=torch.pi) -> Dict[str, torch.Tensor]:
        wb = self.get_weights_and_biases(self)
        if not self.scale_on_export:
            return wb
        return {k: self._scale(v, angle_bound) for k, v in wb.items()}

    def _cond_penalty(self, eps=1e-6) -> torch.Tensor:
        p = 0.0
        for m in self.modules():
            if isinstance(m, nn.Linear):
                s = torch.linalg.svdvals(m.weight)
                p += s.max() / s.min().clamp(min=eps)
        return p

    @staticmethod
    def _scale(w: torch.Tensor, bound: float) -> torch.Tensor:
        s = bound / w.abs().max().clamp(min=1e-12)
        return w * s
    
    @staticmethod
    def get_model_from_state_dict(model_state_fp: str) -> "FlexibleNN":
        model_state = torch.load(model_state_fp, map_location="cpu")
        # ── extract linear-Layer indices and shapes ─────────────────────────────
        # Keys look like "net.<idx>.weight"  (idx = 0,2,4,…) inside nn.Sequential
        w_keys = [
            (int(m.group(1)), k)
            for k in model_state
            if (m := re.match(r"net\.(\d+)\.weight", k))
        ]
        w_keys.sort(key=lambda t: t[0])          # order by idx: 0,2,4,…
        weights = [model_state[k] for _, k in w_keys]
        input_dim   = weights[0].shape[1]        # in_features of first Linear
        hidden_dims = [w.shape[0] for w in weights[:-1]]   # all but last layer
        output_dim  = weights[-1].shape[0]       # out_features of last Linear
        # ── rebuild model ───────────────────────────────────────────────────────
        model = FlexibleNN(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            act="relu",              # or whatever was used
            condition_number=0.0,
            scale_on_export=False,
        )
        model.load_state_dict(model_state)
        model.eval()
        model.to(model.device)
        return model
