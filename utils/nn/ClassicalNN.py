import torch
import torch.nn as nn
import torch.nn.functional as F

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
