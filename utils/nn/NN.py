import torch
import torch.nn as nn

from tqdm.auto import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import numpy as np
import torch.profiler as profiler

class NN(nn.Module):
    """
    Base Neural Network class providing common training, evaluation, and history plotting functionalities.
    """
    def __init__(self, num_classes, use_gpu=False):
        """
        Initializes the base NN class.

        Args:
            num_classes (int): The number of output classes.
        """
        super().__init__()
        self.num_classes = num_classes
        self.is_multiclass = num_classes > 2
        self.criterion = None # Must be set by subclasses
        self.history = {
            "epochs": [],
            "train_loss": [], "val_loss": [],
            "train_acc": [], "val_acc": [],
            "train_prec": [], "val_prec": [],
            "train_rec": [], "val_rec": [],
            "train_f1": [], "val_f1": [],
        }
        if use_gpu:
            if torch.cuda.is_available():
                print("Using GPU for training.")
            else:
                print("GPU not available, falling back to CPU.")
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cpu")
        self.to(self.device)
        print(f"Initialized {self.__class__.__name__} on device: {self.device}")
    
    def set_history_to_zero_for_loaded_model(self):
        self.history = {
            "train_loss": [0.0], "val_loss": [0.0],
            "train_acc": [0.0], "val_acc": [0.0],
            "train_prec": [0.0], "val_prec": [0.0],
            "train_rec": [0.0], "val_rec": [0.0],
            "train_f1": [0.0], "val_f1": [0.0],
        }

    def forward(self, x):
        """
        Forward pass of the neural network.
        This method must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement the forward method.")

    def _train_batch(self, xb, yb, optimizer):
        """
        Performs a single training step on a batch of data.
        This method must be implemented by subclasses.

        Args:
            xb (torch.Tensor): Input features for the batch.
            yb (torch.Tensor): True labels for the batch.
            optimizer (torch.optim.Optimizer): The optimizer to use for training.

        Returns:
            torch.Tensor: The loss value for the batch.
        """
        # Subclass implementation should:
        # 1. Zero gradients: optimizer.zero_grad()
        # 2. Get predictions: pred = self(xb) (or logits)
        # 3. Prepare targets: yb_processed = self._prepare_targets_for_loss(yb)
        # 4. Calculate loss: loss = self.criterion(pred, yb_processed)
        # 5. Backpropagate: loss.backward()
        # 6. Update weights: optimizer.step()
        # 7. Return loss
        raise NotImplementedError("Subclasses must implement _train_batch.")

    def _evaluate_batch_loss_and_logits(self, xb, yb_original):
        """
        Evaluates a batch of data and returns logits and loss.
        This method must be implemented by subclasses.

        Args:
            xb (torch.Tensor): Input features for the batch.
            yb_original (torch.Tensor): Original true labels for the batch.

        Returns:
            tuple: (logits (torch.Tensor), loss (torch.Tensor))
        """
        # Subclass implementation should:
        # 1. Get logits: logits = self(xb)
        # 2. Prepare targets: yb_processed = self._prepare_targets_for_loss(yb_original)
        # 3. Calculate loss: loss = self.criterion(logits, yb_processed)
        # 4. Return (logits, loss)
        raise NotImplementedError("Subclasses must implement _evaluate_batch_loss_and_logits.")

    def _prepare_targets_for_loss(self, yb):
        """
        Prepares target labels for loss calculation.
        Default implementation converts to Long type. Subclasses can override.

        Args:
            yb (torch.Tensor): Original target labels.

        Returns:
            torch.Tensor: Processed target labels.
        """
        return yb.long()

    def _prepare_targets_for_comparison(self, yb):
        """
        Prepares target labels for comparison with predictions (e.g., for accuracy calculation).
        Default implementation converts to Long type. Subclasses can override.
        Ensures target is 1D.

        Args:
            yb (torch.Tensor): Original target labels.

        Returns:
            torch.Tensor: Processed target labels.
        """
        return yb.long().view(-1) # Ensure 1D for comparison with argmax output

    def _to_device(self, *tensors):
        """Moves all input tensors to the model's device."""
        return [t.to(self.device) for t in tensors]
    
    def end_epoch(self):          
        """Called once after every training epoch."""
        pass
        
    def fit(self, train_loader, val_loader, epochs, optimizer, scheduler=None, verbose=True, use_profiler=False, eval_every=1):
            """
            Trains the model for a specified number of epochs.
            If verbose is True, uses tqdm for progress display.
            If verbose is False, training is silent.
            Args:
                use_profiler (bool): If True, enables PyTorch profiler during training.
                eval_every (int): Controls the frequency of evaluation. E.g., eval_every=5 runs evaluation every 5 epochs.
            """
    
            profiler_schedule = torch.profiler.schedule(wait=1, warmup=1, active=2, repeat=1)
            profiler_context = None

            if use_profiler:
                activities = [
                    torch.profiler.ProfilerActivity.CPU,
                ]
                if self.device.type == 'cuda':
                    activities.append(torch.profiler.ProfilerActivity.CUDA)

                print("--- PyTorch Profiler Enabled ---")
                profiler_context = profiler.profile(
                    schedule=profiler_schedule,
                    activities=activities,
                    on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/qnn_profile'),
                    with_stack=True,
                    profile_memory=True
                )
                profiler_context.__enter__() # Manually enter the context manager


            # --- Training Loop ---
            for epoch in range(epochs):
                self.train() # Set model to training mode
                
                # Setup tqdm loop if verbose, otherwise iterate directly
                if verbose:
                    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=True)
                else:
                    loop = train_loader 

                n_batches = len(train_loader)
                
                for i, (xb, yb) in enumerate(loop):
                    xb, yb = self._to_device(xb, yb)
                    loss = self._train_batch(xb, yb, optimizer) # Implemented by subclass
                    
                    postfix_data = {"batch_loss": f"{loss.item():.3f}"}

                    # If profiler is active, call prof.step() after each batch
                    if use_profiler and profiler_context is not None:
                        profiler_context.step()
                    
                    # --- MODIFIED BLOCK ---
                    # Check if it's the last batch AND if it's an evaluation epoch
                    if (epoch + 1) % eval_every == 0 and i == n_batches - 1:
                        # Evaluate silently for metrics to update history and postfix
                        train_loss_eval, train_acc_eval, train_prec_eval, train_rec_eval, train_f1_eval, *_ = self.evaluate(train_loader, verbose=False)
                        val_loss_eval,   val_acc_eval, val_prec_eval, val_rec_eval, val_f1_eval, *_ = self.evaluate(val_loader, verbose=False)


                        if scheduler is not None:
                            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                                scheduler.step(val_loss_eval)
                            else:
                                scheduler.step()

                        self.history["epochs"].append(epoch + 1)
                        self.history["train_loss"].append(train_loss_eval)
                        self.history["val_loss"].append(val_loss_eval)
                        self.history["train_acc"].append(train_acc_eval)
                        self.history["val_acc"].append(val_acc_eval)
                        self.history["train_prec"].append(train_prec_eval)
                        self.history["val_prec"].append(val_prec_eval)
                        self.history["train_rec"].append(train_rec_eval)
                        self.history["val_rec"].append(val_rec_eval)
                        self.history["train_f1"].append(train_f1_eval)
                        self.history["val_f1"].append(val_f1_eval)

                        # Update postfix with epoch summaries if verbose
                        if verbose:
                            postfix_data.update({
                                "train_loss": f"{train_loss_eval:.3f}",
                                "train_acc":  f"{train_acc_eval:.3f}",
                                "val_loss":   f"{val_loss_eval:.3f}",
                                "val_acc":    f"{val_acc_eval:.3f}",
                            })
                            if scheduler:
                                current_lr = optimizer.param_groups[0]['lr']
                                postfix_data["lr"] = f"{current_lr:.2e}"
                    
                    # Set postfix for tqdm loop if verbose
                    if verbose and isinstance(loop, tqdm):
                        loop.set_postfix(**postfix_data)
                self.end_epoch() 
            
            # --- Profiler Teardown ---
            if use_profiler and profiler_context is not None:
                profiler_context.__exit__(None, None, None) # Manually exit the context manager
                print("--- PyTorch Profiler Results ---")
                print(profiler_context.key_averages().table(sort_by="cuda_time_total" if self.device.type == 'cuda' else "cpu_time_total", row_limit=20))
                print("\nRun: tensorboard --logdir=./log to view detailed trace.")

    @torch.no_grad()
    def evaluate(self, loader, verbose=True):
        """
        Evaluates the model on a given dataset.

        Args:
            loader (torch.utils.data.DataLoader): DataLoader for the data to evaluate.
            verbose (bool, optional): Whether to show a progress bar. Defaults to True.


        Returns:
            tuple: (average_loss, accuracy, precision, recall, f1_score, y_pred_numpy, y_true_numpy)
        """
        self.eval() # Set model to evaluation mode
        total_loss_sum, correct_preds, total_samples = 0.0, 0, 0
        all_y_pred_tensors, all_y_true_tensors = [], []
        
        loop = tqdm(loader, desc="Evaluating", leave=False, disable=not verbose)
        for xb, yb_original in loop:
            xb, yb_original = self._to_device(xb, yb_original)
            logits, loss_value = self._evaluate_batch_loss_and_logits(xb, yb_original) # Implemented by subclass
            yb_for_comparison = self._prepare_targets_for_comparison(yb_original)

            total_loss_sum += loss_value.item() * xb.size(0)
            y_pred_batch = logits.argmax(dim=1)

            if y_pred_batch.shape != yb_for_comparison.shape:
                if yb_for_comparison.numel() == y_pred_batch.numel():
                    yb_for_comparison = yb_for_comparison.view_as(y_pred_batch)
                else:
                    # This should ideally be handled by _prepare_targets_for_comparison
                    # Or indicates a more fundamental issue in data prep / model output.
                    print(f"Warning: Shape mismatch in evaluation. y_pred: {y_pred_batch.shape}, y_true: {yb_for_comparison.shape}.")


            correct_preds += (y_pred_batch == yb_for_comparison).sum().item()
            total_samples += yb_original.size(0)

            all_y_pred_tensors.append(y_pred_batch.cpu())
            all_y_true_tensors.append(yb_for_comparison.cpu())

        if total_samples == 0:
            if verbose: print("Warning: Evaluation loader is empty.")
            empty_np_array = np.array([])
            return 0.0, 0.0, 0.0, 0.0, 0.0, empty_np_array, empty_np_array

        avg_epoch_loss = total_loss_sum / total_samples
        epoch_accuracy = correct_preds / total_samples

        y_pred_np = torch.cat(all_y_pred_tensors).numpy()
        y_true_np = torch.cat(all_y_true_tensors).numpy()

        avg_mode = 'macro' if self.is_multiclass else 'binary'

        precision = precision_score(y_true_np, y_pred_np, average=avg_mode, zero_division=0)
        recall    = recall_score(y_true_np, y_pred_np, average=avg_mode, zero_division=0)
        f1        = f1_score(y_true_np, y_pred_np, average=avg_mode, zero_division=0)

        return avg_epoch_loss, epoch_accuracy, precision, recall, f1, y_pred_np, y_true_np

    def plot_history(self):
        """
        Plots training and validation loss and/or accuracy curves.
        """
        epochs = self.history["epochs"] 
        
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, self.history["train_loss"], 'b-o', label="Train") # Added 'o' marker
        plt.plot(epochs, self.history["val_loss"], 'r-o', label="Validation") # Added 'o' marker
        plt.xlabel("Epoch")
        plt.ylabel("Loss (NLL)")
        plt.title("Loss Curves")
        plt.legend()
        plt.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.show()