import torch
import torch.optim as optim
from qiskit import QuantumCircuit
from tqdm import tqdm
import copy
import itertools
import torch.nn.functional as F

from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import accuracy_score, confusion_matrix

from utils.nn.QuantumNN import QuantumNN

class DARTSTrainer:
    """
    Manages the bi-level optimization loop for the QuantumDARTS search.
    This version uses a more stable decoupled update strategy and includes gradient monitoring.
    """

    def __init__(self, model, train_loader, validation_data, lr_arch=0.01, lr_rot=0.05):
        self.model = model
        self.train_loader = train_loader
        self.validation_data = validation_data

        self.optimizer_arch = optim.Adam([self.model.P, self.model.Q], lr=lr_arch)
        self.optimizer_rot = optim.Adam([self.model.rotation_params], lr=lr_rot)

        self._baseline = None

        self.search_arch_loss_history = []
        self.search_rot_loss_history = []
        self.derived_circuit_performance_history = []

    def _log_gradient_stats(self, pbar, loss_value):
        """Helper to compute and display gradient norms."""
        rot_grad_norm = arch_grad_norm = 0.0
        if self.model.rotation_params.grad is not None:
            rot_grad_norm = self.model.rotation_params.grad.norm().item()
        
        p_grad_norm = q_grad_norm = 0.0
        if self.model.P.grad is not None: p_grad_norm = self.model.P.grad.norm().item()
        if self.model.Q.grad is not None: q_grad_norm = self.model.Q.grad.norm().item()
        arch_grad_norm = p_grad_norm + q_grad_norm
        
        pbar.set_postfix(
            loss=f"{loss_value:.4f}",
            rot_grad=f"{rot_grad_norm:.2e}", 
            arch_grad=f"{arch_grad_norm:.2e}"
        )

    def search(self, num_epochs, perform_eval=False, eval_every=1):
        """
        Runs the bi-level optimization search using a more stable, decoupled update strategy.
        """

        scheduler_arch = CosineAnnealingLR(self.optimizer_arch, T_max=num_epochs)
        scheduler_rot = CosineAnnealingLR(self.optimizer_rot, T_max=num_epochs)

        pbar = tqdm(range(num_epochs), desc="Searching")
        for epoch in pbar:
            # Create a single iterator for the training data
            train_iterator = iter(self.train_loader)
            
            # We iterate through half the dataset length because we consume two batches per step
            for i in range(len(self.train_loader) // 2):
                # --- Step 1: Update Rotation Parameters on one batch ---
                try:
                    rot_data, rot_labels = next(train_iterator)
                except StopIteration:
                    break
                
                self.optimizer_rot.zero_grad()
                loss_rot = self.model.compute_loss(rot_data, rot_labels)
                loss_rot.backward()
                self.optimizer_rot.step()

                # --- Step 2: Update Architecture Parameters on a *different* batch ---

                try:
                    arch_data, arch_labels = next(train_iterator)
                except StopIteration:
                    break

                # 1) Get the scalar loss for this batch (used only as a coefficient)
                loss_arch = self.model.compute_loss(arch_data, arch_labels)

                # 2) Grab the logits and the one-hot sample used in that forward
                logits = self.model.architecture_logits                    # [L, Q, |pool|]
                logp   = torch.log_softmax(logits, dim=-1)

                sample_one_hot = getattr(self.model, "fixed_one_hot", None)
                if sample_one_hot is None:
                    sample_one_hot = getattr(self.model, "last_sample_one_hot", None)
                    if sample_one_hot is None:
                        # force a forward pass to create a fresh sample
                        _ = self.model(arch_data)
                        sample_one_hot = self.model.last_sample_one_hot

                # 3) Log-prob of the sampled choices
                log_prob_sample = (sample_one_hot * logp).sum()            # scalar

                # 4) Moving baseline (init from first loss to avoid bias/NaNs)
                if self._baseline is None:
                    self._baseline = float(loss_arch.item())
                else:
                    self._baseline = 0.9 * self._baseline + 0.1 * float(loss_arch.item())

                # 5) REINFORCE objective (minimize expected loss):  (L - b) * log π(a)
                policy_loss = (loss_arch.detach() - self._baseline) * log_prob_sample

                self.optimizer_arch.zero_grad()
                policy_loss.backward()
                # optional: clip exploding grads
                # torch.nn.utils.clip_grad_norm_([self.model.P, self.model.Q], max_norm=1.0)
                self.optimizer_arch.step()
        
            scheduler_rot.step()
            scheduler_arch.step()
            
            # Save the last architecture loss of the epoch
            if 'loss_arch' in locals():
                self.search_arch_loss_history.append(loss_arch.item())
            if 'loss_rot' in locals():
                self.search_rot_loss_history.append(loss_rot.item())
            
            if perform_eval and (epoch + 1) % eval_every == 0:
                performance = self.evaluate_derived_circuit()
                self.derived_circuit_performance_history.append({"epoch": epoch + 1, "performance": performance})
                pbar.set_postfix(loss=f"{loss_arch.item():.4f}", acc=f"{performance:.4f}")
            elif 'loss_arch' in locals():
                 pbar.set_postfix(loss=f"{loss_arch.item():.4f}")

    def derive_best_circuit(self):
        gate_indices = self.model.architecture_logits.detach().cpu().argmax(dim=-1)
        n_qubits, n_layers = self.model.n_qubits, self.model.n_layers
        qc = QuantumCircuit(n_qubits)
        param_counter = 0
        for l in range(n_layers):
            for q in range(n_qubits):
                gate_info = self.model.gate_pool[gate_indices[l, q].item()]
                if gate_info["gate"] is None: continue
                if gate_info["n_qubits"] == 1:
                    if gate_info["n_params"] > 0:
                        qc.append(gate_info["gate"](0), [q])
                        param_counter += 1
                    else:
                        qc.append(gate_info["gate"](), [q])
                elif gate_info["n_qubits"] == 2:
                    qc.append(gate_info["gate"](), [q, (q + 1) % n_qubits])
        if param_counter > 0:
            from qiskit.circuit import ParameterVector
            params = ParameterVector("θ", length=param_counter)
            final_qc = QuantumCircuit(n_qubits)
            param_idx = 0
            for inst in qc.data:
                op, qubits = inst.operation, inst.qubits
                if len(op.params) > 0:
                    final_qc.append(op.__class__(params[param_idx]), qubits)
                    param_idx += 1
                else:
                    final_qc.append(op, qubits)
            return final_qc
        return qc

    def init_quantum_nn(self, circuit, input_dim, output_dim, use_gpu=False, quantum_lr=0.01, epochs_quantum=20):
        model_q = QuantumNN(
            ansatz=circuit,
            n_qubits=input_dim,
            num_classes=output_dim,
            use_gpu=use_gpu, 
            gradient_method="guided_spsa"
        )

        # Optimizer and Scheduler for Quantum Model
        optimizer_q = optim.Adam(model_q.parameters(), lr=quantum_lr)
        scheduler_q = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_q,
            mode="min",
            factor=0.3,
            patience=max(1, epochs_quantum // 5),
            min_lr=1e-4,
            verbose=False,
        )
        return model_q, optimizer_q, scheduler_q
    

    def final_evaluation(self, train_dataset, val_dataset, test_dataset, retrain_epochs=10, retrain_lr=0.05):
        """
        Retrains the final derived circuit and evaluates its performance.
        """
        print("\n--- Starting Final Evaluation ---")

        circuit = self.derive_best_circuit()
        model, optimizer, scheduler = self.init_quantum_nn(
            circuit=circuit,
            input_dim=self.model.n_qubits,
            output_dim=self.model.n_classes,
            use_gpu=torch.cuda.is_available(),
            quantum_lr=retrain_lr,
            epochs_quantum=retrain_epochs
        )
        model.fit(
            train_dataset,
            val_dataset,
            epochs=retrain_epochs,
            optimizer=optimizer,
            scheduler=scheduler,
            verbose=True,
            eval_every=2
        )
        eval_output = model.evaluate(test_dataset, verbose=False)
        test_metrics = {
            "test_acc": eval_output[1],
            "test_loss": eval_output[0],
            "test_prec": eval_output[2],
            "test_rec": eval_output[3],
            "test_f1": eval_output[4],
            "y_pred": eval_output[5],
            "y_true": eval_output[6]

        }
        print(f"Test Results: Loss={test_metrics['test_loss']:.4f}, Acc={test_metrics['test_acc']:.4f}, F1={test_metrics['test_f1']:.4f}")
        cm = confusion_matrix(test_metrics['y_true'], test_metrics['y_pred'])
        return {"model": model, "accuracy": test_metrics['test_acc'], "confusion_matrix": cm, "predictions": test_metrics['y_pred']}
    

    def evaluate_derived_circuit(self, training_steps=50, lr=0.1):
        """
        Fix the current architecture, fine-tune only rotation params for a few steps,
        compute validation accuracy, then restore the original model state.
        """
        # --- snapshot state we’ll mutate ---
        mode_train = self.model.training
        with torch.no_grad():
            rot_snapshot = self.model.rotation_params.detach().clone()

            # P and Q are the LEAF params for architecture; snapshot + flags
            P_req_prev = self.model.P.requires_grad
            Q_req_prev = self.model.Q.requires_grad
            P_snapshot = self.model.P.detach().clone()
            Q_snapshot = self.model.Q.detach().clone()

            had_fixed = hasattr(self.model, "fixed_one_hot")
            fixed_snapshot = self.model.fixed_one_hot.clone() if had_fixed else None

        # --- freeze architecture (use argmax one-hot) ---
        self.model.fix_architecture_()            # forward should now use fixed_one_hot
        self.model.P.requires_grad_(False)        # freeze arch params (LEAVES)
        self.model.Q.requires_grad_(False)

        # --- optimize rotations only ---
        optimizer = optim.Adam([self.model.rotation_params], lr=lr)
        train_iterator = itertools.cycle(self.train_loader)
        for _ in range(training_steps):
            xb, yb = next(train_iterator)
            optimizer.zero_grad()
            loss = self.model.compute_loss(xb, yb)  # should only depend on rotation_params now
            loss.backward()
            optimizer.step()

        # --- validation accuracy ---
        X_val, y_val = self.validation_data.tensors
        with torch.no_grad():
            out = self.model(X_val)
            pred = torch.argmax(out, dim=1)
            acc = (pred == y_val).float().mean().item()

        # --- restore original state ---
        with torch.no_grad():
            self.model.rotation_params.copy_(rot_snapshot)

            self.model.P.copy_(P_snapshot)
            self.model.Q.copy_(Q_snapshot)
            self.model.P.requires_grad_(P_req_prev)
            self.model.Q.requires_grad_(Q_req_prev)

            if had_fixed:
                self.model.fixed_one_hot = fixed_snapshot
            elif hasattr(self.model, "fixed_one_hot"):
                delattr(self.model, "fixed_one_hot")

        self.model.train(mode_train)
        return acc


    # def derive_best_circuit(self):
    #     final_weights = self.model.architecture_logits.detach().cpu()
    #     gate_indices = torch.argmax(final_weights, dim=-1)
    #     n_qubits = self.model.n_qubits
    #     n_layers = self.model.n_layers
    #     qc = QuantumCircuit(n_qubits)
    #     param_counter = 0
    #     for l in range(n_layers):
    #         for q in range(n_qubits):
    #             gate_idx = gate_indices[l, q].item()
    #             gate_info = self.model.gate_pool[gate_idx]
    #             if gate_info["gate"] is None: continue
    #             if gate_info["n_qubits"] == 1:
    #                 if len(gate_info["gate"](0).params) > 0:
    #                     qc.append(gate_info["gate"](0), [q])
    #                     param_counter += 1
    #                 else:
    #                     qc.append(gate_info["gate"](), [q])
    #             elif gate_info["n_qubits"] == 2:
    #                 qc.append(gate_info["gate"](), [q, (q + 1) % n_qubits])
    #         if n_layers > 1: qc.barrier()
    #     if param_counter > 0:
    #         from qiskit.circuit import ParameterVector
    #         final_params = ParameterVector("θ", length=param_counter)
    #         param_idx = 0
    #         final_qc = QuantumCircuit(n_qubits)
    #         for inst in qc.data:
    #             op, qubits = inst.operation, inst.qubits
    #             if len(op.params) > 0:
    #                 final_qc.append(op.__class__(final_params[param_idx]), qubits)
    #                 param_idx += 1
    #             else:
    #                 final_qc.append(op, qubits)
    #         return final_qc
    #     return qc

