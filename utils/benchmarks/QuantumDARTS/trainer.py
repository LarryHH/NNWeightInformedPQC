import torch
import torch.optim as optim
from qiskit import QuantumCircuit
from tqdm import tqdm
import copy
import itertools
import torch.nn.functional as F
import math

from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import accuracy_score, confusion_matrix
from scipy.optimize import linear_sum_assignment

from utils.nn.QuantumNN import QuantumNN


def _learn_perm_from_probs(P_NC, y_N, C):
    preds = P_NC.argmax(dim=1).cpu().numpy()
    yv = y_N.cpu().numpy()
    cm = confusion_matrix(yv, preds, labels=list(range(C)))
    r, c = linear_sum_assignment(-cm)
    P = torch.zeros(C, C, dtype=P_NC.dtype, device=P_NC.device)
    for i, j in zip(r, c):
        P[i, j] = 1.0
    return P


def _calibrate_perm_on_loader(model, loader, max_batches=4):
    """
    Learn a CxC permutation P on a small slice of the train loader and cache it in model.P_align.
    Works whether the model’s head is probs-over-buckets or Z-expectation-softmax,
    because we learn P from the current model outputs.
    """
    dev = next(model.parameters()).device
    xs, ys = [], []
    it = iter(loader)
    for _ in range(max_batches):
        try:
            xb, yb = next(it)
        except StopIteration:
            break
        xs.append(xb.to(dev))
        ys.append(yb.to(dev))
    if not xs:  # empty loader
        return
    X = torch.cat(xs, dim=0)
    y = torch.cat(ys, dim=0)

    with torch.no_grad():
        out = model(X)  # (N,C) probabilities (QuantumDARTSModel.forward returns probs)
        P = _learn_perm_from_probs(out, y, model.n_classes)
    model.set_perm(P)


def _loader_to_tensors(loader, device):
    xs, ys = [], []
    for xb, yb in loader:
        xs.append(xb)
        ys.append(yb)
    X = torch.cat(xs, dim=0).to(device)
    y = torch.cat(ys, dim=0).to(device)
    return X, y


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
        if self.model.P.grad is not None:
            p_grad_norm = self.model.P.grad.norm().item()
        if self.model.Q.grad is not None:
            q_grad_norm = self.model.Q.grad.norm().item()
        arch_grad_norm = p_grad_norm + q_grad_norm

        pbar.set_postfix(
            loss=f"{loss_value:.4f}",
            rot_grad=f"{rot_grad_norm:.2e}",
            arch_grad=f"{arch_grad_norm:.2e}",
        )
            
    def search(self, num_epochs=20, perform_eval=True, eval_every=5):
        K = 10                  # as in the paper
        angle_l2 = 1e-4         # tiny L2 on θ  (L_theta)
        ent_beta = 5e-3         # tiny entropy on α (L_alpha)
        self.model.set_tau(0.05)  # fixed per paper

        pbar = tqdm(range(num_epochs), desc="Searching")
        train_iterator = itertools.cycle(self.train_loader)

        for epoch in pbar:
            # Optional: you can skip perm calibration during search with mod-C head
            # _calibrate_perm_on_loader(self.model, self.train_loader, max_batches=4)

            # ---- Sample ONE architecture and freeze it for K θ-steps ----
            freeze_batch_x, freeze_batch_y = next(train_iterator)
            _ = self.model(freeze_batch_x)                     # trigger gumbel sample
            with torch.no_grad():
                one_hot = self.model.last_sample_one_hot
                self.model.register_buffer("fixed_one_hot", one_hot)  # freeze sample

            # ---- K inner updates on θ (gate parameters) ----
            for _ in range(K):
                rot_x, rot_y = next(train_iterator)
                self.optimizer_rot.zero_grad()
                loss_rot = self.model.compute_loss(rot_x, rot_y)
                loss_rot = loss_rot + angle_l2 * (self.model.rotation_params ** 2).mean()
                loss_rot.backward()
                torch.nn.utils.clip_grad_norm_([self.model.rotation_params], 1.0)
                self.optimizer_rot.step()

            # ---- Unfreeze before α update so gumbel-softmax can backprop to α ----
            if hasattr(self.model, "fixed_one_hot"):
                delattr(self.model, "fixed_one_hot")

            # ---- One update on α (architecture parameters) ----
            arch_x, arch_y = next(train_iterator)
            self.optimizer_arch.zero_grad()
            loss_arch = self.model.compute_loss(arch_x, arch_y)
            loss_total = loss_arch + ent_beta * self.model.architecture_negative_entropy()
            loss_total.backward()
            torch.nn.utils.clip_grad_norm_([self.model.P, self.model.Q], 1.0)
            self.optimizer_arch.step()

            # bookkeeping / eval
            self.search_rot_loss_history.append(loss_rot.item())
            self.search_arch_loss_history.append(loss_arch.item())

            if perform_eval and (epoch + 1) % eval_every == 0:
                performance = self.evaluate_derived_circuit()
                with torch.no_grad():
                    X_val, y_val = self.validation_data.tensors
                    out = self.model(X_val)
                    preds = out.argmax(dim=1)
                    pred_hist = torch.bincount(preds, minlength=self.model.n_classes).tolist()
                    val_acc = (preds == y_val.to(preds.device)).float().mean().item()
                print(f"epoch={epoch+1} pred_hist={pred_hist} val_acc={val_acc:.4f}")
                self.derived_circuit_performance_history.append({"epoch": epoch + 1, "performance": performance})
                pbar.set_postfix(loss=f"{loss_arch.item():.4f}", acc=f"{performance:.4f}")
            else:
                pbar.set_postfix(loss=f"{loss_arch.item():.4f}")


    def derive_best_circuit(self):
        gate_indices = self.model.architecture_logits.detach().cpu().argmax(dim=-1)
        n_qubits, n_layers = self.model.n_qubits, self.model.n_layers
        qc = QuantumCircuit(n_qubits)
        param_counter = 0
        for l in range(n_layers):
            for q in range(n_qubits):
                gate_info = self.model.gate_pool[gate_indices[l, q].item()]
                if gate_info["gate"] is None:
                    continue
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

    def init_quantum_nn(
        self,
        circuit,
        input_dim,
        output_dim,
        use_gpu=False,
        quantum_lr=0.01,
        epochs_quantum=20,
    ):
        model_q = QuantumNN(
            ansatz=circuit,
            n_qubits=input_dim,
            num_classes=output_dim,
            use_gpu=use_gpu,
            gradient_method="guided_spsa",
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

    def final_evaluation(
        self,
        train_dataset,
        val_dataset,
        test_dataset,
        retrain_epochs=10,
        retrain_lr=0.05,
    ):
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
            epochs_quantum=retrain_epochs,
        )
        model.fit(
            train_dataset,
            val_dataset,
            epochs=retrain_epochs,
            optimizer=optimizer,
            scheduler=scheduler,
            verbose=True,
            eval_every=2,
        )
        # === BEGIN: permutation-calibrated test scoring (QuantumNN unchanged) ===
        # Get tensors and move to the same device as the QuantumNN

        # === Permutation-calibrated test scoring (QuantumNN unchanged) ===
        # DataLoaders → full tensors on the model device
        X_train, y_train = _loader_to_tensors(train_dataset, model.device)
        X_test, y_test = _loader_to_tensors(test_dataset, model.device)

        with torch.no_grad():
            # 1) calibrate permutation on TRAIN (use probs, since helper expects probs)
            logp_tr = model(X_train)  # (N_tr, C) log-probs
            P = _learn_perm_from_probs(logp_tr.exp(), y_train, self.model.n_classes)

            # 2) apply to TEST
            logp_te = model(X_test)  # (N_te, C) log-probs
            probs_te = logp_te.exp() @ P.t()
            y_pred = probs_te.argmax(dim=1)

            test_acc = (y_pred == y_test).float().mean().item()
            test_loss = F.nll_loss((probs_te + 1e-12).log(), y_test.long()).item()
            cm = confusion_matrix(y_test.cpu().numpy(), y_pred.cpu().numpy())

        print(
            f"Test Results: Loss={test_loss:.4f}, Acc={test_acc:.4f}, F1={(test_acc if self.model.n_classes==2 else None) if False else test_acc*0+0:.4f}"
        )
        return {
            "model": model,
            "accuracy": test_acc,
            "confusion_matrix": cm,
            "predictions": y_pred.cpu().numpy(),
        }

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
        self.model.fix_architecture_()  # forward should now use fixed_one_hot
        self.model.P.requires_grad_(False)  # freeze arch params (LEAVES)
        self.model.Q.requires_grad_(False)

        # --- optimize rotations only ---
        optimizer = optim.Adam([self.model.rotation_params], lr=lr)
        train_iterator = itertools.cycle(self.train_loader)
        for _ in range(training_steps):
            xb, yb = next(train_iterator)
            optimizer.zero_grad()
            loss = self.model.compute_loss(
                xb, yb
            )  # should only depend on rotation_params now
            loss.backward()
            optimizer.step()

        _calibrate_perm_on_loader(self.model, self.train_loader, max_batches=4)

        # --- validation accuracy ---
        X_val, y_val = self.validation_data.tensors
        with torch.no_grad():
            out = self.model(X_val)  # (N_val, C)
            if getattr(self.model, "P_align", None) is not None:
                out = out @ self.model.P_align.t()
            pred = out.argmax(dim=1)
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
