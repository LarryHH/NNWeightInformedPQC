import torch
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import ZZFeatureMap
from qiskit.quantum_info import Statevector, SparsePauliOp
import math

class DifferentiableQuantumCircuit(torch.autograd.Function):

    @staticmethod
    def _probs_from_params(x, gate_select_one_hot, rot_params, n_qubits, n_layers, gate_pool, n_classes):
        gate_indices = torch.argmax(gate_select_one_hot, dim=-1)
        batch_size = x.shape[0]
        out_dtype = x.dtype if x.is_floating_point() else torch.float32
        output_probabilities = torch.zeros(batch_size, n_classes, device=x.device, dtype=out_dtype)

        for i in range(batch_size):
            full_qc = DifferentiableQuantumCircuit._build_full_circuit(
                x[i], gate_indices, rot_params, n_qubits, n_layers, gate_pool
            )
            sv = Statevector.from_int(0, 2**n_qubits).evolve(full_qc)
            probs = torch.as_tensor(sv.probabilities(), device=x.device, dtype=out_dtype)
            M = probs.numel()  # = 2**n_qubits
            # how many bitstrings end up in each class with idx % n_classes
            counts = torch.tensor(
                [M // n_classes + (1 if c < (M % n_classes) else 0) for c in range(n_classes)],
                device=x.device, dtype=out_dtype
            )
            idx = torch.arange(M, device=x.device) % n_classes
            binned = torch.zeros(n_classes, device=x.device, dtype=out_dtype).scatter_add_(0, idx, probs)
            binned = binned / counts
            output_probabilities[i] = binned / binned.sum()
        return output_probabilities
    

    @staticmethod
    def forward(ctx, x, gate_select_one_hot, rot_params, n_qubits, n_layers, gate_pool, n_classes):
        probs = DifferentiableQuantumCircuit._probs_from_params(
            x, gate_select_one_hot, rot_params, n_qubits, n_layers, gate_pool, n_classes
        )
        gate_indices = torch.argmax(gate_select_one_hot, dim=-1)
        ctx.save_for_backward(x, gate_select_one_hot, rot_params, gate_indices)
        ctx.n_qubits, ctx.n_layers, ctx.gate_pool, ctx.n_classes = n_qubits, n_layers, gate_pool, n_classes
        return probs

    @staticmethod
    def backward(ctx, grad_output):
        x, gate_select_one_hot, rot_params, gate_indices = ctx.saved_tensors
        n_qubits, n_layers, gate_pool, n_classes = ctx.n_qubits, ctx.n_layers, ctx.gate_pool, ctx.n_classes
        grad_x = grad_gate_select = None
        grad_rot = torch.zeros_like(rot_params)

        if ctx.needs_input_grad[2]:
            shift = np.pi / 2
            param_counter = 0
            with torch.no_grad():
                for l in range(n_layers):
                    for q in range(n_qubits):
                        gate_idx = gate_indices[l, q].item()
                        if gate_pool[gate_idx]["n_params"] > 0:
                            plus_params = rot_params.clone(); plus_params[param_counter] += shift
                            minus_params = rot_params.clone(); minus_params[param_counter] -= shift
                            probs_plus = DifferentiableQuantumCircuit._probs_from_params(
                                x, gate_select_one_hot, plus_params, n_qubits, n_layers, gate_pool, n_classes
                            )
                            probs_minus = DifferentiableQuantumCircuit._probs_from_params(
                                x, gate_select_one_hot, minus_params, n_qubits, n_layers, gate_pool, n_classes
                            )
                            dprobs = 0.5 * (probs_plus - probs_minus)

                            contrib = (grad_output * dprobs).sum(dim=1) 
                            grad_rot[param_counter] = contrib.sum()       

                            param_counter += 1

        return grad_x, grad_gate_select, grad_rot, None, None, None, None

    @staticmethod
    def _build_full_circuit(x_i, gate_indices, rot_params, n_qubits, n_layers, gate_pool, fm_reps: int = 2):
        """
        Build a *fully bound* circuit for one sample x_i:
        1) ZZFeatureMap(x) with reps=fm_reps
        2) sampled gates with numeric rot_params
        """
        qc = QuantumCircuit(n_qubits)

        fm = ZZFeatureMap(feature_dimension=n_qubits, reps=fm_reps)
        fm_params = list(fm.parameters)
        qc.compose(fm, inplace=True)

        # --- 2) Append sampled architecture with numeric params ---
        param_counter = 0
        for l in range(n_layers):
            for q in range(n_qubits):
                ginfo = gate_pool[gate_indices[l, q].item()]
                if ginfo["gate"] is None:
                    continue
                if ginfo["n_qubits"] == 1:
                    if ginfo["n_params"] > 0:
                        theta = float(rot_params[param_counter].item())
                        qc.append(ginfo["gate"](theta), [q])
                        param_counter += 1
                    else:
                        qc.append(ginfo["gate"](), [q])
                elif ginfo["n_qubits"] == 2:
                    qc.append(ginfo["gate"](), [q, (q + 1) % n_qubits])
            if n_layers > 1:
                qc.barrier()

        # --- 3) Bind feature-map params to x ---
        # Expect len(x_i) == n_qubits (as in your pipeline with PCA→n_qubits)
        x_vals = x_i.detach().cpu().numpy().tolist()
        assert len(x_vals) >= n_qubits, "x has fewer features than n_qubits"
        bind_map = {fm_params[j]: x_vals[j] for j in range(n_qubits)}
        qc = qc.assign_parameters(bind_map, inplace=False)

        return qc

    @staticmethod
    def _get_batch_expectation(x, gate_indices, rot_params, ctx, observable_list):
        batch_size = x.shape[0]
        expectations = torch.zeros(batch_size, device=x.device)
        for i in range(batch_size):
            qc = DifferentiableQuantumCircuit._build_full_circuit(
                x[i], gate_indices, rot_params, ctx.n_qubits, ctx.n_layers, ctx.gate_pool
            )
            state = Statevector.from_int(0, 2**ctx.n_qubits).evolve(qc)
            local_exp_val = sum(state.expectation_value(obs).real for obs in observable_list)
            expectations[i] = local_exp_val / len(observable_list)
        return expectations

