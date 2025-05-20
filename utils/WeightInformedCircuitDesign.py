import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter # For trainable parameters
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, OrderedDict
from collections import OrderedDict as OrderedDictType # For type hinting
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- PyTorch Model Weight Extraction ---
def get_weights_and_biases(model: nn.Module) -> Tuple[Dict[str, np.ndarray], List[str]]:
    """
    Extracts weights and biases from a PyTorch model's Linear layers.
    Returns a dictionary where keys are like 'fc1.weight', 'fc1.bias'.
    """
    layer_base_names_in_order = []
    weights_and_biases = {}
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            weights_and_biases[f"{name}.weight"] = module.weight.detach().cpu().numpy()
            weights_and_biases[f"{name}.bias"] = module.bias.detach().cpu().numpy()
            if name not in layer_base_names_in_order:
                layer_base_names_in_order.append(name)
    return weights_and_biases, layer_base_names_in_order

# --- Helper Functions / Utilities for PQC ---
def scale_to_angle(coeff, factor=0.1):
    """Scales a coefficient to a small angle for quantum rotations."""
    return factor * np.arctan(coeff)

@dataclass
class ClassicalLayerInfo:
    """Stores weights and biases for a classical neural network layer."""
    weights: np.ndarray
    biases: np.ndarray
    name: str = "" # Optional name for the layer
    in_features: int = 0
    out_features: int = 0

    def __post_init__(self):
        if self.weights.ndim != 2:
            raise ValueError(f"Weights for layer {self.name} must be a 2D array.")
        if self.biases.ndim != 1:
            raise ValueError(f"Biases for layer {self.name} must be a 1D array.")
        
        self.out_features, self.in_features = self.weights.shape
        
        if self.out_features != self.biases.shape[0]:
            raise ValueError(f"Output dimension of weights ({self.out_features}) "
                             f"must match bias dimension ({self.biases.shape[0]}) for layer {self.name}.")
class PQCGenerator:
    """
    Generates a Parameterized Quantum Circuit (PQC) based on classical
    neural network layer parameters, with trainable rotation angles.
    """
    def __init__(self, n_qubits: int, angle_scale_factor: float = 0.1):
        if n_qubits <= 0:
            raise ValueError("Number of qubits must be positive.")
        self.n_qubits = n_qubits
        self.angle_scale_factor = angle_scale_factor
        self.qc = QuantumCircuit(n_qubits, name="PQC_from_NN")
        
        # For trainable parameters
        self.trainable_parameters: List[Parameter] = []
        self.initial_parameter_values: OrderedDictType[Parameter, float] = OrderedDict() # Store initial values

        self._param_counter = 0 # To ensure unique parameter names
        self._add_initial_hadamards()

    def _add_initial_hadamards(self):
        for i in range(self.n_qubits):
            self.qc.h(i)
        self.qc.barrier(label="H")

    def _get_new_parameter(self, base_name: str, initial_value: float) -> Parameter:
        """Creates a new Qiskit Parameter, stores it, and its initial value."""
        param_name = f"{base_name}_p{self._param_counter}"
        self._param_counter += 1
        param = Parameter(param_name)
        self.trainable_parameters.append(param)
        self.initial_parameter_values[param] = initial_value
        return param

    def _get_rotation_params_initial_values(self, W_matrix: np.ndarray, b_vector: np.ndarray,
                                            n_angles_per_qubit: int = 3) -> np.ndarray:
        # This function now ONLY calculates the initial numerical values for rotations.
        # Parameter objects will be created in add_rotation_layer.
        initial_rot_values = np.zeros((self.n_qubits, n_angles_per_qubit), dtype=np.float32)
        D_out, D_in = W_matrix.shape
        feature_matrix_for_qubits = None 

        if D_in == self.n_qubits: 
            feature_matrix_for_qubits = W_matrix
        elif D_out == self.n_qubits: 
            feature_matrix_for_qubits = W_matrix.T
        elif D_in > self.n_qubits and D_out >= n_angles_per_qubit: 
            # print(f"Info: W_matrix shape ({D_out},{D_in}) is wider than n_qubits ({self.n_qubits}). Using first {self.n_qubits} columns.")
            feature_matrix_for_qubits = W_matrix[:, :self.n_qubits]
        elif D_out > self.n_qubits and D_in >= n_angles_per_qubit: 
            # print(f"Info: W_matrix shape ({D_out},{D_in}) is taller than n_qubits ({self.n_qubits}). Using first {self.n_qubits} rows (transposed).")
            feature_matrix_for_qubits = W_matrix[:self.n_qubits, :].T
        else: 
            # print(f"Info: W_matrix shape ({D_out},{D_in}) has general mismatch for n_qubits ({self.n_qubits}). Attempting SVD.")
            try:
                U, s, Vh = np.linalg.svd(W_matrix.astype(np.float64), full_matrices=False)
                U = U.astype(np.float32)
                s = s.astype(np.float32)
                Vh = Vh.astype(np.float32)
                principal_components = U @ np.diag(s[:min(D_out, D_in)])
                
                if principal_components.shape[1] >= self.n_qubits: 
                    feature_matrix_for_qubits = principal_components[:, :self.n_qubits]
                elif principal_components.shape[0] >= self.n_qubits: 
                    feature_matrix_for_qubits = principal_components[:self.n_qubits, :].T
                else:
                    print(f"Warning: SVD for W_matrix shape ({D_out},{D_in}) did not yield enough features for {self.n_qubits} qubits. Returning zeros for initial values.")
                    return initial_rot_values
            except np.linalg.LinAlgError:
                print(f"SVD failed for W_matrix shape ({D_out},{D_in}). Returning zeros for initial values.")
                return initial_rot_values

        if feature_matrix_for_qubits is None:
            print(f"Error: Could not determine feature_matrix_for_qubits for W_matrix ({D_out},{D_in}). Returning zeros for initial values.")
            return initial_rot_values
        
        num_rows_as_features = feature_matrix_for_qubits.shape[0]
        for j_qubit in range(self.n_qubits):
            if j_qubit < feature_matrix_for_qubits.shape[1]: 
                for i_angle in range(n_angles_per_qubit):
                    if i_angle < num_rows_as_features:
                        bias_val = b_vector[i_angle] if i_angle < len(b_vector) else 0.0
                        coeff = feature_matrix_for_qubits[i_angle, j_qubit] + bias_val
                        initial_rot_values[j_qubit, i_angle] = scale_to_angle(coeff, self.angle_scale_factor)
        return initial_rot_values

    def _get_entangler_pairs(self, W_interaction_source: np.ndarray) -> Tuple[List[Tuple[int, int]], np.ndarray]:
        M_full = W_interaction_source.T @ W_interaction_source
        D_interaction_defined_over = M_full.shape[0] 
        M_for_entanglers = M_full

        if D_interaction_defined_over != self.n_qubits:
            # print(f"Info: Interaction matrix source input dim {D_interaction_defined_over} != n_qubits {self.n_qubits}.")
            if D_interaction_defined_over > self.n_qubits:
                # print(f"Taking top-left {self.n_qubits}x{self.n_qubits} submatrix for entanglers from M_full ({M_full.shape}).")
                M_for_entanglers = M_full[:self.n_qubits, :self.n_qubits]
            else: 
                # print(f"Padding interaction matrix to {self.n_qubits}x{self.n_qubits} from M_full ({M_full.shape}).")
                M_for_entanglers = np.zeros((self.n_qubits, self.n_qubits), dtype=np.float32)
                M_for_entanglers[:D_interaction_defined_over, :D_interaction_defined_over] = M_full
        
        entangler_pairs = []
        if self.n_qubits > 1 and M_for_entanglers.shape == (self.n_qubits, self.n_qubits):
            edge_strengths = [{'strength': abs(M_for_entanglers[i, k]), 'pair': (i, k)}
                              for i in range(self.n_qubits) for k in range(i + 1, self.n_qubits)]
            sorted_edges = sorted(edge_strengths, key=lambda x: x['strength'], reverse=True)
            num_entanglers_to_add = min(self.n_qubits, len(sorted_edges))
            entangler_pairs = [edge['pair'] for edge in sorted_edges[:num_entanglers_to_add]]
        elif M_for_entanglers.shape != (self.n_qubits, self.n_qubits):
             print(f"Error: M_for_entanglers shape {M_for_entanglers.shape} is not ({self.n_qubits},{self.n_qubits}). No entanglers added for this step.")
        return entangler_pairs, M_for_entanglers


    def add_rotation_layer(self, W_matrix: np.ndarray, b_vector: np.ndarray, label_suffix: str = ""):
        initial_rot_values = self._get_rotation_params_initial_values(W_matrix, b_vector)
        print(f"Rot Layer {label_suffix} Initial Values (Source W: {W_matrix.shape}, b: {b_vector.shape}):\n{initial_rot_values}")
        
        for j in range(self.n_qubits):
            # Create Qiskit Parameters for each angle and use them in gates
            # Rz
            param_rz = self._get_new_parameter(f"q{j}_rz_{label_suffix}", float(initial_rot_values[j, 0]))
            self.qc.rz(param_rz, j)
            # Ry
            param_ry = self._get_new_parameter(f"q{j}_ry_{label_suffix}", float(initial_rot_values[j, 1]))
            self.qc.ry(param_ry, j)
            # Rx
            param_rx = self._get_new_parameter(f"q{j}_rx_{label_suffix}", float(initial_rot_values[j, 2]))
            self.qc.rx(param_rx, j)
        self.qc.barrier(label=f"Rot{label_suffix}")

    def add_entangling_layer(self, W_interaction_source: np.ndarray, label_suffix: str = ""):
        ent_pairs, M_actual = self._get_entangler_pairs(W_interaction_source)
        # print(f"M Interaction for Ent{label_suffix} (Source W_in_dim: {W_interaction_source.shape[1]}, M_actual dim {M_actual.shape}):\n{M_actual}")
        # print(f"Ent{label_suffix} Pairs: {ent_pairs}")
        if ent_pairs:
            for pair in ent_pairs:
                self.qc.cx(pair[0], pair[1])
            self.qc.barrier(label=f"Ent{label_suffix}")
        else:
            # print(f"No entangler pairs selected for Ent{label_suffix}.")
            pass # Avoid too much printing, can be enabled for debug


    def add_pqc_block(self, current_layer: ClassicalLayerInfo, next_layer: ClassicalLayerInfo, block_index: int):
        print(f"\n--- PQC Block {block_index} ---")
        block_label = str(block_index)

        # Rotation Layer 1 (e.g., Rot1.1 for block_index=1)
        self.add_rotation_layer(current_layer.weights, current_layer.biases, label_suffix=f"{block_label}.1")

        # Entangling Layer (e.g., Ent1 for block_index=1)
        self.add_entangling_layer(current_layer.weights, label_suffix=block_label)
        
        # Effective weights and biases for the second rotation layer in this block
        W_eff = next_layer.weights @ current_layer.weights
        b_eff = next_layer.weights @ current_layer.biases + next_layer.biases
        self.add_rotation_layer(W_eff, b_eff, label_suffix=f"{block_label}.2")

    def get_trainable_parameters_with_initial_values(self) -> Tuple[List[Parameter], OrderedDictType[Parameter, float]]:
        """Returns the list of Qiskit Parameter objects and their initial numerical values."""
        return self.trainable_parameters, self.initial_parameter_values

    def draw(self, output_format='text', fold_text=-1):
        print("\n--- Final PQC Structure ---")
        try:
            # Bind parameters to their initial values for drawing, if desired
            # This makes the drawing show numbers instead of parameter names.
            # For seeing parameter names, don't bind.
            # bound_circuit = self.qc.assign_parameters(self.initial_parameter_values)
            # return bound_circuit.draw(output=output_format, fold=fold_text)
            return self.qc.draw(output=output_format, fold=fold_text) # Shows parameter names
        except ImportError as e:
            print(f"Drawing with {output_format} failed ({e}). Try 'text' or ensure dependencies are installed.")
            if output_format != 'text':
                return self.qc.draw(output='text', fold=fold_text)
        except Exception as e:
            print(f"Could not draw circuit: {e}")
        return None