try:
    from .GenericAnsatz import GenericAnsatz
except ImportError:
    from GenericAnsatz import GenericAnsatz

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter # For trainable parameters
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, OrderedDict, Optional # Added Optional
from collections import OrderedDict as OrderedDictType # For type hinting
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- PyTorch Model Weight Extraction ---
def get_weights_and_biases(model: nn.Module) -> Tuple[Dict[str, np.ndarray], List[str]]:
    """
    Extracts weights and biases from a PyTorch model's Linear layers.
    Returns a dictionary of weights/biases and a list of layer base names
    in the order they were encountered.
    """
    layer_base_names_in_order = [] 
    weights_and_biases = {}
    
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            weights_and_biases[f"{name}.weight"] = module.weight.detach().cpu().numpy().astype(np.float32)
            weights_and_biases[f"{name}.bias"] = module.bias.detach().cpu().numpy().astype(np.float32)
            
            if name not in layer_base_names_in_order:
                 layer_base_names_in_order.append(name)
    return weights_and_biases, layer_base_names_in_order

# --- Helper Functions / Utilities for PQC ---
def scale_to_angle(coeff: float, factor: float = 0.1) -> float: # Added type hints
    """Scales a coefficient to a small angle for quantum rotations."""
    return factor * np.arctan(coeff)

@dataclass
class ClassicalLayerInfo:
    """Stores weights and biases for a classical neural network layer."""
    weights: np.ndarray
    biases: np.ndarray
    name: str = "" 
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
        
        self.trainable_parameters: List[Parameter] = []
        self.initial_parameter_values: OrderedDictType[Parameter, float] = OrderedDict() 

        self._param_counter = 0 
        self._add_initial_hadamards()

    def _add_initial_hadamards(self):
        for i in range(self.n_qubits):
            self.qc.h(i)
        self.qc.barrier(label="H")

    def _get_new_parameter(self, base_name: str, initial_value: float) -> Parameter:
        param_name = f"{base_name}_p{self._param_counter}"
        self._param_counter += 1
        param = Parameter(param_name)
        self.trainable_parameters.append(param)
        self.initial_parameter_values[param] = initial_value
        return param

    # --- EDITED PART STARTS ---
    def _get_rotation_params_initial_values(self, W_matrix: np.ndarray, b_vector: np.ndarray,
                                            n_angles_per_qubit: int = 3) -> np.ndarray:
        """
        Calculates initial numerical values for rotation angles for n_qubits.
        Each qubit gets n_angles_per_qubit (typically 3 for Rz, Ry, Rx).
        Uses a chunking strategy to incorporate all features from the derived
        feature vector for each qubit.
        """
        if n_angles_per_qubit != 3:
            # This specific implementation is tailored for 3 angles (Rz, Ry, Rx)
            # due to the chunking logic and bias association.
            raise ValueError("This implementation derives 3 rotation angles (for Rz, Ry, Rx).")

        initial_rot_values = np.zeros((self.n_qubits, 3), dtype=np.float32) # For Rz, Ry, Rx
        D_out, D_in = W_matrix.shape
        
        # This matrix will have shape (M_features_per_qubit_vector, self.n_qubits)
        # Each column is the M-dimensional feature vector for the corresponding qubit.
        feature_matrix_for_qubits: Optional[np.ndarray] = None 

        # --- Step 1: Determine feature_matrix_for_qubits (M rows, n_qubits columns) ---
        # This matrix aligns classical features with PQC qubits.
        if D_in == self.n_qubits: 
            feature_matrix_for_qubits = W_matrix 
        elif D_out == self.n_qubits: 
            feature_matrix_for_qubits = W_matrix.T 
        elif D_in > self.n_qubits and D_out >= 1: 
            # print(f"Info: W_matrix shape ({D_out},{D_in}) -> Using first {self.n_qubits} columns.")
            feature_matrix_for_qubits = W_matrix[:, :self.n_qubits] 
        elif D_out > self.n_qubits and D_in >= 1: 
            # print(f"Info: W_matrix shape ({D_out},{D_in}) -> Using first {self.n_qubits} rows (transposed).")
            feature_matrix_for_qubits = W_matrix[:self.n_qubits, :].T 
        else: 
            # print(f"Info: W_matrix shape ({D_out},{D_in}) -> Attempting SVD.")
            try:
                U, s, Vh = np.linalg.svd(W_matrix.astype(np.float64), full_matrices=False) 
                U = U.astype(np.float32)
                s = s.astype(np.float32)
                
                principal_components = U @ np.diag(s[:min(D_out, D_in)]) 
                
                if principal_components.shape[1] >= self.n_qubits: 
                    feature_matrix_for_qubits = principal_components[:, :self.n_qubits]
                elif principal_components.shape[0] >= self.n_qubits: 
                    feature_matrix_for_qubits = principal_components[:self.n_qubits, :].T
                else:
                    # print(f"Warning: SVD for W_matrix shape ({D_out},{D_in}) did not yield enough features for {self.n_qubits} qubits. Returning zeros.")
                    return initial_rot_values 
            except np.linalg.LinAlgError:
                # print(f"SVD failed for W_matrix shape ({D_out},{D_in}). Returning zeros.")
                return initial_rot_values 
        # --- End of block to determine feature_matrix_for_qubits ---

        if feature_matrix_for_qubits is None:
            # print(f"Error: Could not determine feature_matrix_for_qubits for W_matrix ({D_out},{D_in}). Returning zeros.")
            return initial_rot_values 
        
        M_features_per_qubit_vector = feature_matrix_for_qubits.shape[0]

        # --- Step 2: Extract 3 angle coefficients (cz, cy, cx) using chunking ---
        for j_qubit in range(self.n_qubits):
            if j_qubit < feature_matrix_for_qubits.shape[1]: 
                
                feature_vector_fj = feature_matrix_for_qubits[:, j_qubit] 

                coeff_z, coeff_y, coeff_x = 0.0, 0.0, 0.0 
                
                if M_features_per_qubit_vector > 0:
                    m1_size = (M_features_per_qubit_vector + 2) // 3 
                    m2_size = (M_features_per_qubit_vector - m1_size + 1) // 2
                    m3_size = M_features_per_qubit_vector - m1_size - m2_size

                    idx1_end = m1_size
                    idx2_start = m1_size
                    idx2_end = m1_size + m2_size
                    idx3_start = m1_size + m2_size
                    
                    if m1_size > 0:
                        coeff_z = np.sum(feature_vector_fj[0:idx1_end])
                    if m2_size > 0:
                        coeff_y = np.sum(feature_vector_fj[idx2_start:idx2_end])
                    if m3_size > 0:
                        coeff_x = np.sum(feature_vector_fj[idx3_start:])
                
                coeff_z += b_vector[0] if len(b_vector) >= 1 else 0.0
                coeff_y += b_vector[1] if len(b_vector) >= 2 else 0.0
                coeff_x += b_vector[2] if len(b_vector) >= 3 else 0.0
                
                initial_rot_values[j_qubit, 0] = scale_to_angle(coeff_z, self.angle_scale_factor) 
                initial_rot_values[j_qubit, 1] = scale_to_angle(coeff_y, self.angle_scale_factor) 
                initial_rot_values[j_qubit, 2] = scale_to_angle(coeff_x, self.angle_scale_factor) 
            
        return initial_rot_values
    # --- EDITED PART ENDS ---

    def _get_entangler_pairs(self, W_interaction_source: np.ndarray) -> Tuple[List[Tuple[int, int]], np.ndarray]:
        M_full = W_interaction_source.T @ W_interaction_source
        D_interaction_defined_over = M_full.shape[0] 
        M_for_entanglers = M_full

        if D_interaction_defined_over != self.n_qubits:
            if D_interaction_defined_over > self.n_qubits:
                M_for_entanglers = M_full[:self.n_qubits, :self.n_qubits]
            else: 
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
             print(f"Error: M_for_entanglers logic resulted in shape {M_for_entanglers.shape}, expected ({self.n_qubits},{self.n_qubits}). No entanglers added.")
        return entangler_pairs, M_for_entanglers


    def add_rotation_layer(self, W_matrix: np.ndarray, b_vector: np.ndarray, label_suffix: str = ""):
        initial_rot_values = self._get_rotation_params_initial_values(W_matrix, b_vector)
        # print(f"Rot Layer {label_suffix} Initial Values (Source W: {W_matrix.shape}, b: {b_vector.shape}):\n{initial_rot_values}")
        
        for j in range(self.n_qubits):
            param_rz = self._get_new_parameter(f"q{j}_rz_{label_suffix}", float(initial_rot_values[j, 0]))
            self.qc.rz(param_rz, j)
            param_ry = self._get_new_parameter(f"q{j}_ry_{label_suffix}", float(initial_rot_values[j, 1]))
            self.qc.ry(param_ry, j)
            param_rx = self._get_new_parameter(f"q{j}_rx_{label_suffix}", float(initial_rot_values[j, 2]))
            self.qc.rx(param_rx, j)
        self.qc.barrier(label=f"Rot{label_suffix}")

    def add_entangling_layer(self, W_interaction_source: np.ndarray, label_suffix: str = ""):
        ent_pairs, _ = self._get_entangler_pairs(W_interaction_source)
        if ent_pairs:
            for pair in ent_pairs:
                self.qc.cx(pair[0], pair[1])
            self.qc.barrier(label=f"Ent{label_suffix}")

    def add_pqc_block(self, current_layer: ClassicalLayerInfo, next_layer: ClassicalLayerInfo, block_index: int):
        block_label = str(block_index)
        self.add_rotation_layer(current_layer.weights, current_layer.biases, label_suffix=f"{block_label}.1")
        self.add_entangling_layer(current_layer.weights, label_suffix=block_label)
        
        W_eff = next_layer.weights @ current_layer.weights
        b_eff = next_layer.weights @ current_layer.biases + next_layer.biases
        self.add_rotation_layer(W_eff, b_eff, label_suffix=f"{block_label}.2")

    def get_trainable_parameters_with_initial_values(self) -> Tuple[List[Parameter], OrderedDictType[Parameter, float]]:
        return self.trainable_parameters, self.initial_parameter_values

    def draw(self, output_format='text', fold_text=-1):
        try:
            return self.qc.draw(output=output_format, fold=fold_text) 
        except ImportError as e:
            print(f"Drawing with {output_format} failed ({e}). Try 'text' or ensure dependencies are installed.")
            if output_format != 'text':
                return self.qc.draw(output='text', fold=fold_text)
        except Exception as e:
            print(f"Could not draw circuit: {e}")
        return None

class WeightInformedPQCAnsatz(GenericAnsatz):
    """
    An ansatz class that generates a PQC initialized from a classical NN.
    """
    def __init__(self, classical_model: nn.Module, angle_scale_factor: float = 0.1):
        self.classical_model = classical_model
        self.angle_scale_factor = angle_scale_factor

        # Process classical layers to determine n_qubits and store info
        self._wandb, self._extracted_layer_names = get_weights_and_biases(self.classical_model)
        self.classical_layers_info: List[ClassicalLayerInfo] = self._process_classical_layers()

        if not self.classical_layers_info:
            raise ValueError("No classical linear layers found or processed from the model.")
        
        n_qubits = self.classical_layers_info[0].in_features
        if n_qubits <= 0:
             raise ValueError("Number of input features in the first classical layer must be positive to define n_qubits.")

        # Initialize attributes that will be populated by PQCGenerator via create_ansatz
        self._pqc_trainable_parameters: List[Parameter] = []
        self._pqc_initial_parameter_values: OrderedDictType[Parameter, float] = OrderedDict()
        
        super().__init__(n_qubits) # Calls self.create_ansatz()
        # After super().__init__() calls create_ansatz(), self.ansatz is populated.
        # self._pqc_trainable_parameters and self._pqc_initial_parameter_values are also populated.

    def _process_classical_layers(self) -> List[ClassicalLayerInfo]:
        """Helper to convert extracted weights/biases into ClassicalLayerInfo objects."""
        processed_layers = []
        for layer_base_name in self._extracted_layer_names:
            weight_key = f"{layer_base_name}.weight"
            bias_key = f"{layer_base_name}.bias"
            if weight_key in self._wandb and bias_key in self._wandb:
                processed_layers.append(
                    ClassicalLayerInfo(
                        weights=self._wandb[weight_key],
                        biases=self._wandb[bias_key],
                        name=layer_base_name
                    )
                )
            else:
                print(f"Warning: Could not find weights/biases for layer base name '{layer_base_name}' during processing.")
        return processed_layers

    def create_ansatz(self) -> QuantumCircuit:
        """
        Creates the PQC using PQCGenerator based on the classical model's layers.
        This method is called by GenericAnsatz's __init__.
        """
        pqc_gen = PQCGenerator(self.n_qubits, self.angle_scale_factor)
        
        num_pqc_blocks = len(self.classical_layers_info) - 1
        if num_pqc_blocks < 1:
            print("Warning: Not enough classical layers (need at least 2) to form a PQC block. "
                  "Resulting PQC will only have initial Hadamards.")
            if self.classical_layers_info: # If only one classical layer, add one rotation layer
                 print("Adding a single rotation layer based on the first classical layer.")
                 pqc_gen.add_rotation_layer(
                     self.classical_layers_info[0].weights,
                     self.classical_layers_info[0].biases,
                     label_suffix="0.1" # Block 0, layer 1
                 )
        else:
            for i in range(num_pqc_blocks):
                current_classical_layer = self.classical_layers_info[i]
                next_classical_layer = self.classical_layers_info[i+1]
                pqc_gen.add_pqc_block(current_classical_layer, next_classical_layer, block_index=i+1)
        
        # Store the parameters and initial values from the generator
        self._pqc_trainable_parameters, self._pqc_initial_parameter_values = \
            pqc_gen.get_trainable_parameters_with_initial_values()
            
        return pqc_gen.qc

    def get_params(self) -> List[Parameter]:
        """Return the trainable Qiskit Parameter objects of this ansatz."""
        # Ensure ansatz is created if get_params is called before it might have been
        if self.ansatz is None: self.ansatz = self.create_ansatz()
        return self._pqc_trainable_parameters

    def get_num_params(self) -> int:
        """Return the number of trainable parameters in this ansatz."""
        if self.ansatz is None: self.ansatz = self.create_ansatz()
        return len(self._pqc_trainable_parameters)

    def get_initial_values_dict(self) -> OrderedDictType[Parameter, float]:
        """Returns the dictionary of Qiskit Parameters mapped to their initial numerical values."""
        if self.ansatz is None: self.ansatz = self.create_ansatz()
        return self._pqc_initial_parameter_values

    def get_initial_point(self) -> List[float]:
        """Returns a list of initial numerical parameter values, ordered as per get_params()."""
        if self.ansatz is None: self.ansatz = self.create_ansatz()
        return [self._pqc_initial_parameter_values[param] for param in self._pqc_trainable_parameters]

    def draw(self, **kwargs):
        """Draw the ansatz circuit. Overrides GenericAnsatz to use self.ansatz directly."""
        # Optional: Bind parameters for drawing with values, or draw with parameter names
        # circuit_to_draw = self.ansatz.assign_parameters(self.get_initial_point()) # Shows numbers
        circuit_to_draw = self.ansatz # Shows parameter names
        
        if circuit_to_draw:
            kwargs.setdefault('fold', -1)
            try:
                print(circuit_to_draw.draw(**kwargs))
            except Exception as e:
                print(f"Error drawing ClassicalInspiredPQCAnsatz circuit: {e}")
        else:
            print("ClassicalInspiredPQCAnsatz: Ansatz circuit has not been created yet.")
