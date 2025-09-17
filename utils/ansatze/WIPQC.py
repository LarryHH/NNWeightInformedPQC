try:
    from .GenericAnsatz import GenericAnsatz
    from .WIPQCEntanglementSelection import select_entanglers
except ImportError:
    from GenericAnsatz import GenericAnsatz
    from WIPQCEntanglementSelection import select_entanglers

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter  # For trainable parameters
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, OrderedDict, Optional  # Added Optional
from collections import OrderedDict as OrderedDictType  # For type hinting
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
            weights_and_biases[f"{name}.weight"] = (
                module.weight.detach().cpu().numpy().astype(np.float32)
            )
            weights_and_biases[f"{name}.bias"] = (
                module.bias.detach().cpu().numpy().astype(np.float32)
            )

            if name not in layer_base_names_in_order:
                layer_base_names_in_order.append(name)
    return weights_and_biases, layer_base_names_in_order


# --- Helper Functions / Utilities for PQC ---
def scale_to_angle(coeff: float, factor: float = 0.1) -> float:  # Added type hints
    """Scales a coefficient to a small angle for quantum rotations."""
    return factor * np.tanh(coeff)


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
            raise ValueError(
                f"Output dimension of weights ({self.out_features}) "
                f"must match bias dimension ({self.biases.shape[0]}) for layer {self.name}."
            )


class PQCGenerator:
    # --- EDITED PART: __init__ to accept rot_angle_derivation_strategy ---
    def __init__(
        self,
        n_qubits: int,
        angle_scale_factor: float = 0.1,
        rot_use_input_features: bool = True,
        rot_angle_derivation_strategy: str = "chunking",
        ent_corr: str = "gram",            # "cosine" | "gram"
        ent_select: str = "greedy",            # "greedy" | "mwst"
        ent_top_k: int | None = None,                  # only for "greedy"
        ent_direction: str | None = None,  # None | "norm" | "centrality"
    ):  # Default to chunking
        if n_qubits <= 0:
            raise ValueError("Num qubits positive.")
        self.n_qubits = n_qubits
        self.angle_scale_factor = angle_scale_factor
        self.rot_use_input_features = rot_use_input_features  # Default to True, can be set later if needed

        if rot_angle_derivation_strategy not in ["chunking", "svd"]:
            raise ValueError(
                "rot_angle_derivation_strategy must be 'chunking' or 'svd'."
            )
        if ent_select not in {"greedy", "mwst"}:
            raise ValueError("ent_select must be 'original', 'mwst' or 'topk'.")
        if ent_direction not in {None, "norm", "centrality"}:
            raise ValueError("ent_direction must be None, 'norm' or 'centrality'.")

        self.rot_angle_derivation_strategy = rot_angle_derivation_strategy
        self.ent_corr = ent_corr
        self.ent_select = ent_select
        self.ent_top_k = ent_top_k
        self.ent_direction = ent_direction

        self.qc = QuantumCircuit(n_qubits, name="PQC_from_NN")
        self.trainable_parameters: List[Parameter] = []
        self.initial_parameter_values: OrderedDictType[Parameter, float] = OrderedDict()
        self._param_counter = 0
        self._add_initial_hadamards()

    def _add_initial_hadamards(self):
        for i in range(self.n_qubits):
            self.qc.h(i)
        self.qc.barrier(label="H")

    def _get_new_parameter(self, base_name: str, val: float) -> Parameter:
        p_name = f"{base_name}_p{self._param_counter}"
        self._param_counter += 1
        p = Parameter(p_name)
        self.trainable_parameters.append(p)
        self.initial_parameter_values[p] = val
        return p

    # --- EDITED PART: Logic for _get_rotation_params_initial_values and its helpers ---
    def _determine_feature_matrix_for_chunking(
        self, W_matrix: np.ndarray
    ) -> Optional[np.ndarray]:
        """Determines the feature matrix (M x n_qubits) for the chunking strategy."""

        feature_matrix_for_qubits: Optional[np.ndarray] = None
        if self.rot_use_input_features:
            feature_matrix_for_qubits = W_matrix
        else:  # i.e. use output features
            feature_matrix_for_qubits = W_matrix.T
        return feature_matrix_for_qubits

    def _get_coeffs_by_chunking(
        self, feature_matrix_for_qubits: np.ndarray, b_vector: np.ndarray
    ) -> np.ndarray:
        """Derives 2 angle coefficients per qubit using chunking."""
        coeffs_array = np.zeros((self.n_qubits, 2), dtype=np.float32)  # For Ry, Rz
        M_feat = feature_matrix_for_qubits.shape[0]

        for j_q in range(self.n_qubits):
            if j_q < feature_matrix_for_qubits.shape[1]:
                f_vec = feature_matrix_for_qubits[:, j_q]
                c_y, c_z = 0.0, 0.0
                if M_feat > 0:
                    m1 = (M_feat + 1) // 2  # Chunk for Ry
                    m2 = M_feat - m1  # Chunk for Rz
                    if m1 > 0:
                        c_y = np.sum(f_vec[0:m1])
                    if m2 > 0:
                        c_z = np.sum(f_vec[m1:])

                # c_y += b_vector[0] if len(b_vector) >= 1 else 0.0
                # c_z += b_vector[1] if len(b_vector) >= 2 else 0.0
                bias_contrib = float(np.sum(b_vector)) if b_vector.size>0 else 0.0
                c_y += bias_contrib
                c_z += bias_contrib
                coeffs_array[j_q, 0] = c_y  # Ry
                coeffs_array[j_q, 1] = c_z  # Rz
        return coeffs_array

    def _get_coeffs_by_svd(
        self, W_matrix: np.ndarray, b_vector: np.ndarray
    ) -> np.ndarray:
        """Derives 2 angle coefficients per qubit using SVD modes of W_matrix."""
        coeffs_array = np.zeros((self.n_qubits, 2), dtype=np.float32)  # For Ry, Rz
        D_out, D_in = W_matrix.shape

        # SVD strategy is most natural if W_matrix maps roughly N_qubits to N_qubits
        # or if its rank allows N_qubits modes.
        try:
            U, s, Vh = np.linalg.svd(W_matrix.astype(np.float64), full_matrices=False)
            U = U.astype(np.float32)
            s = s.astype(np.float32)
            V = Vh.T.astype(np.float32)  # V columns are right singular vectors

            num_modes = len(s)

            for j_q in range(self.n_qubits):
                c_y, c_z = 0.0, 0.0
                if j_q < num_modes:  # If a j-th mode exists
                    s_j = s[j_q]
                    u_j = U[:, j_q]  # j-th left singular vector (D_out dim)
                    v_j = V[:, j_q]  # j-th right singular vector (D_in dim)

                    # # Ry angle from scaled left singular vector u_j
                    # c_y = np.sum(s_j * u_j)
                    # # Rz angle from scaled right singular vector v_j
                    # c_z = np.sum(s_j * v_j)
                    idx_u = j_q % u_j.size
                    idx_v = j_q % v_j.size
                    c_y = float(s_j * u_j[idx_u])
                    c_z = float(s_j * v_j[idx_v])

                # Add biases - using first two available bias terms
                # c_y += b_vector[0] if len(b_vector) >= 1 else 0.0
                # c_z += b_vector[1] if len(b_vector) >= 2 else 0.0
                bias_contrib = float(np.sum(b_vector)) if b_vector.size>0 else 0.0
                c_y += bias_contrib
                c_z += bias_contrib
                coeffs_array[j_q, 0] = c_y  # Ry
                coeffs_array[j_q, 1] = c_z  # Rz
        except np.linalg.LinAlgError:
            print(
                f"SVD failed for W_matrix shape ({D_out},{D_in}) in SVD mode strategy. Returning zeros."
            )
        return coeffs_array

    def _get_rotation_params_initial_values(
        self, W_matrix: np.ndarray, b_vector: np.ndarray
    ) -> np.ndarray:
        if self.rot_angle_derivation_strategy == "chunking":
            feature_matrix = self._determine_feature_matrix_for_chunking(W_matrix)
            if feature_matrix is None:
                print(
                    f"Warning: Could not determine feature matrix for chunking from W_matrix ({W_matrix.shape}). Using zeros."
                )
                return np.zeros((self.n_qubits, 2), dtype=np.float32)
            coeffs = self._get_coeffs_by_chunking(feature_matrix, b_vector)
        elif self.rot_angle_derivation_strategy == "svd":
            coeffs = self._get_coeffs_by_svd(W_matrix, b_vector)
        else: 
            raise ValueError(
                f"Unknown rot_angle_derivation_strategy: {self.rot_angle_derivation_strategy}"
            )

        initial_rot_values = np.zeros((self.n_qubits, 2), dtype=np.float32)
        for j_q in range(self.n_qubits):
            initial_rot_values[j_q, 0] = scale_to_angle(
                coeffs[j_q, 0], self.angle_scale_factor
            )  # Ry
            initial_rot_values[j_q, 1] = scale_to_angle(
                coeffs[j_q, 1], self.angle_scale_factor
            )  # Rz
        return initial_rot_values


    def _get_entangler_pairs(self, W_src: np.ndarray) -> list[tuple[int, int]]:
        """
        Return a list of qubit pairs according to the user-selected strategy.

        Uses the modular helper in entangler_selection.py:
            • correlation   : self.ent_corr
            • selection     : self.ent_select  (+ self.ent_top_k if top-k)
            • direction rule: self.ent_direction
        """
        return select_entanglers(
            W_src,
            correlation=self.ent_corr,
            selection=self.ent_select,
            top_k=self.ent_top_k,
            direction=self.ent_direction,
        )


    def add_rotation_layer(
        self, W_mat: np.ndarray, b_vec: np.ndarray, label_suffix: str = ""
    ):
        initial_rot_values = self._get_rotation_params_initial_values(
            W_mat, b_vec
        ) 
        # print(f"Rot Layer {label_suffix} Initial Values (Source W: {W_mat.shape}, b: {b_vec.shape}, N_angles: 2):\n{initial_rot_values}")

        for j in range(self.n_qubits):
            # Order: Ry then Rz (can be changed if desired)
            param_ry = self._get_new_parameter(
                f"q{j}_ry_{label_suffix}", float(initial_rot_values[j, 0])
            )
            self.qc.ry(param_ry, j)
            param_rz = self._get_new_parameter(
                f"q{j}_rz_{label_suffix}", float(initial_rot_values[j, 1])
            )
            self.qc.rz(param_rz, j)
        self.qc.barrier(label=f"Rot{label_suffix}")

    def add_entangling_layer(self, W_src: np.ndarray, label_suffix: str = ""):
        pairs = self._get_entangler_pairs(W_src)     
        if pairs:
            for control, target in pairs:                       
                self.qc.cx(control, target)
            self.qc.barrier(label=f"Ent{label_suffix}")

    def add_pqc_block(
        self, 
        current_layer: ClassicalLayerInfo, 
        next_layer: ClassicalLayerInfo,
        n_weight_layers: int,
        block_index: int,
        rot_use_w1w2_block2: bool = True, 
    ):
        lbl = str(block_index)
        # print(lbl, n_weight_layers)
        self.add_rotation_layer(current_layer.weights, current_layer.biases, label_suffix=f"{lbl}.1")
        self.add_entangling_layer(current_layer.weights, label_suffix=lbl)
        if rot_use_w1w2_block2:
            W_ef = next_layer.weights @ current_layer.weights
            b_ef = next_layer.weights @ current_layer.biases + next_layer.biases
            self.add_rotation_layer(W_ef, b_ef, label_suffix=f"{lbl}.2")
        else:
            if block_index == n_weight_layers:
                W_ef = next_layer.weights @ current_layer.weights
                b_ef = next_layer.weights @ current_layer.biases + next_layer.biases
                self.add_rotation_layer(W_ef, b_ef, label_suffix=f"{lbl}.2")

    def get_trainable_parameters_with_initial_values(
        self,
    ) -> Tuple[List[Parameter], OrderedDictType[Parameter, float]]:
        return self.trainable_parameters, self.initial_parameter_values

class WeightInformedPQCAnsatz(GenericAnsatz):
    """
    An ansatz class that generates a PQC initialized from a classical NN.
    """

    def __init__(
        self,
        classical_model: nn.Module,
        angle_scale_factor: float = 0.1,
        rot_use_input_features: bool = True,
        rot_angle_derivation_strategy: str = "chunking",
        rot_use_w1w2_block2: bool = False,  # Whether to use W1W2 block2 in PQC
        ent_corr: str = "gram",            # "cosine" | "gram"
        ent_select: str = "greedy",            # "greedy" | "mwst"
        ent_top_k: int | None = None,                  # only for "greedy"
        ent_direction: str | None = None,  # None | "norm" | "centrality"
    ):
        self.classical_model = classical_model
        self.angle_scale_factor = angle_scale_factor
        self.rot_use_input_features = rot_use_input_features
        self.rot_angle_derivation_strategy = rot_angle_derivation_strategy
        self.rot_use_w1w2_block2 = rot_use_w1w2_block2
        self.ent_corr = ent_corr
        self.ent_select = ent_select
        self.ent_top_k = ent_top_k
        self.ent_direction = ent_direction

        if not self.rot_use_input_features:
            print(
                "Warning: rot_use_input_features is set to False. " \
                "This may cause unexpected behaviour when processing the final classical layer, since this is not guaranteed to be of size [n_qubits, n_qubits];" \
                " it may be of size [n_qubits, out_dim] instead."
            )

        self._wandb, self._extracted_layer_names = get_weights_and_biases(
            self.classical_model
        )
        self.classical_layers_info: List[ClassicalLayerInfo] = (
            self._process_classical_layers()
        )

        if not self.classical_layers_info:
            raise ValueError(
                "No classical linear layers found or processed from the model."
            )

        n_qubits = self.classical_layers_info[0].in_features
        if n_qubits <= 0:
            raise ValueError(
                "Number of input features in the first classical layer must be positive to define n_qubits."
            )

        self._pqc_trainable_parameters: List[Parameter] = []
        self._pqc_initial_parameter_values: OrderedDictType[Parameter, float] = (
            OrderedDict()
        )

        super().__init__(n_qubits)

    def _process_classical_layers(self) -> List[ClassicalLayerInfo]:
        processed_layers = []
        for layer_base_name in self._extracted_layer_names:
            weight_key = f"{layer_base_name}.weight"
            bias_key = f"{layer_base_name}.bias"
            if weight_key in self._wandb and bias_key in self._wandb:
                processed_layers.append(
                    ClassicalLayerInfo(
                        weights=self._wandb[weight_key],
                        biases=self._wandb[bias_key],
                        name=layer_base_name,
                    )
                )
        return processed_layers

    def create_ansatz(self) -> QuantumCircuit:
        """
        Creates the PQC using PQCGenerator based on the classical model's layers.
        """
        pqc_gen = PQCGenerator(
            n_qubits=self.n_qubits, 
            angle_scale_factor=self.angle_scale_factor, 
            rot_use_input_features=self.rot_use_input_features, 
            rot_angle_derivation_strategy=self.rot_angle_derivation_strategy,
            ent_corr=self.ent_corr,
            ent_select=self.ent_select,
            ent_top_k=self.ent_top_k,
            ent_direction=self.ent_direction,
        )

        num_pqc_blocks = len(self.classical_layers_info) - 1
        if num_pqc_blocks < 1:
            raise ValueError(
                "Not enough classical layers (need at least 2) to form a PQC block."
            )

        for i in range(num_pqc_blocks):
            pqc_gen.add_pqc_block(
                self.classical_layers_info[i],
                self.classical_layers_info[i + 1],
                n_weight_layers=num_pqc_blocks,
                block_index=i + 1,
                rot_use_w1w2_block2=self.rot_use_w1w2_block2
            )

        self._pqc_trainable_parameters, self._pqc_initial_parameter_values = (
            pqc_gen.get_trainable_parameters_with_initial_values()
        )

        return pqc_gen.qc

    def get_params(self) -> List[Parameter]:
        if self.ansatz is None:
            self.ansatz = self.create_ansatz()  # Ensure ansatz is created
        return self._pqc_trainable_parameters

    def get_num_params(self) -> int:
        if self.ansatz is None:
            self.ansatz = self.create_ansatz()  # Ensure ansatz is created
        return len(self._pqc_trainable_parameters)

    def get_initial_values_dict(self) -> OrderedDictType[Parameter, float]:
        if self.ansatz is None:
            self.ansatz = self.create_ansatz()  # Ensure ansatz is created
        return self._pqc_initial_parameter_values

    def get_initial_point(self) -> List[float]:
        if self.ansatz is None:
            self.ansatz = self.create_ansatz()  # Ensure ansatz is created
        return [
            self._pqc_initial_parameter_values[param]
            for param in self._pqc_trainable_parameters
        ]


if __name__ == "__main__":
    from utils.nn.ClassicalNN import ClassicalNN, FlexibleNN


    rot_use_input_features = True  # Set to False if you want to use output features instead. Only affects the chunking strategy.
    rot_angle_derivation_strategy = "svd"  #  in ["chunking", "svd"]
    rot_use_w1w2_block2 = False  # Whether to use W1W2 block2 in PQC

    n_qubits = 4
    # model_c = ClassicalNN(n_qubits, hidden_dim=n_qubits, output_dim=2)
    model_c = FlexibleNN(
        input_dim=n_qubits,
        hidden_dims=[n_qubits] * 3,  # Two hidden layers of size n_qubits
        output_dim=2,  # Output layer with 2 features
        act="relu",
        condition_number=0.0,  # No condition number scaling
        scale_on_export=False,
    )
    
    ansatz = WeightInformedPQCAnsatz(
        model_c, 
        angle_scale_factor=0.1, 
        rot_use_input_features=rot_use_input_features, 
        rot_angle_derivation_strategy=rot_angle_derivation_strategy,
        rot_use_w1w2_block2=rot_use_w1w2_block2
    )
    print(
        f"{ansatz.get_name()}, n_qubits = {ansatz.get_num_qubits()}, with {ansatz.get_num_params()} parameters"
    )
    print(f"Angle derivation strategy: {ansatz.rot_angle_derivation_strategy}")
    print(ansatz.get_initial_values_dict())
    ansatz.draw_to_img('weight_informed_pqc_ansatz.png')
