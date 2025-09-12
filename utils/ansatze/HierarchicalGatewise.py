# # He, Z., Deng, M., Zheng, S., Li, L., & Situ, H. (2024, March). Training-free quantum architecture search. In Proceedings of the AAAI conference on artificial intelligence (Vol. 38, No. 11, pp. 12430-12438).import random

# """
# 1. 2 distinct search spaces: gatewise, layerwise
# 2. for each generate 50,000 random candidate circuits. 
# 3. DAG filter
# 4. Expressibility filter
# 5. 5000 (10% remaining) best candidates, ranked by expressibility
# """


# import random
# import numpy as np
# import qiskit
# from scipy.special import softmax
# from qiskit import QuantumCircuit
# from qiskit.circuit import ParameterVector
# from qiskit.circuit.library import RXGate, RYGate, RZGate
# from qiskit.circuit.library import RXXGate, RYYGate, RZZGate, CXGate
# from utils.ansatze.GenericAnsatz import GenericAnsatz


# class HierarchicalGatewiseAnsatz(GenericAnsatz):
#     """
#     Generates a PQC by adding gates one by one using hierarchical sampling.
#     - Enforces a ring topology for two-qubit gates.
#     - Uses a biased, probabilistic method for gate selection.
#     - Enforces that num_two_qubit_gates <= num_single_qubit_gates.
#     """
#     def __init__(self, n_qubits, n_gates, add_hadamard_layer=False, seed=None):
#         self.n_gates = n_gates
#         self.seed = seed
#         self.add_hadamard_layer = add_hadamard_layer # New argument for TFIM

#         self.gate_pool = [RXGate, RYGate, RZGate, CXGate] # RXXGate, RYYGate, RZZGate]
#         self.single_qubit_indices = [0, 1, 2]
#         self.two_qubit_indices = [i for i in range(len(self.gate_pool)) if i not in self.single_qubit_indices]
#         self.n_gate_types = len(self.gate_pool)
#         self.topology = [(i, (i + 1) % n_qubits) for i in range(n_qubits)]

#         super().__init__(n_qubits)

#     def create_ansatz(self):
#         if self.seed is not None:
#             np.random.seed(self.seed)
#             random.seed(self.seed)

#         qc = QuantumCircuit(self.n_qubits)
#         param_vec = ParameterVector("θ", length=self.n_gates)
#         param_idx = 0
        
#         if self.add_hadamard_layer:
#             qc.h(range(self.n_qubits))
#             qc.barrier()
        
#         n_sq_gates = 0
#         n_tq_gates = 0

#         for _ in range(self.n_gates):
#             logits = np.random.normal(loc=0, scale=1.35, size=self.n_gate_types)
#             logits[self.single_qubit_indices] += 0.5
            
#             if n_tq_gates >= n_sq_gates:
#                 logits[self.two_qubit_indices] = -np.inf

#             probabilities = softmax(logits)
#             gate_idx = np.random.choice(self.n_gate_types, p=probabilities)
#             Gate = self.gate_pool[gate_idx]

#             # --- CORE FIX: Check if the gate is CXGate ---
#             if Gate is CXGate:
#                 if self.n_qubits >= 2:
#                     q_pair = random.choice(self.topology)
#                     # Append without a parameter
#                     qc.append(Gate(), list(q_pair))
#                     n_tq_gates += 1
#             else: # It must be a parameterized single-qubit gate
#                 q = random.randrange(self.n_qubits)
#                 # Append WITH a parameter
#                 qc.append(Gate(param_vec[param_idx]), [q])
#                 param_idx += 1
#                 n_sq_gates += 1
        
#         # Re-bind parameters for a clean, correctly-sized circuit
#         final_qc = QuantumCircuit(self.n_qubits)
#         if param_idx > 0:
#             final_params = ParameterVector("θ", length=param_idx)
#             param_map = {p: new_p for p, new_p in zip(param_vec[:param_idx], final_params)}
#             final_qc = qiskit.transpiler.passes.RemoveBarriers()(qc.assign_parameters(param_map))
#         else:
#             final_qc = qc
        
#         return final_qc
    


# if __name__ == "__main__":
#     n_qubits = 4
#     seed = 42
    
#     gw_gates = 12
#     ansatz_gw = HierarchicalGatewiseAnsatz(n_qubits, gw_gates, seed=seed)
#     ansatz_gw.draw()


import random
import numpy as np
import qiskit
from scipy.special import softmax
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import RXGate, RYGate, RZGate
from qiskit.circuit.library import RXXGate, RYYGate, RZZGate, CXGate
from utils.ansatze.GenericAnsatz import GenericAnsatz


class HierarchicalGatewiseAnsatz(GenericAnsatz):
    """
    Generates a PQC with temporal and spatial correlations.
    - Enforces a ring topology for two-qubit gates.
    - Uses a biased, probabilistic method for gate selection.
    - Enforces that num_two_qubit_gates <= num_single_qubit_gates.
    """
    def __init__(self, n_qubits, n_gates, add_hadamard_layer=False, 
                 temporal_bias=0.75, spatial_bias=1.0, seed=None):
        
        self.n_gates = n_gates
        self.add_hadamard_layer = add_hadamard_layer
        self.seed = seed
        
        # --- NEW: Bias parameters ---
        self.temporal_bias = temporal_bias # Bonus for choosing the same gate type again
        self.spatial_bias = spatial_bias   # Bonus for placing a gate near the last one

        self.gate_pool = [RXGate, RYGate, RZGate, CXGate]
        self.single_qubit_indices = [0, 1, 2]
        self.two_qubit_indices = [i for i in range(len(self.gate_pool)) if i not in self.single_qubit_indices]
        self.n_gate_types = len(self.gate_pool)
        self.topology = [(i, (i + 1) % n_qubits) for i in range(n_qubits)]

        super().__init__(n_qubits)

    def create_ansatz(self):
        if self.seed is not None:
            np.random.seed(self.seed)
            random.seed(self.seed)

        qc = QuantumCircuit(self.n_qubits)
        param_vec = ParameterVector("θ", length=self.n_gates)
        param_idx = 0
        
        if self.add_hadamard_layer:
            qc.h(range(self.n_qubits))
            qc.barrier()
        
        n_sq_gates = 0
        n_tq_gates = 0
        
        # --- NEW: State tracking variables ---
        last_gate_idx = None
        last_sq_qubit = None

        for _ in range(self.n_gates):
            # --- Gate Selection (Temporal Bias) ---
            logits = np.random.normal(loc=0, scale=1.35, size=self.n_gate_types)
            logits[self.single_qubit_indices] += 0.5
            
            # --- NEW: Apply temporal bias to favor repeating the last gate type ---
            if self.temporal_bias > 0 and last_gate_idx is not None:
                logits[last_gate_idx] += self.temporal_bias
            
            if n_tq_gates >= n_sq_gates:
                logits[self.two_qubit_indices] = -np.inf

            probabilities = softmax(logits)
            gate_idx = np.random.choice(self.n_gate_types, p=probabilities)
            Gate = self.gate_pool[gate_idx]

            # --- Gate Placement (Spatial Bias) ---
            if Gate is CXGate:
                if self.n_qubits >= 2:
                    q_pair = random.choice(self.topology)
                    qc.append(Gate(), list(q_pair))
                    n_tq_gates += 1
            else: # It's a single-qubit gate
                # --- NEW: Apply spatial bias for qubit placement ---
                if self.spatial_bias > 0 and last_sq_qubit is not None:
                    probs = np.full(self.n_qubits, 1.0)
                    neighbor1 = (last_sq_qubit + 1) % self.n_qubits
                    neighbor2 = (last_sq_qubit - 1) % self.n_qubits
                    
                    # Add bonus to the last qubit and its neighbors
                    probs[last_sq_qubit] += self.spatial_bias
                    probs[neighbor1] += self.spatial_bias / 2 
                    probs[neighbor2] += self.spatial_bias / 2
                    
                    probs /= probs.sum() # Normalize to get probabilities
                    q = np.random.choice(self.n_qubits, p=probs)
                else:
                    q = random.randrange(self.n_qubits) # Fallback to uniform random
                
                qc.append(Gate(param_vec[param_idx]), [q])
                param_idx += 1
                n_sq_gates += 1
                last_sq_qubit = q # Update state
            
            last_gate_idx = gate_idx # Update state

        # Re-bind parameters for a clean, correctly-sized circuit
        if param_idx > 0:
            final_params = ParameterVector("θ", length=param_idx)
            param_map = {p: new_p for p, new_p in zip(param_vec[:param_idx], final_params)}
            final_qc = qiskit.transpiler.passes.RemoveBarriers()(qc.assign_parameters(param_map))
        else:
            final_qc = qc
        
        return final_qc