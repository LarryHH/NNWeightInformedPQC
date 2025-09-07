# He, Z., Deng, M., Zheng, S., Li, L., & Situ, H. (2024, March). Training-free quantum architecture search. In Proceedings of the AAAI conference on artificial intelligence (Vol. 38, No. 11, pp. 12430-12438).import random

"""
1. 2 distinct search spaces: gatewise, layerwise
2. for each generate 50,000 random candidate circuits. 
3. DAG filter
4. Expressibility filter
5. 5000 (10% remaining) best candidates, ranked by expressibility
"""


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
    Generates a PQC by adding gates one by one using hierarchical sampling.
    - Enforces a ring topology for two-qubit gates.
    - Uses a biased, probabilistic method for gate selection.
    - Enforces that num_two_qubit_gates <= num_single_qubit_gates.
    """
    def __init__(self, n_qubits, n_gates, add_hadamard_layer=False, seed=None):
        self.n_gates = n_gates
        self.seed = seed
        self.add_hadamard_layer = add_hadamard_layer # New argument for TFIM

        self.gate_pool = [RXGate, RYGate, RZGate, CXGate] # RXXGate, RYYGate, RZZGate]
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

        for _ in range(self.n_gates):
            logits = np.random.normal(loc=0, scale=1.35, size=self.n_gate_types)
            logits[self.single_qubit_indices] += 0.5
            
            if n_tq_gates >= n_sq_gates:
                logits[self.two_qubit_indices] = -np.inf

            probabilities = softmax(logits)
            gate_idx = np.random.choice(self.n_gate_types, p=probabilities)
            Gate = self.gate_pool[gate_idx]

            # --- CORE FIX: Check if the gate is CXGate ---
            if Gate is CXGate:
                if self.n_qubits >= 2:
                    q_pair = random.choice(self.topology)
                    # Append without a parameter
                    qc.append(Gate(), list(q_pair))
                    n_tq_gates += 1
            else: # It must be a parameterized single-qubit gate
                q = random.randrange(self.n_qubits)
                # Append WITH a parameter
                qc.append(Gate(param_vec[param_idx]), [q])
                param_idx += 1
                n_sq_gates += 1
        
        # Re-bind parameters for a clean, correctly-sized circuit
        final_qc = QuantumCircuit(self.n_qubits)
        if param_idx > 0:
            final_params = ParameterVector("θ", length=param_idx)
            param_map = {p: new_p for p, new_p in zip(param_vec[:param_idx], final_params)}
            final_qc = qiskit.transpiler.passes.RemoveBarriers()(qc.assign_parameters(param_map))
        else:
            final_qc = qc
        
        return final_qc
    


if __name__ == "__main__":
    n_qubits = 4
    seed = 42
    
    gw_gates = 12
    ansatz_gw = HierarchicalGatewiseAnsatz(n_qubits, gw_gates, seed=seed)
    ansatz_gw.draw()