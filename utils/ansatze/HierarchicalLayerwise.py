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
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import RXGate, RYGate, RZGate
from qiskit.circuit.library import RXXGate, RYYGate, RZZGate, CXGate
from utils.ansatze.GenericAnsatz import GenericAnsatz


class HierarchicalLayerwiseAnsatz(GenericAnsatz):
    """
    Generates a structured, layerwise PQC using hierarchical sampling.
    - Selects one gate type per half-layer.
    - Applies the gate to even or odd sites.
    """
    def __init__(self, n_qubits, depth, add_hadamard_layer=False, seed=None):
        self.depth = depth
        self.seed = seed
        self.add_hadamard_layer = add_hadamard_layer # New argument for TFIM

        self.single_qubit_pool = [RXGate, RYGate, RZGate]
        self.two_qubit_pool = [CXGate] # [RXXGate, RYYGate, RZZGate]
        
        super().__init__(n_qubits)


    def create_ansatz(self):
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)

        qc = QuantumCircuit(self.n_qubits)
        
        if self.add_hadamard_layer:
            qc.h(range(self.n_qubits))
            qc.barrier()
        
        param_idx = 0
        max_params = self.depth * self.n_qubits 
        param_vec = ParameterVector("θ", length=max_params)

        for _ in range(self.depth):
            # Single-qubit layer (always parameterized)
            sq_gate = random.choice(self.single_qubit_pool)
            for i in range(self.n_qubits):
                qc.append(sq_gate(param_vec[param_idx]), [i])
                param_idx += 1

            # Even-pair entanglement layer
            if self.n_qubits >= 2:
                tq_gate_even = random.choice(self.two_qubit_pool)
                # --- CORE FIX 1: Check the gate type for the even layer ---
                if tq_gate_even is CXGate:
                    for i in range(0, self.n_qubits - 1, 2):
                        qc.append(tq_gate_even(), [i, i+1])
                else: # For other potential parameterized gates
                    for i in range(0, self.n_qubits - 1, 2):
                        qc.append(tq_gate_even(param_vec[param_idx]), [i, i+1])
                        param_idx += 1
            
            # Odd-pair entanglement layer
            if self.n_qubits >= 3:
                tq_gate_odd = random.choice(self.two_qubit_pool)
                # --- CORE FIX 2: Check the gate type for the odd layer ---
                if tq_gate_odd is CXGate:
                    for i in range(1, self.n_qubits - 1, 2):
                        qc.append(tq_gate_odd(), [i, i+1])
                else: # For other potential parameterized gates
                    for i in range(1, self.n_qubits - 1, 2):
                        qc.append(tq_gate_odd(param_vec[param_idx]), [i, i+1])
                        param_idx += 1
        
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
    
    # Note the add_hadamard_layer=True
    lw_depth = 2
    ansatz_tfim = HierarchicalLayerwiseAnsatz(n_qubits, lw_depth, seed=seed, add_hadamard_layer=True)
    ansatz_tfim.draw()