try:
    from .GenericAnsatz import GenericAnsatz
except ImportError:
    from GenericAnsatz import GenericAnsatz

import random
import qiskit
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector, Barrier
from qiskit.circuit.library import RYGate, RZGate, CXGate, XGate, YGate, ZGate

class RandomPQCAnsatz(GenericAnsatz):
    """
    Random Parameterized Quantum Circuit (PQC) ansatz.
    
    Generates a random circuit with trainable parameters for variational quantum algorithms.
    The circuit has a specified number of layers, with randomized single-qubit and two-qubit gates.
    """
    def __init__(self, n_qubits, depth, seed=None, 
                 max_single_qubit_gates=None, max_two_qubit_gates=None,
                 single_qubit_gate_pool=None, two_qubit_gate_pool=None):
        """
        Initialize a random parameterized quantum circuit ansatz.
        
        Args:
            n_qubits: Number of qubits in the circuit
            depth: Number of layers in the circuit
            seed: Random seed for reproducibility
            max_single_qubit_gates: Maximum number of single-qubit gates per layer
            max_two_qubit_gates: Maximum number of two-qubit gates per layer
            single_qubit_gate_pool: List of single-qubit gates to use (defaults to RY and RZ)
            two_qubit_gate_pool: List of two-qubit gates to use (defaults to CX)
        """
        self.depth = depth
        self.seed = seed
        self.max_single_qubit_gates = max_single_qubit_gates or n_qubits
        self.max_two_qubit_gates = max_two_qubit_gates or n_qubits // 2
        
        # Default gate pools if not provided
        self.single_qubit_pool = single_qubit_gate_pool or [
            RYGate(0),  # placeholder for trainable RY
            RZGate(0),  # placeholder for trainable RZ
        ]
        self.two_qubit_pool = two_qubit_gate_pool or [CXGate()]
        
        # Pairs of gates that shouldn't appear consecutively on the same qubit
        self.forbidden_pairs = {
            ('s', 'sdg'), ('sdg', 's'), ('t', 'tdg'), ('tdg', 't'),
            ('ry', 'ry'), ('rz', 'rz'),
            ('z', 'rz'), ('y', 'ry')
        }
        
        # Now initialize the base class, which will call create_ansatz()
        super().__init__(n_qubits)
    
    def gate_name_lower(self, gate):
        """Helper method to get lowercase gate name."""
        return gate.name.lower()
    
    def create_ansatz(self):
        """
        Generate a random parameterized quantum circuit with exactly 'depth' layers.
        Each layer contains a random number of single-qubit and two-qubit gates.
        """
        if self.seed is not None:
            random.seed(self.seed)

        # Create a new quantum circuit
        qc = QuantumCircuit(self.n_qubits)
        
        # We'll create a single ParameterVector that is big enough
        # for potential RY/RZ usage. We'll rename them as needed when binding.
        num_parameters = 2 * self.n_qubits * self.depth
        param_vec = ParameterVector("theta", length=num_parameters)
        param_idx = 0
        
        # Keep track of the last single-qubit gate on each wire
        last_single_gate = [None] * self.n_qubits
        
        for layer_index in range(self.depth):
            # Randomly pick how many single- and two-qubit gates for this layer
            while True:
                n_sq = random.randint(0, self.max_single_qubit_gates)
                n_tq = random.randint(0, self.max_two_qubit_gates)
                if n_sq + 2*n_tq <= self.n_qubits:
                    break
    
            # Shuffle qubits so gate placements are random
            available_qubits = list(range(self.n_qubits))
            random.shuffle(available_qubits)
    
            # Single-qubit gates
            for _ in range(n_sq):
                if not available_qubits:
                    break
                q = available_qubits.pop()
                while True:
                    candidate_gate = random.choice(self.single_qubit_pool).copy()
                    cand_name = self.gate_name_lower(candidate_gate)
    
                    prev_gate_name = last_single_gate[q]
                    if prev_gate_name is None or (prev_gate_name, cand_name) not in self.forbidden_pairs:
                        if cand_name in {"ry", "rz"}:
                            candidate_gate.params = [param_vec[param_idx]]
                            param_idx += 1
    
                        qc.append(candidate_gate, [q])
                        last_single_gate[q] = cand_name
                        break
    
            # Two-qubit gates
            last_two_qubit_gates = {}
            for _ in range(n_tq):
                if len(available_qubits) < 2:
                    break
                q1 = available_qubits.pop()
                q2 = available_qubits.pop()
                gate_2q = random.choice(self.two_qubit_pool)
                q1, q2 = sorted([q1, q2])
    
                if (q1, q2) not in last_two_qubit_gates or last_two_qubit_gates[(q1, q2)] != self.gate_name_lower(gate_2q):
                    qc.append(gate_2q, [q1, q2])
                    last_two_qubit_gates[(q1, q2)] = self.gate_name_lower(gate_2q)
    
            # Barrier (just for layer separation)
            qc.barrier()
    
        # Remove barriers at the end (makes the final circuit simpler)
        qc_no_barriers = QuantumCircuit(self.n_qubits)
    
        for instruction in qc.data:
            instr = instruction.operation
            qargs = instruction.qubits
            cargs = instruction.clbits
            if not isinstance(instr, Barrier):
                qc_no_barriers.append(instr, qargs, cargs)
    
        return qc_no_barriers

    def get_depth(self):
        """Return the approximate depth of the circuit."""
        return self.depth
    
    def get_actual_depth(self):
        """Return the exact depth as calculated by Qiskit."""
        return self.ansatz.depth()
    
    def set_seed(self, seed):
        """Change the random seed and regenerate the circuit."""
        self.seed = seed
        self.ansatz = self.create_ansatz()
        return self.ansatz


if __name__ == "__main__":
    n_qubits = 4
    depth = 2
    ansatz = RandomPQCAnsatz(n_qubits, depth)
    print(f"{ansatz.get_name()}, n_qubits = {ansatz.get_num_qubits()}, depth = {ansatz.get_depth()}, actual depth = {ansatz.get_actual_depth()}, with {ansatz.get_num_params()} parameters")
    ansatz.draw()