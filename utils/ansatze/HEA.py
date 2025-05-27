try:
    from .GenericAnsatz import GenericAnsatz
except ImportError:
    from GenericAnsatz import GenericAnsatz

from qiskit.circuit.library import TwoLocal

class HEA(GenericAnsatz):
    """
    Hardware Efficient Ansatz implementation.
    
    Uses the TwoLocal circuit from Qiskit with customizable rotation gates,
    entanglement pattern, and circuit depth.
    """
    def __init__(self, n_qubits, depth, rotation_blocks=None, entanglement_blocks='cx', entanglement='full'):
        self.depth = depth
        self.rotation_blocks = rotation_blocks or ['ry', 'rz']
        self.entanglement_blocks = entanglement_blocks
        self.entanglement = entanglement
        super().__init__(n_qubits)
    
    def create_ansatz(self):
        return TwoLocal(
            self.n_qubits,
            rotation_blocks=self.rotation_blocks,
            entanglement_blocks=self.entanglement_blocks,
            entanglement=self.entanglement,
            reps=self.depth
        )

    def get_depth(self):
        """Return the depth of the ansatz."""
        return self.depth

if __name__ == "__main__":
    n_qubits = 4
    depth = 2
    ansatz = HEA(n_qubits, depth)
    print(f"{ansatz.get_name()}, n_qubits = {ansatz.get_num_qubits()}, depth = {ansatz.depth}, with {ansatz.get_num_params()} parameters")
    ansatz.draw()