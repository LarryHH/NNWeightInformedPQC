from GenericAnsatz import GenericAnsatz
from qiskit.circuit.library import RealAmplitudes

class RealAmplitudeAnsatz(GenericAnsatz):
    """
    Real Amplitude Ansatz implementation.
    
    Uses the RealAmplitudes circuit from Qiskit with customizable entanglement pattern and circuit depth.
    """
    def __init__(self, n_qubits, depth, entanglement='reverse_linear'):
        self.depth = depth
        self.entanglement = entanglement
        super().__init__(n_qubits)
    
    def create_ansatz(self):
        return RealAmplitudes(
            self.n_qubits,
            reps=self.depth,
            entanglement=self.entanglement
        )
    

if __name__ == "__main__":
    n_qubits = 4
    depth = 2
    ansatz = RealAmplitudeAnsatz(n_qubits, depth)
    print(f"{ansatz.get_name()}, n_qubits = {ansatz.get_num_qubits()}, depth = {ansatz.depth}, with {ansatz.get_num_params()} parameters")
    ansatz.draw()