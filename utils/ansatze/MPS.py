try:
    from .GenericAnsatz import GenericAnsatz
except ImportError:
    from GenericAnsatz import GenericAnsatz

from qiskit import QuantumCircuit
from qiskit.circuit.library import RealAmplitudes


def MPS(num_qubits: int, **kwargs) -> QuantumCircuit:
    """
    Constructs a Matrix Product State (MPS) quantum circuit using two-qubit blocks.

    Args:
        num_qubits (int): The number of qubits in the circuit.
        **kwargs: Keyword args forwarded to RealAmplitudes (e.g., reps, entanglement).

    Returns:
        QuantumCircuit: The constructed MPS circuit.
    """
    qc = QuantumCircuit(num_qubits)
    qubits = list(range(num_qubits))

    # Apply two-qubit RealAmplitudes blocks along the chain
    for i, j in zip(qubits[:-1], qubits[1:]):
        two_qubit_block = RealAmplitudes(
            num_qubits=2,
            parameter_prefix=f"θ_{i}_{j}",
            **kwargs
        )
        qc.compose(two_qubit_block, [i, j], inplace=True)
        qc.barrier()

    return qc


class MPSAnsatz(GenericAnsatz):
    """
    Matrix Product State (MPS) Ansatz implementation.

    Uses sequential two-qubit RealAmplitudes blocks to form an MPS structure.
    Accepts the same parameters as RealAmplitudes (reps, entanglement, etc.)
    """
    def __init__(self, n_qubits: int, **real_amplitude_kwargs):
        self.real_amplitude_kwargs = real_amplitude_kwargs
        super().__init__(n_qubits)

    def create_ansatz(self) -> QuantumCircuit:
        return MPS(self.n_qubits, **self.real_amplitude_kwargs)

if __name__ == "__main__":
    n_qubits = 4
    ansatz = MPSAnsatz(n_qubits, reps=1, entanglement='linear') # reps = bond dimension
    print(f"{ansatz.get_name()}, n_qubits = {ansatz.get_num_qubits()}, with {ansatz.get_num_params()} parameters")
    ansatz.draw()
    ansatz.draw_to_img()