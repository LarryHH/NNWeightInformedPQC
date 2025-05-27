try:
    from .GenericAnsatz import GenericAnsatz
except ImportError:
    from GenericAnsatz import GenericAnsatz

from qiskit import QuantumCircuit
from qiskit.circuit.library import RealAmplitudes


def _generate_tree_tuples(n: int):
    """
    Generate tuples representing TTN tree connectivity for n qubits.

    Args:
        n (int): Must be power of 2 and >0.
    Returns:
        List of lists of tuples, each tuple is a pair of qubit indices.
    """
    levels = []
    # initial pairing
    pairs = [(i, i+1) for i in range(0, n, 2)]
    levels.append(pairs)

    while len(pairs) > 1:
        new_pairs = []
        for a, b in zip(pairs[0::2], pairs[1::2]):
            new_pairs.append((a[1], b[1]))
        levels.append(new_pairs)
        pairs = new_pairs

    return levels

def TTN(num_qubits: int, **kwargs) -> QuantumCircuit:
    """
    Constructs a Tree Tensor Network (TTN) quantum circuit.

    Args:
        num_qubits (int): Number of qubits (power of 2).
        **kwargs: Keyword args forwarded to RealAmplitudes.

    Returns:
        QuantumCircuit: The constructed TTN circuit.
    """

    # Compute qubit indices
    assert num_qubits & (
        num_qubits -
        1) == 0 and num_qubits != 0, "Number of qubits must be a power of 2"

    qc = QuantumCircuit(num_qubits)
    levels = _generate_tree_tuples(num_qubits)

    for layer in levels:
        for i, j in layer:
            block = RealAmplitudes(
                num_qubits=2,
                parameter_prefix=f"θ_{i}_{j}",
                **kwargs
            )
            qc.compose(block, [i, j], inplace=True)
        qc.barrier()

    return qc


class TTNAnsatz(GenericAnsatz):
    """
    Tree Tensor Network (TTN) Ansatz implementation.

    Uses hierarchical two-qubit RealAmplitudes blocks following a binary tree.
    Accepts the same parameters as RealAmplitudes (reps, entanglement, etc.)
    """
    def __init__(self, n_qubits: int, **real_amplitude_kwargs):
        self.real_amplitude_kwargs = real_amplitude_kwargs
        super().__init__(n_qubits)

    def create_ansatz(self) -> QuantumCircuit:
        return TTN(self.n_qubits, **self.real_amplitude_kwargs)


if __name__ == "__main__":
    n_qubits = 4
    ansatz = TTNAnsatz(n_qubits, reps=1, entanglement='linear') # reps = bond dimension
    print(f"{ansatz.get_name()}, n_qubits = {ansatz.get_num_qubits()}, with {ansatz.get_num_params()} parameters")
    ansatz.draw()