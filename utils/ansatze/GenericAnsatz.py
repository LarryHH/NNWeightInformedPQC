from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
from qiskit.circuit import QuantumCircuit
from qiskit.circuit import ParameterVector
import numpy as np

def twolocal_nontranspiled(n_qubits, depth, rotation_blocks, entanglement_blocks, entanglement):
    """
    Builds the HEA circuit manually for performance.
    
    Note: This implementation currently handles 'linear' entanglement with 'cx' gates.
          It can be extended to support more options if needed.
    """
    num_params = (depth + 1) * n_qubits * len(rotation_blocks)
    theta = ParameterVector("θ", num_params)
    p_index = 0

    qc = QuantumCircuit(n_qubits)

    for rep in range(depth + 1):
        # Rotation layer
        for gate_name in rotation_blocks:
            for i in range(n_qubits):
                # getattr(qc, gate_name) dynamically calls qc.ry, qc.rz, etc.
                getattr(qc, gate_name.lower())(theta[p_index], i)
                p_index += 1

        # Entanglement layer (except for the last repetition)
        if rep < depth:
            if entanglement == 'linear':
                for i in range(n_qubits - 1):
                    # This assumes the entanglement block is a single gate like 'cx'
                    getattr(qc, entanglement_blocks.lower())(i, i + 1)
            else:
                # Placeholder for other entanglement patterns like 'full', 'circular'
                raise NotImplementedError(f"Entanglement type '{entanglement}' not implemented in unrolled version.")
            
    return qc

def zzfeaturemap_nontranspiled(n_qubits, reps=2):
    x = ParameterVector("x", n_qubits)
    qc = QuantumCircuit(n_qubits)
    qc.h(range(n_qubits))
    for _ in range(reps):
        # Layer of RZ encoding - ANGLE IS 2*x[i]
        for i in range(n_qubits):
            qc.rz(2 * x[i], i)

        # Full pairwise ZZ entanglement - ANGLE IS 2*(pi-x[i])*(pi-x[j])
        for i in range(n_qubits):
            for j in range(i + 1, n_qubits):
                # Using the CX-RZ-CX decomposition for Rzz
                qc.cx(i, j)
                qc.rz(2 * (np.pi - x[i]) * (np.pi - x[j]), j)
                qc.cx(i, j)

    return qc

class GenericAnsatz(ABC):
    """
    Abstract base class for quantum ansatz circuits.
    
    This class establishes a common interface for all ansatz implementations,
    defining standard methods for accessing and visualizing quantum circuits.
    """
    def __init__(self, n_qubits, ansatz_name=None):
        self.ansatz_name = ansatz_name if ansatz_name else self.__class__.__name__
        self.n_qubits = n_qubits
        self.ansatz = self.create_ansatz()
    
    @abstractmethod
    def create_ansatz(self):
        """
        Create and return the quantum circuit for this ansatz.
        Must be implemented by derived classes.
        """
        pass
    
    def get_ansatz(self):
        """Return the quantum circuit for this ansatz."""
        return self.ansatz
    
    def get_num_qubits(self):
        """Return the number of qubits in the ansatz circuit."""
        return self.n_qubits
    
    def get_name(self):
        """Return the name of the ansatz circuit."""
        return self.ansatz_name
    
    def get_params(self):
        """Return the parameters of the ansatz circuit."""
        return self.ansatz.parameters
    
    def get_num_params(self):
        """Return the number of parameters in the ansatz circuit."""
        return len(self.ansatz.parameters)
    
    def draw(self, **kwargs):
        """Draw the ansatz circuit with optional parameters."""
        kwargs.setdefault('fold', -1)  # Default to unfolded circuit
        print(self.ansatz.decompose().draw(**kwargs))

    def draw_to_img(self, filename=None):
        """Draw the ansatz circuit to an image file, or render to display."""
        circ = self.ansatz.decompose()
        original_global_phase = circ.global_phase

        # Temporarily set global_phase to 0 to prevent it from being drawn
        circ.global_phase = 0
        figure = circ.draw(output='mpl', filename=None)
        circ.global_phase = original_global_phase

        # figure.set_size_inches(14, 7)
        figure.tight_layout()
        if filename:
            figure.savefig(filename)
            print(f"Circuit saved to {filename}")
            plt.close(figure)
        else:
            plt.show()