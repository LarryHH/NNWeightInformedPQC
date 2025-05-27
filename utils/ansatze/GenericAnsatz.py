from abc import ABC, abstractmethod

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