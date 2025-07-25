from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
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