try:
    from .GenericAnsatz import GenericAnsatz
except ImportError:
    from GenericAnsatz import GenericAnsatz
    
import numpy as np
import re
import random
import qiskit
from qiskit.circuit import QuantumCircuit, ParameterVector
from qiskit.circuit.library import (
    RXGate, RYGate, RZGate,
    XGate, YGate, ZGate,
    HGate, SGate, TGate,
    SdgGate, TdgGate
)

class SpecifiedPQCAnsatz(GenericAnsatz):
    """
    Specified Parameterized Quantum Circuit (PQC) ansatz.

    Creates a quantum circuit based on a matrix specification where:
    - Each row represents a qubit
    - Each column represents a time step or layer
    - Each cell contains a gate specification as a string

    Gate specifications:
    1. Single-qubit gates:
    - Use the gate name directly: 'h', 'x', 'y', 'z', 's', 't', 'sdg', 'tdg'
    - For parameterized gates: 'rx', 'ry', 'rz'
    - For identity/no operation: 'i', 'id', or '' (empty string)

    2. Two-qubit gates:
    - Format: '{gate_name}_{index}{direction}'
    - gate_name: 'cx', 'cz', 'swap', 'cnot'
    - index: A number to identify which gate this is within the layer (e.g., '1', '2')
    - direction: 'c' for control qubit, 't' for target qubit
    - Example: 'cx_1c' and 'cx_1t' form a CNOT gate where '1c' is the control and '1t' is the target
    - Multiple two-qubit gates in the same layer should use different indices

    Example circuit_spec:
    [
        ['h', 'cx_1t', 'rx', 'cz_1c'],  # Qubit 0
        ['x', 'ry', 'cx_2c', 'cz_1t'],  # Qubit 1
        ['ry', 'cx_1c', 'cx_2t', 'rz']  # Qubit 2
    ]

    This creates a 3-qubit circuit with 4 time steps, including both single-qubit gates
    and properly connected two-qubit gates.
    """
    
    # Define gate mappings - maps string identifiers to gate constructors
    GATE_MAP = {
        # Parameterized rotation gates
        'rx': lambda param: RXGate(param),
        'ry': lambda param: RYGate(param),
        'rz': lambda param: RZGate(param),
        
        # Fixed gates (no parameters)
        'x': lambda _: XGate(),
        'y': lambda _: YGate(),
        'z': lambda _: ZGate(),
        'h': lambda _: HGate(),
        's': lambda _: SGate(),
        't': lambda _: TGate(),
        'sdg': lambda _: SdgGate(),
        'tdg': lambda _: TdgGate(),

        # Two-qubit gates - these will be handled separately
        'cx': lambda _: qiskit.circuit.library.CXGate(),
        'cz': lambda _: qiskit.circuit.library.CZGate(),
        'swap': lambda _: qiskit.circuit.library.SwapGate(),
        'cnot': lambda _: qiskit.circuit.library.CXGate(),
        
        # Identity/None - no gate applied
        'i': None,
        '': None,
        'id': None,
        None: None
    }
    
    # Define which gates require parameters
    PARAMETERIZED_GATES = {'rx', 'ry', 'rz'}
    
    # Regular expression to match two-qubit gate format
    TWO_QUBIT_PATTERN = re.compile(r'^(cx|cz|swap|cnot)_(\d+)(c|t)$')
    
    def __init__(self, circuit_spec, param_prefix="θ"):
        """
        Initialize a specified parameterized quantum circuit ansatz.
        
        Args:
            circuit_spec: Matrix-like specification of the circuit
                          Each row corresponds to a qubit
                          Each column corresponds to a time step
                          Each cell contains a gate specification (string)
            param_prefix: Prefix for parameter names (default: "θ")
        """
        self.circuit_spec = np.array(circuit_spec)
        self.param_prefix = param_prefix
        
        # Extract dimensions
        self.n_qubits = self.circuit_spec.shape[0]
        self.depth = self.circuit_spec.shape[1]
        
        # Initialize the base class (which calls create_ansatz)
        super().__init__(self.n_qubits)

    def validate_circuit_spec(self):
        """
        Validate the circuit specification to ensure all two-qubit gates are properly paired.
        """
        for step in range(self.depth):
            # Dictionary to track two-qubit gates at this step
            two_qubit_gates = {}
            
            # Find all two-qubit gate specifications at this step
            for qubit in range(self.n_qubits):
                gate_spec = self.circuit_spec[qubit, step]
                if not isinstance(gate_spec, str):
                    continue
                    
                match = self.TWO_QUBIT_PATTERN.match(gate_spec.lower())
                if match:
                    gate_name, gate_idx, direction = match.groups()
                    key = f"{gate_name}_{gate_idx}"
                    
                    if key not in two_qubit_gates:
                        two_qubit_gates[key] = {}
                    
                    two_qubit_gates[key][direction] = qubit
            
            # Check that each two-qubit gate has both a control and target
            for gate_key, directions in two_qubit_gates.items():
                if 'c' not in directions or 't' not in directions:
                    raise ValueError(f"Two-qubit gate {gate_key} at step {step} is missing a control or target")
    
    def create_ansatz(self):
        """
        Create the quantum circuit based on the specification matrix.
        """
        # Validate the circuit specification
        self.validate_circuit_spec()
        
        # Create quantum circuit
        qc = QuantumCircuit(self.n_qubits)
        
        # Count parameters needed
        param_count = self._count_parameters_needed()
        
        # Create parameters
        if param_count > 0:
            params = ParameterVector(self.param_prefix, param_count)
            param_idx = 0
        else:
            params = []
            param_idx = 0
        
        # For each time step (column)
        for step in range(self.depth):
            # Collect two-qubit gate information for this step
            two_qubit_gates = {}
            
            for qubit in range(self.n_qubits):
                gate_spec = self.circuit_spec[qubit, step]
                if not isinstance(gate_spec, str) or not gate_spec:
                    continue
                    
                match = self.TWO_QUBIT_PATTERN.match(gate_spec.lower())
                if match:
                    gate_name, gate_idx, direction = match.groups()
                    key = f"{gate_name}_{gate_idx}"
                    
                    if key not in two_qubit_gates:
                        two_qubit_gates[key] = {'name': gate_name}
                    
                    two_qubit_gates[key][direction] = qubit
            
            # Apply single-qubit gates first
            for qubit in range(self.n_qubits):
                gate_spec = self.circuit_spec[qubit, step]
                
                # Skip if no gate, invalid gate, or two-qubit gate (handled separately)
                if (gate_spec is None or gate_spec == '' or gate_spec == 'i' or gate_spec == 'id' or 
                    (isinstance(gate_spec, str) and self.TWO_QUBIT_PATTERN.match(gate_spec.lower()))):
                    continue
                
                # Convert to lowercase for case-insensitive matching
                gate_name = gate_spec.lower() if isinstance(gate_spec, str) else gate_spec
                
                # Get the gate constructor
                gate_constructor = self.GATE_MAP.get(gate_name)
                
                if gate_constructor is None:
                    raise ValueError(f"Unknown gate: {gate_spec}")
                
                # Create the gate with a parameter if needed
                if gate_name in self.PARAMETERIZED_GATES:
                    gate = gate_constructor(params[param_idx])
                    param_idx += 1
                else:
                    gate = gate_constructor(None)
                
                # Add gate to circuit
                qc.append(gate, [qubit])
            
            # Now apply two-qubit gates
            for gate_key, gate_info in two_qubit_gates.items():
                gate_name = gate_info['name']
                control = gate_info.get('c')
                target = gate_info.get('t')
                
                # Skip if either control or target is missing
                if control is None or target is None:
                    continue
                
                # Get the gate constructor
                gate_constructor = self.GATE_MAP.get(gate_name)
                
                if gate_constructor is None:
                    raise ValueError(f"Unknown two-qubit gate: {gate_name}")
                
                # Create the gate
                gate = gate_constructor(None)
                
                # Add gate to circuit
                qc.append(gate, [control, target])
            
            # Add a barrier between time steps for clarity
            if step < self.depth - 1:
                qc.barrier()
                
        return qc
    
    def _count_parameters_needed(self):
        """
        Count how many parameters we need for the circuit.
        """
        count = 0
        for gate_spec in self.circuit_spec.flatten():
            if gate_spec is not None and isinstance(gate_spec, str):
                # Skip two-qubit gates when counting parameters
                if self.TWO_QUBIT_PATTERN.match(gate_spec.lower()):
                    continue
                    
                gate_name = gate_spec.lower()
                if gate_name in self.PARAMETERIZED_GATES:
                    count += 1
        return count
    
    def update_circuit_spec(self, new_spec):
        """
        Update the circuit specification and recreate the circuit.
        
        Args:
            new_spec: New matrix-like specification for the circuit
        """
        self.circuit_spec = np.array(new_spec)
        self.n_qubits = self.circuit_spec.shape[0]
        self.depth = self.circuit_spec.shape[1]
        self.ansatz = self.create_ansatz()
        return self.ansatz
    
    def get_circuit_spec(self):
        """
        Get the circuit specification matrix.
        """
        return self.circuit_spec
    
    def get_depth(self):
        """
        Get the depth (number of time steps) of the circuit.
        """
        return self.depth


if __name__ == "__main__":

    circuit_spec = [
        ['h', 'cx_1t', 'cx_2c', 'rz'],  
        ['ry', 'cx_1c', 'rz', 'h'],     
        ['rz', 'ry', 'cx_2t', 'rx'],
        ['rz', 'ry', 'rx', 'rx'],
        ['rz', 'ry', 'cx_3t', 'rx'],
        ['rz', 'ry', 'rx', 'rx'],
        ['rz', 'ry', 'cx_3c', 'rx'],
        ['rz', 'y', 'rx', 'rx']
    ]
    
    ansatz = SpecifiedPQCAnsatz(circuit_spec)
    print(f"{ansatz.get_name()}, n_qubits = {ansatz.get_num_qubits()}, depth = {ansatz.depth}, with {ansatz.get_num_params()} parameters")
    ansatz.draw()