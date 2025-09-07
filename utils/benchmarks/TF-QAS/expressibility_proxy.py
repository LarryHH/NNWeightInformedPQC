import qiskit
import numpy as np
from scipy.special import kl_div
from qiskit.quantum_info import Statevector

def calculate_expressibility_proxy(circuit: qiskit.QuantumCircuit, n_samples: int = 2000) -> float:
    """
    Calculates a faithful measure of expressibility for a PQC.

    This function implements the methodology described in the paper:
    E(C) = -D_KL(P(C, F) || P_Haar(F)).

    Args:
        circuit: The parameterized Qiskit QuantumCircuit. The circuit should
                 not contain measurements.
        n_samples: The number of state fidelities to sample to approximate P(C, F).

    Returns:
        The expressibility score. A larger value (closer to 0) indicates
        higher expressibility.
    """
    
    n_qubits = circuit.num_qubits
    params = circuit.parameters
    num_params = len(params)
    
    # 1. Approximate P(C, F) by sampling state fidelities
    fidelities = []
    for _ in range(n_samples):
        # Generate two random parameter sets
        param_vals1 = np.random.uniform(0, 2 * np.pi, num_params)
        param_vals2 = np.random.uniform(0, 2 * np.pi, num_params)
        
        param_dict1 = dict(zip(params, param_vals1))
        param_dict2 = dict(zip(params, param_vals2))

        # Simulate from the |0...0> state to get two statevectors
        state1 = Statevector.from_instruction(circuit.assign_parameters(param_dict1))
        state2 = Statevector.from_instruction(circuit.assign_parameters(param_dict2))
        
        # Calculate fidelity: F = |<ψθ|ψφ>|^2
        fidelity = np.abs(state1.inner(state2))**2
        fidelities.append(fidelity)
        
    # 2. Create the probability distributions as histograms
    n_bins = 75 # A reasonable number of bins for the histogram
    bins = np.linspace(0, 1, n_bins + 1)
    
    # Create the circuit's estimated fidelity distribution
    p_pqc, _ = np.histogram(fidelities, bins=bins, density=True)
    p_pqc = p_pqc / np.sum(p_pqc)
    p_pqc += 1e-12 # Add epsilon to avoid log(0) errors

    
    # 3. Calculate the ideal Haar distribution for each bin
    N = 2**n_qubits # Hilbert space dimension
    
    # To get the true probability for each bin, we integrate the Haar PDF
    # over the bin's width. The integral of (N-1)(1-F)^(N-2) is -(1-F)^(N-1).
    bin_edges_right = bins[1:]
    bin_edges_left = bins[:-1]
    
    p_haar = -( (1 - bin_edges_right)**(N-1) - (1 - bin_edges_left)**(N-1) )
    p_haar += 1e-10

    # 4. Calculate the final score using KL Divergence
    # D_KL(P || Q) = sum(P * log(P/Q))
    kl_divergence = np.sum(kl_div(p_pqc, p_haar))
    
    # Expressibility is defined as the negative KL divergence
    expressibility = -kl_divergence
    
    return expressibility



if __name__ == "__main__":
    from qiskit.circuit.library import EfficientSU2, RYGate
    n_qubits = 4
    
    # --- 1. A HIGHLY EXPRESSIVE CIRCUIT ---
    # We use EfficientSU2 with high repetition (depth) to ensure it can
    # thoroughly explore the Hilbert space.
    
    deep_pqc = EfficientSU2(n_qubits, reps=10, entanglement='full')
    print(f"--- Sanity Check 1: High Expressibility Circuit ---")
    print(f"Calculating expressibility for a deep ({deep_pqc.reps} reps) EfficientSU2 circuit...")
    
    high_expr_score = calculate_expressibility_proxy(deep_pqc)
    
    print(f"\nExpressibility Score: {high_expr_score:.4f}")
    print("Expected result: A score very close to 0.")
    
    print("\n" + "="*50 + "\n")
    
    # --- 2. A VERY INEXPRESSIVE CIRCUIT ---
    # This circuit has only single-qubit rotations and NO entanglement.
    # Without entanglement, it cannot generate all possible states.
    
    simple_pqc = qiskit.QuantumCircuit(n_qubits)
    for i in range(n_qubits):
        # Using RYGate which is a parameterized gate
        simple_pqc.ry(qiskit.circuit.Parameter(f'θ_{i}'), i)

    print(f"--- Sanity Check 2: Low Expressibility Circuit ---")
    print(f"Calculating expressibility for a simple, non-entangling circuit...")
    
    low_expr_score = calculate_expressibility_proxy(simple_pqc)
    
    print(f"\nExpressibility Score: {low_expr_score:.4f}")
    print("Expected result: A large negative number.")
