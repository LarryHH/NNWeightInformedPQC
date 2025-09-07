from utils.ansatze.HierarchicalGatewise import HierarchicalGatewiseAnsatz
from utils.ansatze.HierarchicalLayerwise import HierarchicalLayerwiseAnsatz

from .expressibility_proxy import calculate_expressibility_proxy
from .path_proxy import calculate_path_proxy

from collections import Counter

def run_tf_qas(ansatze_class, ansatz_args, S, R, K):
    """
    Orchestrates the full two-stage Training-Free QAS algorithm.

    Args:
        ansatze_class: The class to use for generating circuits (e.g., HierarchicalGatewiseAnsatz).
        ansatz_args: A dictionary of arguments for the ansatz class.
        S: The initial number of circuits to sample.
        R: The number of circuits to keep after the path-proxy filter.
        K: The final number of top circuits to return.

    Returns:
        A list of the top K circuits, ranked by expressibility.
    """
    print(f"--- Starting TF-QAS ---")
    print(f"Initial circuits (S): {S}, Path-proxy filtered (R): {R}, Final circuits (K): {K}")

    # --- Step 1: Randomly sample S circuits from the search space ---
    # Each circuit is stored as a dictionary to keep its data together.
    initial_circuits = []
    for i in range(S):
        ansatz = ansatze_class(**ansatz_args, seed=i)
        initial_circuits.append({"seed": i, "ansatz": ansatz})
    print(f"\nStep 1: Generated {len(initial_circuits)} initial circuits.")

    # --- Step 2 & 3: Filter down to R circuits using the path-based proxy ---
    for circuit_data in initial_circuits:
        path_count = calculate_path_proxy(circuit_data["ansatz"].get_ansatz())
        circuit_data["path_count"] = path_count
    
    # Sort by path_count (descending, as larger is better)
    sorted_by_path = sorted(initial_circuits, key=lambda x: x["path_count"], reverse=True)
    
    top_R_circuits = sorted_by_path[:R]
    print(f"Step 2: Filtered down to the top {len(top_R_circuits)} circuits based on path count.")
    print(f"Best path counts: {[c['path_count'] for c in top_R_circuits[:5]]}...")

    # --- Step 4 & 5: Rank the R circuits by expressibility and return the top K ---
    for circuit_data in top_R_circuits:
        expressibility = calculate_expressibility_proxy(circuit_data["ansatz"].get_ansatz(), n_samples=200) # Using 200 for speed
        circuit_data["expressibility"] = expressibility

    # Sort by expressibility (descending, as a score closer to 0 is better)
    sorted_by_expressibility = sorted(top_R_circuits, key=lambda x: x["expressibility"], reverse=True)

    top_K_circuits = sorted_by_expressibility[:K]
    print(f"\nStep 3: Ranked by expressibility and selected the final {len(top_K_circuits)} circuits.")
    print(f"Best expressibility scores: {[c['expressibility'] for c in top_K_circuits]}")

    return top_K_circuits

def main(mode='gate'):
    S = 1000  # Initial sample size (Paper uses 50000)
    R = 100   # Number to keep after path filter (Paper uses 5000)
    K = 10    # Final number of top circuits to return (Paper uses ~100)

    ansatz_args = {"n_qubits": 4}
    if mode == 'gate':
        ansatz_args['n_gates'] = 20
        run_tf_qas(HierarchicalGatewiseAnsatz, ansatz_args, S, R, K)
    elif mode == 'layer':
        ansatz_args['depth'] = 4
        run_tf_qas(HierarchicalLayerwiseAnsatz, ansatz_args, S, R, K)
    else:
        raise ValueError("Mode must be either 'gate' or 'layer'")

if __name__ == "__main__":
    main(mode='gate')