import inspect, os, json, time
from pathlib import Path
from typing import Dict, Sequence, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt

from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Operator

from utils.nn.QuantumNN import QuantumNN 
from utils.nn.ClassicalNN import FlexibleNN
from utils.ansatze.GenericAnsatz import GenericAnsatz

# ──────────────────────────────────────────────────────────────────────
# Barren Plateaus: gradient-variance
# ──────────────────────────────────────────────────────────────────────
def gradient_variance(qnn: QuantumNN, n_qubits: int, *,
                      n_samples: int = 50, seed: int = 42,
                      param_init_fn = lambda p: p.data.uniform_(0, 2 * np.pi),
                      tracked_grad_scalar_idx: int = 0                         
                     ) -> float:
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    x = torch.rand((1, n_qubits), dtype=torch.float32) 
    y = torch.tensor([0], dtype=torch.long) 

    trainable_params = [p for p in qnn.parameters() if p.requires_grad]

    if not trainable_params:
        print(f"Warning: No trainable parameters found for QNN (n_qubits={n_qubits}). Returning float('nan').")
        return float('nan')

    total_scalar_params = sum(p.numel() for p in trainable_params)
    if total_scalar_params == 0:
        print(f"Warning: Zero scalar trainable parameters found for QNN (n_qubits={n_qubits}). Returning float('nan').")
        return float('nan')

    if not (0 <= tracked_grad_scalar_idx < total_scalar_params):
        print(f"Warning: tracked_grad_scalar_idx ({tracked_grad_scalar_idx}) is out of bounds for "
              f"{total_scalar_params} scalar trainable parameters in QNN (n_qubits={n_qubits}). "
              f"Returning float('nan').")
        return float('nan')

    single_grad_component_samples = np.zeros(n_samples, dtype=float)

    for i_sample in range(n_samples):
        for p in trainable_params:
            if p.is_leaf: # Ensure we only initialize leaf tensors that are parameters
                param_init_fn(p)
            else:
                # This case should be rare for typical nn.Parameters
                print(f"Warning: Skipped initialization for non-leaf tensor during QNN (n_qubits={n_qubits}) parameter init.")


        qnn.zero_grad(set_to_none=True)
        
        output = qnn(x)
        loss = qnn.criterion(output, y)

        if not loss.requires_grad and total_scalar_params > 0:
            print(f"Warning: Loss does not require grad for QNN (n_qubits={n_qubits}) "
                  f"in sample {i_sample}, but trainable parameters exist. Returning float('nan').")
            return float('nan')

        loss.backward()

        grad_tensor_list = []
        for p in trainable_params:
            if p.grad is not None:
                grad_tensor_list.append(p.grad.view(-1))
            # else:
                # If p.grad is None for a trainable parameter, it implies it didn't contribute to the loss,
                # or was part of a non-differentiable operation or detached.
                # This will be caught if the full_grad_vector is too short or grad_tensor_list is empty.

        if not grad_tensor_list:
            print(f"Warning: No non-None gradients computed for any trainable parameter in QNN "
                  f"(n_qubits={n_qubits}) in sample {i_sample}. Returning float('nan').")
            return float('nan')

        full_grad_vector = torch.cat(grad_tensor_list)

        if full_grad_vector.numel() != total_scalar_params:
            print(f"Warning: Mismatch in expected total scalar parameters ({total_scalar_params}) "
                  f"and actual collected gradient elements ({full_grad_vector.numel()}) "
                  f"for QNN (n_qubits={n_qubits}) in sample {i_sample}. Returning float('nan').")
            return float('nan')
            
        single_grad_component_samples[i_sample] = full_grad_vector.cpu().numpy()[tracked_grad_scalar_idx]

    return single_grad_component_samples.var(ddof=1)

def plot_barren_plateau_curve(xs: Sequence[int], ys: Sequence[float], *, title: str, save_to: Path, tracked_grad_scalar_idx: int):
    plt.figure(figsize=(8,4))
    
    plt.semilogy(xs, ys, "o-", label="measured")
    
    # Prepare data for reference line by filtering NaNs
    valid_indices = [i for i, y_val in enumerate(ys) if not np.isnan(y_val)]
    
    if valid_indices:
        valid_xs_for_ref = np.array(xs)[valid_indices]
        valid_ys_for_ref = np.array(ys)[valid_indices]
        
        if len(valid_ys_for_ref) > 0: # Ensure there's at least one valid point
            ref_start_val = valid_ys_for_ref[0]
            ref_start_qubit = valid_xs_for_ref[0]
            ref = ref_start_val * 2.0 ** (-(np.array(valid_xs_for_ref) - ref_start_qubit))
            plt.semilogy(valid_xs_for_ref, ref, "--", label=r"$\propto 2^{-n}$") 
    
    plt.xlabel("qubits")
    plt.ylabel(rf"Var($\partial C / \partial \theta_{{{tracked_grad_scalar_idx}}}$)") # Updated label
    plt.title(title)
    plt.grid(True, which="both", ls="--")
    plt.legend() 
    plt.tight_layout()
    save_to.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_to, dpi=300, bbox_inches="tight")
    print(f"  Plot saved to {save_to}")
    plt.close()

# ──────────────────────────────────────────────────────────────────────
# Expressibility is the extent to which the PQC and uniformly cover the Haar distribution (i.e., the set of all possible quantum states)
# in general, a higher expressibility can cause barren plateaus:
#    - can explore a larger space of quantum states. 
#    - larger space means the cost function's gradients tend to average out to zero almost everywhere
# Trainability is the property of an ansatz that allows its parameters to be successfully optimized to find a solution, characterized by its feature landscape the gradients of the cost function with respect to the parameters
#    - essentially, trainability is the abscene of barren plateaus
# Importantly, expressibility is not the same as trainability:
#    - a circuit need not be expressible to adeuqately cover the solution space of a problem; i.e. trainability is only concerned with the congruence of the ansatz with the problem at hand
# ──────────────────────────────────────────────────────────────────────


def expressibility(pqc: QuantumCircuit, *, ensemble_size: int = 100, seed: int = 42) -> float:
    """
    Computes the expressibility of a Parametrized Quantum Circuit (PQC).

    Expressibility measures how uniformly the ansatz explores the unitary space.
    This is calculated using the state-dependent frame potential, comparing the
    ansatz's distribution to the uniform Haar distribution. A value close to 1.0
    indicates high expressibility.

    Args:
        pqc (QuantumCircuit): The Parametrized Quantum Circuit (ansatz) to analyze.
                              It must contain qiskit.circuit.Parameter objects.
        ensemble_size (int): The number of random parameter sets to sample. A larger
                             size gives a more accurate result but takes longer.
        seed (int): A seed for the random number generator for reproducibility.

    Returns:
        float: The computed expressibility. Returns float('nan') if PQC has no parameters.
    """
    np.random.seed(seed)
    
    # 1. Get circuit parameters
    params = sorted(pqc.parameters, key=lambda p: p.name)
    num_params = len(params)
    num_qubits = pqc.num_qubits
    d = 2**num_qubits

    if num_params == 0:
        # This warning is helpful for debugging configurations
        print(f"Warning: The provided circuit for expressibility calculation (n_qubits={num_qubits}) has no parameters.")
        return float('nan')

    # 2. Generate an ensemble of unitaries from the PQC
    unitary_ensemble = []
    for _ in range(ensemble_size):
        param_values = np.random.uniform(0, 2 * np.pi, num_params)
        param_binding = dict(zip(params, param_values))
        bound_circuit = pqc.assign_parameters(param_binding)
        unitary = Operator(bound_circuit).data
        unitary_ensemble.append(unitary)

    fidelities = np.array([np.abs(u[0, 0])**2 for u in unitary_ensemble])

    # 3. Calculate the frame potential for the ansatz (F_ansatz)
    f_ansatz = 0.0
    for i in range(ensemble_size):
        u = unitary_ensemble[i]
        for j in range(ensemble_size):
            v_dag = unitary_ensemble[j].conj().T
            # For the probe state |0><0|, Tr(X U V_dag) is just the top-left
            # element of the matrix U V_dag, which is <0|U V_dag|0>.
            trace_val = (u @ v_dag)[0, 0]
            f_ansatz += np.abs(trace_val)**4

    f_ansatz /= (ensemble_size**2)

    # 4. Calculate the frame potential for the Haar distribution (F_Haar)
    # For a pure probe state like |0><0|, this simplifies to F_Haar = 2 / (d*(d+1))
    f_haar = 2 / (d * (d + 1))
    
    # 5. Compute and return expressibility
    if f_haar == 0:
        return float('inf') # Avoid division by zero, though unlikely
    return f_ansatz / f_haar, fidelities

def plot_expressibility_histogram(fidelities: np.ndarray, n_qubits: int, *, title: str, save_to: Path):
    """
    Plots a histogram of fidelities to visualize expressibility.

    Args:
        fidelities (np.ndarray): Array of fidelity values |<ψ|U|ψ>|^2.
        n_qubits (int): Number of qubits.
        title (str): Title for the plot.
        save_to (Path): Path to save the plot image.
    """
    plt.figure(figsize=(8, 4))
    
    # Plot histogram of the ansatz's fidelities
    if np.ptp(fidelities) < 1e-9:
        plt.axvline(fidelities[0], color='blue', linestyle='-', label="Ansatz Distribution (single value)")
        plt.ylim(0, 1) 
    else:
        plt.hist(fidelities, bins=50, density=True, label="Ansatz Distribution", alpha=0.7, range=(0,1))
    
    # Plot theoretical Haar distribution P_Haar(F) = (d-1)(1-F)^(d-2)
    d = 2**n_qubits
    f_vals = np.linspace(0, 1, 200)
    p_haar = (d - 1) * (1 - f_vals)**(d - 2)
    plt.plot(f_vals, p_haar, "r--", label=f"Haar Distribution (d={d})")

    plt.xlabel(r"Fidelity F = $|\langle\psi|U|\psi\rangle|^2$")

    plt.ylabel("Probability Density (log)")
    plt.yscale('log')

    plt.title(f"Expressibility: {title}")
    plt.legend()
    plt.grid(True, which="both", ls="--")
    # if d < 1024: # Limit y-axis for small d to avoid clutter
    #     max_p_haar = (d - 1)
    #     plt.ylim(0, max(1, max_p_haar * 1.1))
    plt.tight_layout()
    save_to.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_to, dpi=300, bbox_inches="tight")
    print(f"  Expressibility plot saved to {save_to}")
    plt.close()


# ──────────────────────────────────────────────────────────────────────
# utilities
# ──────────────────────────────────────────────────────────────────────
def _needs(arg: str, fn) -> bool:
    return arg in inspect.signature(fn).parameters

def create_classical_model(n_qubits: int, depth: int) -> FlexibleNN:
    return FlexibleNN(
        input_dim=n_qubits,
        hidden_dims=[n_qubits] * depth,
        output_dim=2, 
        act="relu",
        condition_number=0.0, 
        scale_on_export=False,
    )

def build_ansatz(cfg: Dict, *, n_qubits: int, depth: int, seed: int) -> Tuple[GenericAnsatz, Dict]:
    cls, init_args = cfg["_class"], dict(cfg.get("init_args", {}))
    if _needs("n_qubits", cls) and "n_qubits" not in init_args:
        init_args["n_qubits"] = n_qubits
    if _needs("depth", cls) and "depth" not in init_args:
        init_args["depth"] = depth
    if _needs("seed", cls) and "seed" not in init_args:
        init_args["seed"] = seed

    if "classical_model" in init_args:
        init_args["classical_model"] = create_classical_model(n_qubits, depth)

    ansatz_obj = cls(**init_args)

    qnn_kwargs = cfg.get("qnn_extra_args", {}).copy()

    if "qnn_extra_args_builder" in cfg:
        builder_func = cfg["qnn_extra_args_builder"]
        if callable(builder_func):
            qnn_kwargs.update(builder_func(ansatz_obj))
        else:
            print(f"Warning: 'qnn_extra_args_builder' in cfg for '{cfg.get('name', 'UnknownAnsatz')}' is not a function.")
            
    qnn_kwargs.setdefault("seed", seed)
    return ansatz_obj, qnn_kwargs


# ──────────────────────────────────────────────────────────────────────
# main sweep
# ──────────────────────────────────────────────────────────────────────
def sweep(cfg: Dict, n_qubits_seq: Sequence[int], depths_seq: Sequence[int], *, # Renamed for clarity
          samples: int = 50, seed: int = 42,
          param_init_fn = lambda p: p.data.uniform_(0, 2 * np.pi), # Default to uniform
          tracked_grad_scalar_idx: int = 0                         # Default to 0-th component
         ):
    results_barren_plateau: Dict[Tuple[int,int], float] = {} # (n_qubits, depth) -> variance
    results_expressibility: Dict[Tuple[int,int], float] = {} # (n_qubits, depth) -> expressibility
    
    print(f"\n--- Sweeping for Ansatz: {cfg['name']} ---")
    print(f"Parameter initialization: {'Uniform [0, 2pi)' if param_init_fn.__code__.co_code == (lambda p: p.data.uniform_(0,2*np.pi)).__code__.co_code else 'Custom'}")
    print(f"Tracked gradient component index: {tracked_grad_scalar_idx}")

    for d in depths_seq:
        current_depth_results_q = []
        current_depth_results_v = []

        # fidelities_for_plotting = None
        # nq_for_plotting = 0

        print(f"  Depth: {d}")
        for nq in n_qubits_seq:

            if 'TTN' in cfg['name'] and nq not in [2**i for i in range(1, 6)]:
                print(f"    Skipping {cfg['name']} for n_qubits={nq} (not a power of 2)")
                continue

            start_time = time.time()
            ansatz_obj, qnn_kwargs = build_ansatz(cfg, n_qubits=nq, depth=d, seed=seed)

            randompqc_seed = 0
            while True: # TODO: how to incorporate max retries?
                if cfg['name'] == 'RandomPQC' and ansatz_obj.get_num_params() == 0:
                    randompqc_seed += 1
                    print(f"    Retrying {cfg['name']} (n_qubits={nq}, depth={d}) due to zero parameters. Retry number: {randompqc_seed}")
                    ansatz_obj, qnn_kwargs = build_ansatz(cfg, n_qubits=nq, depth=d, seed=seed + randompqc_seed)
                else:
                    break

            qnn = QuantumNN(ansatz_obj.get_ansatz(), n_qubits=nq, num_classes=2, **qnn_kwargs)
            
            # BARREN PLATEAU ANALYSIS
            var = gradient_variance(qnn, nq, n_samples=samples, seed=seed,
                                    param_init_fn=param_init_fn,
                                    tracked_grad_scalar_idx=tracked_grad_scalar_idx)
            # EXPRESSIBILITY ANALYSIS
            expr, fidelities = expressibility(ansatz_obj.get_ansatz(), ensemble_size=samples, seed=seed)
            
            elapsed_time = time.time() - start_time
            print(f"    {cfg['name']:<18}  n={nq:2d}  depth={d:2d}  var={var:.3e}  expr={expr:.4f}  time={elapsed_time:.2f}s")
            
            results_barren_plateau[(nq,d)] = var
            current_depth_results_q.append(nq)
            current_depth_results_v.append(var)

            results_expressibility[(nq,d)] = expr
            plot_title_expr = f"{cfg['name']} (n={nq}, depth={d})"
            out_filename_base = cfg['name'].replace(' ','_').replace('(','').replace(')','').replace(',','_')
            out_path_expr = Path("results/analysis/expressibility/") / f"{out_filename_base}_{nq}q_depth{d}.png"
            plot_expressibility_histogram(fidelities, nq, title=plot_title_expr, save_to=out_path_expr)

        if current_depth_results_q:
            plot_title = f"{cfg['name']} (depth={d})"
            out_filename_base = cfg['name'].replace(' ','_').replace('(','').replace(')','').replace(',','_')
            out_path = Path("results/analysis/barren_plateaus/") / f"{out_filename_base}_depth{d}.png"
            
            plot_barren_plateau_curve(current_depth_results_q, current_depth_results_v,
                       title=plot_title, save_to=out_path,
                       tracked_grad_scalar_idx=tracked_grad_scalar_idx)

        else:
            print(f"    Skipping plot for {cfg['name']} (depth={d}) as no qubit configurations were processed or yielded data.")

    out_json_filename_base = cfg['name'].replace(' ','_').replace('(','').replace(')','').replace(',','_')

    out_json = Path("results/analysis/barren_plateaus") / f"{out_json_filename_base}_data.json"
    json_results_bp = {f"{nq}_{d}": (v if not np.isnan(v) else None) for (nq,d),v in results_barren_plateau.items()}
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(json_results_bp, indent=2))
    print(f"  BP Results saved to {out_json}")

    out_json_expr = Path("results/analysis/expressibility") / f"{out_json_filename_base}_expressibility.json"
    json_results_expr = {f"{nq}_{d}": (v if not np.isnan(v) else None) for (nq,d),v in results_expressibility.items()}
    out_json_expr.parent.mkdir(parents=True, exist_ok=True)
    out_json_expr.write_text(json.dumps(json_results_expr, indent=2))
    print(f"  EXP Results saved to {out_json_expr}")


if __name__ == "__main__":
    from utils.ansatze.HEA            import HEA
    from utils.ansatze.MPS            import MPSAnsatz
    from utils.ansatze.TTN            import TTNAnsatz
    from utils.ansatze.RealAmplitudes import RealAmplitudeAnsatz
    from utils.ansatze.RandomPQC      import RandomPQCAnsatz
    from utils.ansatze.WIPQC          import WeightInformedPQCAnsatz

    SEED      = 42
    N_QUBITS  = [2, 4, 6, 8, 10]
    DEPTHS    = [2, 3, 4, 5]
    SAMPLES_PER_POINT = 100

    # Define the parameter initialization function and tracked gradient index
    # These are now the defaults in sweep, but shown here for clarity / future modification
    # STANDARD_PARAM_INIT_FN = lambda p: p.data.uniform_(0, 2 * np.pi)
    # STANDARD_TRACKED_GRAD_IDX = 0
    
    ANSATZ_CONFIGS = [
        dict(name="RandomPQC",    _class=RandomPQCAnsatz, init_args={}),
        # dict(name="WI-PQC (E)",   _class=WeightInformedPQCAnsatz, init_args=dict(classical_model=None, angle_scale_factor=0.1)),
        # dict(name="WI-PQC (W+E, chunking)", _class=WeightInformedPQCAnsatz, init_args=dict(classical_model=None, angle_scale_factor=0.1, rot_angle_derivation_strategy="chunking"),
        #      qnn_extra_args_builder=lambda ansatz_obj: {"initial_point": ansatz_obj.get_initial_point()}),
        # dict(name="WI-PQC (W+E, svd)", _class=WeightInformedPQCAnsatz, init_args=dict(classical_model=None, angle_scale_factor=0.1, rot_angle_derivation_strategy="svd"),
        #      qnn_extra_args_builder=lambda ansatz_obj: {"initial_point": ansatz_obj.get_initial_point()}),
        # dict(name="WI-PQC (W1+W2+E, chunking)", _class=WeightInformedPQCAnsatz, init_args=dict(classical_model=None, angle_scale_factor=0.1, rot_use_w1w2_block2=True, rot_angle_derivation_strategy="chunking"),
        #      qnn_extra_args_builder=lambda ansatz_obj: {"initial_point": ansatz_obj.get_initial_point()}),
        # dict(name="WI-PQC (W1+W2+E, svd)", _class=WeightInformedPQCAnsatz, init_args=dict(classical_model=None, angle_scale_factor=0.1, rot_use_w1w2_block2=True, rot_angle_derivation_strategy="svd"),
        #      qnn_extra_args_builder=lambda ansatz_obj: {"initial_point": ansatz_obj.get_initial_point()}),

        dict(name="HEA (ry,rz)",  _class=HEA, init_args=dict(rotation_blocks=["ry", "rz"], entanglement="linear")),
        dict(name="RealAmps",     _class=RealAmplitudeAnsatz, init_args={}),
        dict(name="MPS (d=1)",    _class=MPSAnsatz, init_args=dict(reps=1)),
        dict(name="MPS (d=2)",    _class=MPSAnsatz, init_args=dict(reps=2)),
        dict(name="MPS (d=3)",    _class=MPSAnsatz, init_args=dict(reps=3)),
        dict(name="TTN (d=1)",    _class=TTNAnsatz, init_args=dict(reps=1)),
        dict(name="TTN (d=2)",    _class=TTNAnsatz, init_args=dict(reps=2)),
        dict(name="TTN (d=3)",    _class=TTNAnsatz, init_args=dict(reps=3)),
    ]

    os.makedirs("results/analysis/barren_plateaus", exist_ok=True)

    for cfg_item in ANSATZ_CONFIGS:
        sweep(cfg_item, N_QUBITS, DEPTHS,
              samples=SAMPLES_PER_POINT,
              seed=SEED
              # param_init_fn=STANDARD_PARAM_INIT_FN, # Using default
              # tracked_grad_idx=STANDARD_TRACKED_GRAD_IDX # Using default
              )