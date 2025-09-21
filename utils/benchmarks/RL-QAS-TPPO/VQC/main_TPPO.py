import os
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "..", "..", "..", ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from tqdm import tqdm

import numpy as np
import random
import torch
import torch.nn as nn
import sys
import os
import argparse
import pathlib
import copy
from vqc_helpers import get_config, dictionary_of_actions, dict_of_actions_revert_q
from environment_VQC import CircuitEnv, make_simple_multiclass_data
from VQC import _class_probs_from_statevec, _build_zprod_observables, _learn_perm_from_logits, _apply_perm
import agents
import time
torch.set_num_threads(1)
import json
import pickle
import time
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from argparse import Namespace

from utils.data.preprocessing import data_pipeline
from utils.nn.optuna import DATASETS, N_QUBITS


from sklearn.metrics import confusion_matrix
import numpy as np
import math
from qulacs import QuantumState, QuantumCircuit
from qulacs.gate import RY

from qiskit.circuit import QuantumCircuit as QkCircuit, Parameter


def qiskit_to_matrix(circuit: QkCircuit) -> list[list[str]]:
    """
    Converts a Qiskit QuantumCircuit into a matrix-like list of strings,
    preserving the parallel "layered" structure seen in the circuit diagram.

    This function understands that gates on different qubits can occur in the
    same time step (layer) and structures the matrix accordingly. This is
    essential for accurately representing and reconstructing circuits.

    Args:
        circuit: The input Qiskit QuantumCircuit object.

    Returns:
        A list of lists of strings representing the layered circuit structure.
    """
    num_qubits = circuit.num_qubits
    if num_qubits == 0:
        return []

    qubit_map = {qubit: i for i, qubit in enumerate(circuit.qubits)}
    
    # Initialize matrix with one empty list per qubit
    matrix = [[] for _ in range(num_qubits)]
    
    # Tracks the number of columns filled for each qubit's timeline
    # This tells us when each qubit is next available.
    qubit_available_at_col = [0] * num_qubits
    
    multi_qubit_gate_counter = 1
    gate_name_map = {'r': 'rx', 'u1': 'rz', 'x': 'x', 'rz': 'rz', 'rx': 'rx'}

    for instruction in circuit.data:
        op = instruction.operation
        q_args = instruction.qubits

        if not q_args or op.name in ['barrier', 'measure']:
            continue

        op_qubit_indices = [qubit_map[q] for q in q_args]

        # Determine the column where this new gate should start.
        # It's the first column where all its qubits are available.
        start_col = 0
        for idx in op_qubit_indices:
            start_col = max(start_col, qubit_available_at_col[idx])

        # --- Pad all rows with empty strings up to the start column ---
        for i in range(num_qubits):
            padding_needed = start_col - len(matrix[i])
            if padding_needed > 0:
                matrix[i].extend([''] * padding_needed)
        
        # --- Place the gate strings in the correct rows at start_col ---
        if op.num_qubits > 1:
            if op.name == 'cx':
                control_idx, target_idx = op_qubit_indices
                matrix[control_idx].append(f'cx_{multi_qubit_gate_counter}c')
                matrix[target_idx].append(f'cx_{multi_qubit_gate_counter}t')
                multi_qubit_gate_counter += 1
            else:
                # Handle other multi-qubit gates if necessary
                print(f"Warning: Unsupported multi-qubit gate '{op.name}' found.")
                for idx in op_qubit_indices:
                    matrix[idx].append(f'{op.name}_{idx}') # Generic placeholder

        elif op.num_qubits == 1:
            qubit_index = op_qubit_indices[0]
            gate_name = gate_name_map.get(op.name, op.name)
            matrix[qubit_index].append(gate_name)

        # --- Update qubit availability and fill gaps for non-participating qubits ---
        for idx in op_qubit_indices:
            qubit_available_at_col[idx] = start_col + 1
        
        # Ensure all rows are of the same length after adding the gate
        max_len = max(len(row) for row in matrix)
        for i, row in enumerate(matrix):
            if len(row) < max_len:
                row.append('')
                qubit_available_at_col[i] = max_len


    # Final padding to make the matrix rectangular
    max_len = max(len(row) for row in matrix)
    for row in matrix:
        padding_needed = max_len - len(row)
        if padding_needed > 0:
            row.extend([''] * padding_needed)
            
    return matrix

import json

def write_matrix_to_json(matrix: list[list[str]], filename: str):
    """Writes the circuit matrix to a JSON file."""
    with open(filename, 'w') as f:
        json.dump(matrix, f, indent=2) # indent for readability
    print(f"Matrix successfully written to {filename}")

def read_matrix_from_json(filename: str) -> list[list[str]]:
    """Reads a circuit matrix from a JSON file."""
    with open(filename, 'r') as f:
        matrix = json.load(f)
    print(f"Matrix successfully read from {filename}")
    return matrix



# ==============================================================================#
def modify_state_like_training(state, env, conf, device):
    if conf['agent'].get('en_state', 0):
        state = torch.cat((state, torch.tensor(env.prev_energy, dtype=torch.float, device=device).view(1)))
    if conf['agent'].get('threshold_in_state', 0):
        state = torch.cat((state, torch.tensor(env.done_threshold, dtype=torch.float, device=device).view(1)))
    return state

def greedy_eval_episode(env, agent, device, conf):
    """
    Run one greedy pass with the trained policy to assemble a circuit.
    Then try multiple ways to retrieve the Qulacs ParametricQuantumCircuit.
    """
    # 1) Run one greedy pass 
    state = env.reset().clone().detach().to(dtype=torch.double, device=device)
    state = modify_state_like_training(state, env, conf, device)

    for _ in range(env.num_layers + 1):
        illegal = env.illegal_action_new()
        with torch.no_grad():
            logits = agent.policy(state)
            probs  = torch.softmax(logits, dim=-1)
            if illegal is not None:
                probs[illegal] = 0
                probs = probs / probs.sum()
            action = torch.argmax(probs).item()
        next_state, reward, done = env.step(agent.translate[action])
        next_state = next_state.clone().detach().to(dtype=torch.double, device=device)
        next_state = modify_state_like_training(next_state, env, conf, device)
        state = next_state
        if done:
            break

    # 2) Try to fetch a circuit in several ways
    from qulacs import ParametricQuantumCircuit as PQC

    # (a) explicit getter on env, if present
    if hasattr(env, "get_parametric_circuit"):
        try:
            circ = env.get_parametric_circuit()
            if circ is not None:
                return circ
        except Exception as e:
            print(f"[eval] get_parametric_circuit() failed: {e}")
            

    # (b) well-known attribute names
    for name in ("ansatz", "circuit", "parametric_circuit", "pq_circuit"):
        if hasattr(env, name):
            circ = getattr(env, name)
            try:
                if isinstance(circ, PQC):
                    return circ
            except Exception:
                # If isinstance check fails (e.g. different import path) just return it
                return circ

    # (c) ask env to build from its internal state (and expose it)
    if hasattr(env, "make_circuit"):
        try:
            circ = env.make_circuit()
            if circ is not None:
                setattr(env, "ansatz", circ)  # expose for next time
                return circ
        except Exception as e:
            print(f"[eval] make_circuit() failed: {e}")

    # (d) last resort: scan attributes for a PQC instance
    try:
        for _, v in env.__dict__.items():
            if isinstance(v, PQC):
                return v
    except Exception:
        pass

    return None


def regenerate_test_data(conf):
    num_qubits = conf['env']['num_qubits']
    num_samples = int(conf['env']['samples'])
    
    X, y = make_classification(n_samples=num_samples,
                            n_features=num_qubits,
                            n_informative=num_qubits,
                            n_redundant=0,
                            n_clusters_per_class=1,
                            random_state=42, shuffle=True)
    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    _, X_test, _, y_test = train_test_split(
        X, 2*y-1, test_size=0.2, random_state=42)
    return X_test, y_test, num_qubits

def regenerate_open_ml_test_data(conf, seed, n_components=2):
    """
    Loads data from OpenML, processes it, and returns only the test split.
    """    

    openml_id = conf['openml'].get('dataset_id', None)
    if openml_id is None:
        raise ValueError("OpenML ID not found in config.")
    _, quantum_data, _, _, _ = data_pipeline(
        openml_dataset_id=openml_id,
        n_components=n_components,
        do_pca=True, # Assuming PCA is desired for feature reduction
        batch_size=32,
        seed=seed
    )
    (_, _, x_test_q, _, _, y_test_q, _, _, _) = quantum_data
    X_test = x_test_q.numpy()
    y_test = y_test_q.numpy().astype(int)

    print(f"[data] Test set prepared. Shape: {X_test.shape}")
    return X_test, y_test, n_components


def eval_accuracy_on_circuit(
    circuit, X_te, y_te, num_qubits, n_classes,
    X_cal=None, y_cal=None  # ← pass train split here
):
    # decide head
    k = int(math.ceil(math.log2(n_classes))) if n_classes > 1 else 0
    head_type = "probs" if (n_classes > 1 and n_classes == (1 << k) and k <= num_qubits) else "logits"

    # cache observables if logits
    obs_cache = _build_zprod_observables(num_qubits, n_classes) if head_type == "logits" else None

    def _score_batch(X):
        out = []
        for x in X:
            feat = QuantumCircuit(num_qubits)
            for i in range(num_qubits):
                feat.add_gate(RY(i, x[i % len(x)]))
            full = QuantumCircuit(num_qubits)
            full.merge_circuit(feat)
            full.merge_circuit(circuit)
            st = QuantumState(num_qubits)
            full.update_quantum_state(st)

            if head_type == "probs":
                vec = st.get_vector()
                s = _class_probs_from_statevec(vec, num_qubits, n_classes)
            else:
                s = np.array([o.get_expectation_value(st) for o in obs_cache], dtype=float)
            out.append(s)
        return np.vstack(out)  # (N, C)

    # build eval matrix
    M_eval = _score_batch(X_te)

    # learn permutation on calibration split (prefer train; else test)
    if head_type == "logits":
        if X_cal is None or y_cal is None:
            X_cal, y_cal = X_te, y_te
        M_cal = _score_batch(X_cal)
        P = _learn_perm_from_logits(M_cal, y_cal, n_classes)
        M_eval = _apply_perm(M_eval, P)

    preds = np.argmax(M_eval, axis=1)
    acc = float(np.mean(preds == y_te))
    cm = confusion_matrix(y_te, preds, labels=np.arange(n_classes))
    return acc, cm, preds.tolist()

def print_circuit_details(circuit):
    """
    Prints a detailed, human-readable list of gates in a Qulacs circuit.
    """
    print(f"Circuit with {circuit.get_gate_count()} gates and {circuit.get_parameter_count()} parameters.")
    
    gate_count = circuit.get_gate_count()
    
    for i in range(gate_count):
        gate = circuit.get_gate(i)
        
        gate_info = f"Gate {i:2d}: {gate.get_name():<8}"
        targets = f"Targets: {gate.get_target_index_list()}"
        controls = f"Controls: {gate.get_control_index_list()}" if gate.get_control_index_list() else ""
        
        params = ""
        # Safely check if the gate is parametric before calling get_parameter_count()
        if hasattr(gate, 'get_parameter_count') and gate.get_parameter_count() > 0:
            params = f" (Parametric, {gate.get_parameter_count()} params)"

        print(f"  {gate_info} | {targets:<20} | {controls:<15} {params}")

def convert_qulacs_to_qiskit(qulacs_circuit):
    """
    Converts a Qulacs ParametricQuantumCircuit to a Qiskit QuantumCircuit for visualization.
    """
    num_qubits = qulacs_circuit.get_qubit_count()
    qiskit_qc = QkCircuit(num_qubits)
    param_index = 0

    gate_count = qulacs_circuit.get_gate_count()
    for i in range(gate_count):
        gate = qulacs_circuit.get_gate(i)
        name = gate.get_name()
        targets = gate.get_target_index_list()
        controls = gate.get_control_index_list()

        # --- Gate Mapping ---
        if name == "CNOT":
            qiskit_qc.cx(controls[0], targets[0])
        elif name == "ParametricRY":
            theta = Parameter(f'θ{param_index}')
            qiskit_qc.ry(theta, targets[0])
            param_index += 1
        # --- ADDED: Handle RX and RZ gates ---
        elif name == "ParametricRX":
            theta = Parameter(f'θ{param_index}')
            qiskit_qc.rx(theta, targets[0])
            param_index += 1
        elif name == "ParametricRZ":
            theta = Parameter(f'θ{param_index}')
            qiskit_qc.rz(theta, targets[0])
            param_index += 1
        # ------------------------------------
        elif name == "X":
            qiskit_qc.x(targets[0])
        elif name == "H":
            qiskit_qc.h(targets[0])
        else:
            print(f"Warning: Skipping unsupported gate for conversion: {name}")
            
    return qiskit_qc


def save_circuit_to_file(circuit, filename):
    """
    Saves a Qulacs circuit object to a file using pickle.
    """
    with open(filename, 'wb') as f:
        pickle.dump(circuit, f)
    print(f"Circuit successfully saved to {filename}")

# ==============================================================================#


class Saver:
    def __init__(self, results_path, experiment_seed):
        self.stats_file = {'train': {}, 'test': {}}
        self.exp_seed = experiment_seed
        self.rpath = results_path

    def get_new_episode(self, mode, episode_no):
        if mode == 'train':
            self.stats_file[mode][episode_no] = {'loss_policy': [],
                                                 'loss_value': [],
                                                 'actions': [],
                                                 'errors': [],
                                                 'errors_test': [],
                                                 'done_threshold': 0,
                                                 'bond_distance': 0,
                                                 'nfev': [], 
                                                 'opt_ang': [],
                                                 'time': [],
                                                 'rewards': []}
        elif mode == 'test':
            self.stats_file[mode][episode_no] = {'actions': [],
                                                 'errors': [],
                                                 'errors_test': [],
                                                 'done_threshold': 0,
                                                 'bond_distance': 0,
                                                 'nfev': [],
                                                 'opt_ang': [],
                                                 'time': [],
                                                 'rewards': []}

    def save_file(self):
        # print(self.stats_file['train'][0])
        with open(f"{self.rpath}/summary_{self.exp_seed}.json", "w") as outfile:
            json.dump(self.stats_file, outfile)

    def validate_stats(self, episode, mode):
        assert len(self.stats_file[mode][episode]['actions']) == len(self.stats_file[mode][episode]['errors'])

def modify_state(state, env, conf, device):
    if not torch.is_tensor(state):
        state = torch.tensor(state, dtype=torch.double, device=device)
    if conf['agent']['en_state']:
        # print(f"Debug: env.prev_energy type={type(env.prev_energy)}, value={env.prev_energy}")
        if env.prev_energy is None:
            raise ValueError("env.prev_energy is None; ensure CircuitEnv.step() sets it correctly")
        prev_energy = torch.tensor(float(env.prev_energy), dtype=torch.double, device=device).unsqueeze(0)
        state = torch.cat((state, prev_energy))
    if "threshold_in_state" in conf['agent'].keys() and conf['agent']["threshold_in_state"]:
        done_threshold = torch.tensor([env.done_threshold], dtype=torch.double, device=device)
        state = torch.cat((state, done_threshold))
    return state

def agent_test(env, agent, episode_no, seed, output_path, threshold):
    """Testing function of the trained agent."""
    agent.saver.get_new_episode('test', episode_no)
    state = env.reset()
    state = modify_state(state, env, conf, agent.device)
    agent.policy.eval()

    for t in range(env.num_layers + 1):
        ill_action_from_env = env.illegal_action_new()
        with torch.no_grad():
            action, _ = agent.act(state)
            assert type(action) == int
            agent.saver.stats_file['test'][episode_no]['actions'].append(action)
        next_state, reward, done = env.step(agent.translate[action], train_flag=False)
        next_state = modify_state(next_state, env, conf, agent.device)
        state = next_state.clone()
        assert type(env.error) == float 
        agent.saver.stats_file['test'][episode_no]['errors'].append(env.error)
        agent.saver.stats_file['test'][episode_no]['errors_test'].append(env.error_test)
        agent.saver.stats_file['test'][episode_no]['opt_ang'].append(env.opt_ang_save)
        agent.saver.stats_file['test'][episode_no]['rewards'].append(env.current_reward)
        
        if done:
            agent.saver.stats_file['test'][episode_no]['done_threshold'] = env.done_threshold
            errors_current_bond = [val['errors'][-1] for val in agent.saver.stats_file['test'].values()
                                   if val['done_threshold'] == env.done_threshold]
            if len(errors_current_bond) > 0 and min(errors_current_bond) > env.error:
                torch.save(agent.policy.state_dict(), f"{output_path}/thresh_{threshold}_{seed}_best_geo_{env.current_bond_distance}_model.pt")
                torch.save(agent.optim_policy.state_dict(), f"{output_path}/thresh_{threshold}_{seed}_best_geo_{env.current_bond_distance}_optim.pt")
            agent.saver.validate_stats(episode_no, 'test')
            # print("Hello")
            return reward, t

def one_episode(episode_no, env, agent, episodes):
    """Function performing full training episode with TPPO."""
    t0 = time.time()
    agent.saver.get_new_episode('train', episode_no)
    state = env.reset()
    agent.saver.stats_file['train'][episode_no]['done_threshold'] = env.done_threshold
    
    state = modify_state(state, env, conf, agent.device)
    
    done = False
    for itr in range(env.num_layers + 1):
        ill_action_from_env = env.illegal_action_new()
        action, log_prob = agent.act(state)
        next_state, reward, done = env.step(agent.translate[action])
        # Convert reward to CPU scalar before storing
        agent.rewards.append(reward.item() if torch.is_tensor(reward) else reward)
        
        agent.saver.stats_file['train'][episode_no]['actions'].append(action)
        agent.saver.stats_file['train'][episode_no]['errors'].append(env.error)
        agent.saver.stats_file['train'][episode_no]['errors_test'].append(env.error_test)
        agent.saver.stats_file['train'][episode_no]['rewards'].append(env.current_reward)
        agent.saver.stats_file['train'][episode_no]['time'].append(time.time() - t0)

        next_state = modify_state(next_state, env, conf, agent.device)
        state = next_state

        if done:
            break

    agent.update()
    agent.saver.validate_stats(episode_no, 'train')

def train(agent, env, episodes, seed, output_path, threshold):
    """Training loop"""
    threshold_crossed = 0
    for e in tqdm(range(episodes), desc="Training Progress"):
        one_episode(e, env, agent, episodes)
        if e % 20 == 0 and e > 0:
            agent.saver.save_file()
            torch.save(agent.policy.state_dict(), f"{output_path}/thresh_{threshold}_{seed}_policy_model.pt")
            torch.save(agent.value.state_dict(), f"{output_path}/thresh_{threshold}_{seed}_value_model.pt")
            torch.save(agent.optim_policy.state_dict(), f"{output_path}/thresh_{threshold}_{seed}_policy_optim.pt")
            torch.save(agent.optim_value.state_dict(), f"{output_path}/thresh_{threshold}_{seed}_value_optim.pt")
        if env.error <= 0.0016:
            threshold_crossed += 1

def get_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0, help='Seed for reproduction')
    parser.add_argument('--config', type=str, default='h_s_2', help='Name of configuration file')
    parser.add_argument('--experiment_name', type=str, default='lower_bound_energy/', help='Name of experiment')
    parser.add_argument('--gpu_id', type=int, default=0, help='Set specific GPU to run experiment [0, 1, ...]')
    parser.add_argument('--wandb_group', type=str, default='test/', help='Group of experiment run for wandb')
    parser.add_argument('--wandb_name', type=str, default='test/', help='Name of experiment run for wandb')
    args = parser.parse_args(argv)
    return args

if __name__ == '__main__':

    N_QUBITS = [2,4,6,8] 
    DATASETS = {
        "iris": (61, 4),
        "wine": (187, 13),
        "diabetes": (37, 8),
        # "clusters": (None, 2),  # Synthetic dataset with 2 features
    }
    for n_qubits in N_QUBITS:
        for dataset, (_, n_features) in DATASETS.items():
            if n_qubits > n_features:
                print(f"Skipping dataset {dataset} with {n_features} features for {n_qubits} qubits.")
                continue

            args_dict = {
                "seed": 0,
                "config": f'configuration_files/TPPO/{dataset}_coblya_{n_qubits}q_VQC',
                "experiment_name": 'TPPO/'
            }
                
            # 2. Convert the dictionary into a Namespace object
            args = Namespace(**args_dict)

            print("\n### Running Bench-RLQAS with the following setup: ###")
            print("> Seed:", args.seed)
            print("> Config:", args.config)
            print()

            results_path ="results/"
            pathlib.Path(f"{results_path}{args.experiment_name}{args.config}").mkdir(parents=True, exist_ok=True)
            # device = torch.device(f"cuda:{args.gpu_id}")
            device = torch.device("cpu")  # Uncomment to force CPU if needed

            conf = get_config(args.experiment_name, f'{args.config}.cfg')

            torch.backends.cudnn.deterministic = True
            random.seed(args.seed)
            torch.manual_seed(args.seed)
            torch.cuda.manual_seed(args.seed)
            np.random.seed(args.seed)


            """ Environment and Agent initialization """
            environment = CircuitEnv(conf, args.seed, device=device)

            # Calculate effective state_size based on actual state output
            initial_state = environment.reset()
            base_state_size = initial_state.shape[0]  # Use actual size from reset
            effective_state_size = base_state_size
            if conf['agent']['en_state']:
                effective_state_size += 1  # For prev_energy
            if "threshold_in_state" in conf['agent'].keys() and conf['agent']["threshold_in_state"]:
                effective_state_size += 1  # For done_threshold

            # Debugging: Verify state size consistency
            modified_state = modify_state(initial_state, environment, conf, device)
            actual_state_size = modified_state.shape[0]
            # print(f"Environment state_size: {environment.state_size}")
            # print(f"Base state size (from reset): {base_state_size}, Effective state size: {effective_state_size}")
            # print(f"Modified state shape: {modified_state.shape}, Expected size: {effective_state_size}")
            if actual_state_size != effective_state_size:
                raise ValueError(f"State size mismatch: expected {effective_state_size}, got {actual_state_size}")
            if base_state_size != environment.state_size:
                print(f"Warning: environment.state_size ({environment.state_size}) differs from actual state size ({base_state_size})")

            # Initialize agent with the corrected effective state size
            agent = agents.__dict__[conf['agent']['agent_type']].__dict__[conf['agent']['agent_class']](
                conf, environment.action_size, effective_state_size, device
            )
            agent.saver = Saver(f"{results_path}{args.experiment_name}{args.config}", args.seed)
            
            # Debug: Verify device placement
            # print(f"Agent policy device: {next(agent.policy.parameters()).device}")
            # print(f"Agent value device: {next(agent.value.parameters()).device}")
            # print(f"Agent action size: {agent.action_size}, Translate dict length: {len(agent.translate)}")

            train(agent, environment, conf['general']['episodes'], args.seed, 
                f"{results_path}{args.experiment_name}{args.config}", conf['env']['accept_err'])
            agent.saver.save_file()

            
            # ==============================================================================#
            try:
                circuit = greedy_eval_episode(environment, agent, device, conf)
                print("Retrieved circuit via get_parametric_circuit()")
                print_circuit_details(circuit)
                circuit_fp = f"{results_path}{args.experiment_name}{args.config}/thresh_{conf['env']['accept_err']}_{args.seed}"
                qiskit_circuit = convert_qulacs_to_qiskit(circuit)
                qiskit_circuit.draw(output='mpl', filename=f"{circuit_fp}_circuit.png")
                # save_circuit_to_file(circuit, f"{circuit_fp}_circuit.pkl")
                circuit_matrix = qiskit_to_matrix(qiskit_circuit)
                write_matrix_to_json(circuit_matrix, f"{circuit_fp}_circuit.json")
            except Exception as e:
                print(f"[eval] get_parametric_circuit() failed: {e}")
            # ==============================================================================#

            if circuit is None:
                print("[eval] WARNING: Could not retrieve a parametric circuit; skipping accuracy.")
            else:
                if 'openml' in conf:
                    X_te, y_te, num_qubits = regenerate_open_ml_test_data(conf, args.seed, n_components=conf['env']['num_qubits'])
                    n_classes = conf.get('openml', {}).get('n_classes', 2)
                elif 'clusters' in conf:
                    num_qubits = int(conf['env']['num_qubits'])
                    cluster_std = float(conf['clusters'].get('cluster_std', 0.5))
                    n_classes = int(conf['clusters']['n_classes'])
                    _, _, X_te, y_te, _, _ = make_simple_multiclass_data(
                        n_samples=int(conf['env']['samples']),
                        n_features=num_qubits,
                        n_classes=n_classes,
                        random_state=args.seed,
                        cluster_std=cluster_std
                    )
                else:
                    raise ValueError("No valid data source specified in config.")

                acc, cm, _preds = eval_accuracy_on_circuit(circuit, X_te, y_te, num_qubits, n_classes)
                
                print(f"\n[eval] TEST ACCURACY = {acc*100:.2f}%")
                
                # --- MODIFIED PRINTOUT ---
                # This now prints the multiclass confusion matrix correctly
                print("[eval] Confusion matrix (rows=true, cols=pred):")
                print(cm)

                # Save alongside other results
                out_dir = f"{results_path}{args.experiment_name}{args.config}/thresh_{conf['env']['accept_err']}_{args.seed}_results.json"
                print(f"[eval] Saving results to {out_dir}")
                with open(out_dir, "w") as f:
                    # Saving the confusion matrix as a list of lists for JSON compatibility
                    json.dump({
                        "accuracy": float(acc),
                        "confusion_matrix": cm.tolist(), 
                        "test_size": int(len(y_te)),
                        "num_qubits": int(num_qubits),
                        "episodes": int(conf['general']['episodes']),
                        "num_layers": int(conf['env']['num_layers'])
                    }, f, indent=2)
            
            torch.save(agent.policy.state_dict(), f"{results_path}{args.experiment_name}{args.config}/thresh_{conf['env']['accept_err']}_{args.seed}_policy_model.pt")
            torch.save(agent.value.state_dict(), f"{results_path}{args.experiment_name}{args.config}/thresh_{conf['env']['accept_err']}_{args.seed}_value_model.pt")
            torch.save(agent.optim_policy.state_dict(), f"{results_path}{args.experiment_name}{args.config}/thresh_{conf['env']['accept_err']}_{args.seed}_policy_optim.pt")
            torch.save(agent.optim_value.state_dict(), f"{results_path}{args.experiment_name}{args.config}/thresh_{conf['env']['accept_err']}_{args.seed}_value_optim.pt")