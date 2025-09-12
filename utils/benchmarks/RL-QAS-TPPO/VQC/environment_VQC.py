import torch
from qulacs.gate import CNOT, RX, RY, RZ
from vqc_helpers import *
from sys import stdout
import scipy
import VQC as vqc
import numpy as np
import copy
import curricula
from collections import Counter
from qulacs import ParametricQuantumCircuit as _PQC
try:
    from qulacs import QuantumStateGpu as QuantumState
except ImportError:
    from qulacs import QuantumState

from qulacs import ParametricQuantumCircuit, Observable, DensityMatrix
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# KDD data loader used by the environment for TRAINING
import os
import sys
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import mutual_info_classif

from utils.data.preprocessing import data_pipeline

# --- Rest of your .py script code ---

def _mi_top_k(X_df, y, k=8, seed=0):
    mi = mutual_info_classif(X_df.values, y, random_state=seed)
    order = pd.Series(mi, index=X_df.columns).sort_values(ascending=False).index.tolist()
    return order[:k]

def load_kdd_splits_identical(train_csv, test_csv, label_col="class",
                              drop_col="srv_rerror_rate", subsample_train=15000,
                              mi_k=8, seed=42, mi_seed=0, max_features=8):
    tr = pd.read_csv(train_csv); te = pd.read_csv(test_csv)
    tr[label_col] = tr[label_col].map({"normal": 0, "anomaly": 1})
    te[label_col] = te[label_col].map({"normal": 0, "anomaly": 1})
    if drop_col in tr.columns: tr = tr.drop(columns=[drop_col])
    if drop_col in te.columns: te = te.drop(columns=[drop_col])
    from sklearn.model_selection import train_test_split
    tr, _ = train_test_split(tr, train_size=subsample_train,
                             stratify=tr[label_col], random_state=seed)
    X_tr_full, y_tr_full = tr.drop(columns=[label_col]), tr[label_col].to_numpy()
    X_te_full, y_te_full = te.drop(columns=[label_col]), te[label_col].to_numpy()
    top = _mi_top_k(X_tr_full, y_tr_full, k=mi_k, seed=mi_seed)
    scaler = StandardScaler().fit(X_tr_full[top])
    X_tr_s = scaler.transform(X_tr_full[top]); X_te_s = scaler.transform(X_te_full[top])
    X_tr_s = X_tr_s[:, :max_features]; X_te_s = X_te_s[:, :max_features]
    y_tr = np.where(y_tr_full == 0, -1, 1); y_te = np.where(y_te_full == 0, -1, 1)
    return X_tr_s, y_tr, X_te_s, y_te, top


def make_simple_multiclass_data(n_samples=1000, n_features=4, n_classes=4, random_state=42, cluster_std=0.5, separation_scale=3.0):
    """
    Generates a simple synthetic multiclass classification dataset with well-separated clusters.

    This function creates n_classes clusters by distributing their centers along the axes 
    of the feature space, ensuring they are distinct and separated.

    Args:
        n_samples (int): Total number of samples to generate.
        n_features (int): The number of features for each sample.
        n_classes (int): The number of distinct classes (clusters) to generate.
        random_state (int): Seed for the random number generator for reproducibility.
        cluster_std (float): The standard deviation of the Gaussian clusters. Smaller values 
                             lead to tighter, more easily separable clusters.
        separation_scale (float): A factor controlling the distance of cluster centers from the origin.
                                  Larger values result in greater separation between classes.
    """
    
    # This method of creating centers works for up to 2 * n_features classes.
    if n_classes > 2 * n_features:
        raise ValueError(
            f"Cannot create {n_classes} unique centers for {n_features} features "
            f"with this method. Maximum supported classes is {2 * n_features}."
        )

    # Use a modern random number generator for better practice
    rng = np.random.default_rng(random_state)
    
    all_X_parts = []
    all_y_parts = []

    # Distribute samples evenly among classes, handling remainders
    samples_per_class = [n_samples // n_classes] * n_classes
    for i in range(n_samples % n_classes):
        samples_per_class[i] += 1

    for i in range(n_classes):
        # --- Create a unique center for each class ---
        center = np.zeros(n_features)
        # Cycle through feature dimensions to place the center
        feature_idx = i % n_features
        # Use positive and negative directions along the axis
        sign = (-1)**(i // n_features)
        center[feature_idx] = sign * separation_scale
        
        # Generate Gaussian data points around the center
        n_class_samples = samples_per_class[i]
        X_class = (rng.standard_normal((n_class_samples, n_features)) * cluster_std) + center
        y_class = np.full(n_class_samples, i) # Label is the class index

        all_X_parts.append(X_class)
        all_y_parts.append(y_class)

    # Combine the data from all classes into single arrays
    X = np.vstack(all_X_parts)
    y = np.hstack(all_y_parts)
    
    # Shuffle the entire dataset to mix the classes
    shuffle_idx = rng.permutation(n_samples)
    X, y = X[shuffle_idx], y[shuffle_idx]

    # Scale features to a range suitable for angle encoding in quantum circuits
    scaler = MinMaxScaler(feature_range=(0, np.pi))
    X_scaled = scaler.fit_transform(X)

    # --- Split data into training, validation, and test sets ---
    test_size = 0.15
    val_size = 0.15

    # First split to separate training data from a temporary set (validation + test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_scaled, y,
        test_size=(val_size + test_size),
        random_state=random_state,
        stratify=y  # Stratify to ensure class proportions are maintained
    )
    
    # Second split to separate the temporary set into validation and test sets
    relative_test_size = test_size / (val_size + test_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=relative_test_size,
        random_state=random_state,
        stratify=y_temp
    )
    
    return X_train, y_train, X_val, y_val, X_test, y_test



class CircuitEnv():
#step per eps=50-check if implementation is okay
    def __init__(self, conf, seed, device):
        self.num_qubits = conf['env']['num_qubits']
        self.num_layers = conf['env']['num_layers']
        self.random_halt = int(conf['env']['rand_halt'])
        self.n_shots = conf['env']['n_shots']
        noise_models = ['depolarizing', 'two_depolarizing', 'amplitude_damping']
        noise_values = conf['env']['noise_values']
        self.num_samples = int(conf['env']['samples'])
        self.seed = seed
        
        if conf.get('openml', {}).get('dataset_id', None) is not None:
            openml_id = conf.get('openml', {}).get('dataset_id', None)
            print(f"[env] Loading OpenML data for dataset ID: {openml_id}")
            self.n_classes = conf.get('openml', {}).get('n_classes', 2)

            # Note: BATCH_SIZE, DO_PCA etc. could also be made configurable
            _, quantum_data, _, _, _ = data_pipeline(
                openml_dataset_id=int(openml_id),
                n_components=self.num_qubits,
                do_pca=True, # Assuming PCA is desired.
                seed=self.seed
            )
            (x_train_q, _, x_test_q, y_train_q, _, y_test_q, _, _, _) = quantum_data
            
            # Convert tensors to numpy and ensure labels are {-1, 1} for consistency
            self.X_train = x_train_q.numpy()
            self.X_test = x_test_q.numpy()
            # self.y_train = np.where(y_train_q.numpy() == 0, -1, 1)
            # self.y_test = np.where(y_test_q.numpy() == 0, -1, 1)
            self.y_train = y_train_q.numpy().astype(int)
            self.y_test = y_test_q.numpy().astype(int)

            print(f"[env] OpenML splits → train {self.X_train.shape}, test {self.X_test.shape}")

        elif conf.get('clusters', {}).get('n_classes', None) is not None:
            print("[env] Loading clusters data for training and testing")
            cluster_std = float(conf['clusters'].get('cluster_std', 0.5))
            self.n_classes = int(conf['clusters']['n_classes'])
            self.X_train, self.y_train, self.X_test, self.y_test, _, _ = make_simple_multiclass_data(
                n_samples=self.num_samples,
                n_features=self.num_qubits,
                n_classes=self.n_classes,
                random_state=self.seed,
                cluster_std=cluster_std
            )

            print(f"[env] Clusters data splits → train {self.X_train.shape}, test {self.X_test.shape}")

            # # Fallback (kept for safety) – now sized to num_qubits
            # print("[env] WARNING: failed to find paths for KDD data, falling back to synthetic data.")
            # X, y = make_classification(n_samples=self.num_samples,
            #                            n_features=num_qubits,
            #                            n_informative=num_qubits,
            #                            n_redundant=0,
            #                            n_clusters_per_class=1,
            #                            random_state=42, shuffle=True)
            # X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
            # self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            #     X, 2*y-1, test_size=0.2, random_state=42)

        self.mol = conf['problem']['ham_type']

        if noise_values !=0:
            indx = conf['env']['noise_values'].index(',')
            self.noise_values = [float(noise_values[1:indx]), float(noise_values[indx+1:-1])]
        else:
            self.noise_values = []
        self.noise_models = noise_models[0:len(self.noise_values)]
        
        if len(self.noise_models) > 0:
            self.phys_noise = True
        else:
            self.phys_noise = False
        # Initialize VQC with proper parameters
        
        # Problem configuration - now focused on loss minimization
        #self.problem_type = conf['problem']['type']
        self.done_threshold = conf['env'].get('accept_err', 0.01)  # How close we need to get to target
        self.target_loss = self.done_threshold
        # print(self.target_loss, self.done_threshold)

        self.fn_type = conf['env']['fn_type']
        self.cnot_rwd_weight = conf['env'].get('cnot_rwd_weight', 1.)
        self.state_with_angles = conf['agent']['angles']
        self.device = device
        self.state_size = self.num_layers * self.num_qubits * (self.num_qubits + 3 + 3)
        self.action_size = self.num_qubits * (self.num_qubits + 2)
        self.state = torch.zeros((self.num_layers, self.num_qubits+3+3, self.num_qubits))
        self.current_action = [self.num_qubits]*4
        self.previous_action = [0, 0, 0, 0]
        self.illegal_actions = [[]]*self.num_qubits
        self.moments = [0]*self.num_qubits
        self.step_counter = -1
        self.current_number_of_cnots = 0
        self.current_reward = 0
        self.prev_loss = 1000  # Now tracking loss instead of accuracy
        
        self.prev_energy=1000
        # Curriculum learning setup - modified for loss
        self.curriculum_dict = {}
        self.curriculum_dict["classification"] = curricula.__dict__[conf['env']['curriculum_type']](conf['env'], target_energy=self.target_loss)
        self.curriculum = copy.deepcopy(self.curriculum_dict["classification"])
        
        # Optimization configuration (unchanged)
        if 'non_local_opt' in conf:
            self.global_iters = conf['non_local_opt']['global_iters']
            self.optim_method = conf['non_local_opt']["method"]
            self.optim_alg = conf['non_local_opt']['optim_alg']
            self.options = {
                'a': conf['non_local_opt'].get("a", 1.0),
                'alpha': conf['non_local_opt'].get("alpha", 0.602),
                'c': conf['non_local_opt'].get("c", 1.0),
                'gamma': conf['non_local_opt'].get("gamma", 0.101),
                'beta_1': conf['non_local_opt'].get("beta_1", 0.9),
                'beta_2': conf['non_local_opt'].get("beta_2", 0.999)
            }
            if 'lamda' in conf['non_local_opt']:
                self.options['lamda'] = conf['non_local_opt']["lamda"]
            if 'maxfev' in conf['non_local_opt']:
                self.maxfev = {'maxfev': int(conf['non_local_opt']["maxfev"])}
            if 'maxfev1' in conf['non_local_opt']:
                self.maxfevs = {
                    'maxfev1': int(conf['non_local_opt']["maxfev1"]),
                    'maxfev2': int(conf['non_local_opt']["maxfev2"]),
                    'maxfev3': int(conf['non_local_opt']["maxfev3"])
                }
        else:
            self.global_iters = 0
            self.optim_method = None

    def step(self, action, train_flag=True):
        #print(self.make_circuit())
        #print(action)
        next_state = self.state.clone()
        self.step_counter += 1

        # Process action (unchanged)
        ctrl = action[0]
        targ = (action[0] + action[1]) % self.num_qubits
        rot_qubit = action[2]
        rot_axis = action[3]
        self.action = action

        # Update state based on action (unchanged)
        if rot_qubit < self.num_qubits:
            gate_tensor = self.moments[rot_qubit]
        elif ctrl < self.num_qubits:
            gate_tensor = max(self.moments[ctrl], self.moments[targ])
        #print(next_state.shape,gate_tensor)
        if ctrl < self.num_qubits:
            next_state[gate_tensor][targ][ctrl] = 1
            self.current_number_of_cnots += 1
        elif rot_qubit < self.num_qubits:
            next_state[gate_tensor][self.num_qubits+rot_axis-1][rot_qubit] = 1
        # Update moments (unchanged)
        if rot_qubit < self.num_qubits:
            self.moments[rot_qubit] += 1
        elif ctrl < self.num_qubits:
            max_of_two_moments = max(self.moments[ctrl], self.moments[targ])
            self.moments[ctrl] = max_of_two_moments + 1
            self.moments[targ] = max_of_two_moments + 1
            
        self.current_action = action
        self.illegal_action_new()

        # Parameter optimization if enabled - modified to use VQC loss
        if self.optim_method in ["scipy_each_step"]:
            thetas, nfev, opt_ang = self.scipy_optim(self.optim_alg)
            for i in range(self.num_layers):
                for j in range(3):
                    next_state[i][self.num_qubits+3+j,:] = thetas[i][j,:]
        
        self.state = next_state.clone()
        #print(self.make_circuit().calculate_depth())
        # Get loss from VQC
        train_loss, test_loss = self.get_loss()
        self.loss = train_loss
        
        # Curriculum learning update - now using loss
        if train_loss < self.curriculum.lowest_energy and train_flag:
            self.curriculum.lowest_energy = copy.copy(train_loss)
    
        self.error = float(abs(self.target_loss - train_loss))
        self.error_noiseless = self.error
        self.error_test = float(abs(self.target_loss - test_loss))
        self.error_noiseless_test = float(abs(self.target_loss - test_loss))
        # print(self.error, self.error_test)
        rwd = self.reward_fn(train_loss)
        self.current_reward = float(rwd)
        self.prev_loss = copy.copy(train_loss)

        # Check termination conditions
        loss_done = int(self.error < self.done_threshold)
        layers_done = self.step_counter == (self.num_layers - 1)
        done = int(loss_done or layers_done)
        self.previous_action = copy.deepcopy(action)

        if self.random_halt and self.step_counter == self.halting_step:
            done = 1
                
        if done:
            self.curriculum.update_threshold(energy_done=loss_done)
            self.done_threshold = self.curriculum.get_current_threshold()
            self.curriculum_dict["classification"] = copy.deepcopy(self.curriculum)
        
        if self.state_with_angles:
            return next_state.view(-1).to(self.device), torch.tensor(rwd, dtype=torch.float32, device=self.device), done
        else:
            next_state = next_state[:, :self.num_qubits+3]
            return next_state.reshape(-1).to(self.device), torch.tensor(rwd, dtype=torch.float32, device=self.device), done


    # inside class CircuitEnv:
    def get_parametric_circuit(self):
        """
        Try to return the current Qulacs ParametricQuantumCircuit built so far.
        Looks in common places: self.ansatz, self.vqc.ansatz, and scans attributes.
        """
        # common attribute on env
        if hasattr(self, "ansatz"):
            return getattr(self, "ansatz")

        # common: env.vqc.ansatz
        if hasattr(self, "vqc") and hasattr(self.vqc, "ansatz"):
            return self.vqc.ansatz

        # scan attributes for a PQC
        for _, v in self.__dict__.items():
            try:
                if isinstance(v, _PQC):
                    return v
            except Exception:
                pass

        return None

    def get_loss(self, thetas=None):
        """Get loss directly from VQC"""
        circuit = self.make_circuit(thetas)
        #qulacs_inst = vqc.ParametricVQC(num_qubits = self.num_qubits, noise_models = self.noise_models, noise_values = self.noise_values)
        loss_train, loss_test = vqc.loss_wo_angles(self, circuit, n_shots=0)
        return loss_train, loss_test
    
    # def get_loss(self, thetas=None):
    #     """Compute loss on a small subset to speed up env.step()."""
    #     circuit = self.make_circuit(thetas)
    
    #     # Backup full datasets
    #     Xtr_full, ytr_full = self.X_train, self.y_train
    #     Xte_full, yte_full = self.X_test,  self.y_test
    
    #     # --- choose train subset deterministically per step ---
    #     ntr = len(Xtr_full)
    #     if ntr == 0:
    #         return 0.0, 0.0
    #     ktr = min(self.loss_batch, ntr) if self.loss_batch > 0 else max(1, int(self.train_fraction * ntr))
    #     start = (max(self.step_counter, 0) * ktr) % ntr
    #     idx_tr = np.arange(start, start + ktr) % ntr
    #     self.X_train, self.y_train = Xtr_full[idx_tr], ytr_full[idx_tr]
    
    #     # --- choose test subset or skip test ---
    #     if self.compute_test_loss:
    #         nte = len(Xte_full)
    #         if nte > 0:
    #             kte = min(self.loss_batch, nte) if self.loss_batch > 0 else max(1, int(self.test_fraction * nte))
    #             idx_te = np.arange(start, start + kte) % max(nte, 1)
    #             self.X_test, self.y_test = Xte_full[idx_te], yte_full[idx_te]
    #         else:
    #             self.X_test, self.y_test = Xtr_full[:1], ytr_full[:1]  # tiny placeholder
    #     else:
    #         # tiny placeholder so downstream code doesn't choke
    #         self.X_test, self.y_test = Xtr_full[:1], ytr_full[:1]
    
    #     try:
    #         loss_train, loss_test = vqc.loss_wo_angles(self, circuit, n_shots=0)
    #     finally:
    #         # Restore full datasets
    #         self.X_train, self.y_train = Xtr_full, ytr_full
    #         self.X_test,  self.y_test  = Xte_full, yte_full
    
    #     if not self.compute_test_loss:
    #         loss_test = loss_train
    #     return loss_train, loss_test
    

    def scipy_optim(self, method, which_angles=[]):
        state = self.state.clone()
        thetas = state[:, self.num_qubits+3:]
        rot_pos = (state[:,self.num_qubits: self.num_qubits+3] == 1).nonzero( as_tuple = True )
        angles = thetas[rot_pos]
        qulacs_inst = vqc.ParametricVQC(num_qubits=self.num_qubits, num_samples=self.num_samples, noise_models=self.noise_models,noise_values = self.noise_values)
        qulacs_circuit = qulacs_inst.construct_ansatz(state)
        x0 = np.asarray(angles.cpu().detach())

        def cost(x):#get classification loss here
            return vqc.get_classification_loss(self,angles=x, circuit=qulacs_circuit, n_shots=0, which_angles=[])

        if list(which_angles):
            result_min_qulacs = scipy.optimize.minimize(cost, x0 = x0[which_angles], method = method, options = {'maxiter':self.global_iters})
            x0[which_angles] = result_min_qulacs['x']
            thetas = state[:, self.num_qubits+3:]
            thetas[rot_pos] = torch.tensor(x0, dtype=torch.float)
        else:
            result_min_qulacs = scipy.optimize.minimize(cost, x0 = x0, method = method, options = {'maxiter':self.global_iters})
            thetas = state[:, self.num_qubits+3:]
            thetas[rot_pos] = torch.tensor(result_min_qulacs['x'], dtype=torch.float)

        return thetas, result_min_qulacs['nfev'], result_min_qulacs['x']

    def reward_fn(self, loss):
        """Calculate reward based on loss (modified from accuracy)"""
        error = abs(self.target_loss - loss)
        # print(self.fn_type)
        if self.fn_type == "staircase":
            return (0.2 * (error < 15 * self.done_threshold) +
                    0.4 * (error < 10 * self.done_threshold) +
                    0.6 * (error < 5 * self.done_threshold) +
                    1.0 * (error < self.done_threshold)) / 2.2
        elif self.fn_type == "two_step":
            return (0.001 * (error < 5 * self.done_threshold) +
                    1.0 * (error < self.done_threshold))/1.001
        # print(self.prev_loss, loss, )
        elif self.fn_type == "incremental_with_fixed_ends":
            # print('are we here?')
            # print(self.prev_loss, loss, self.prev_loss, self.target_loss)
            max_depth = self.step_counter == (self.num_layers - 1)
            if (self.error < self.done_threshold):
                rwd = 5.
            elif max_depth:
                rwd = -5.
            else:
                rwd = np.clip((self.prev_loss - loss)/abs(self.prev_loss - self.target_loss),-1,1)
            # print(rwd)
            # print(self.num_layers-1, self.step_counter, max_depth, self.error, rwd)
            return rwd
        
        elif self.fn_type == "incremental":
            # Negative because lower loss is better
            # print('???????')
            return -(loss - self.prev_loss)/abs(self.prev_loss - self.target_loss)
        
        elif self.fn_type == "cnot_reduce":
            max_depth = self.step_counter == (self.num_layers - 1)
            if (error < self.done_threshold):
                return self.num_layers - self.cnot_rwd_weight*self.current_number_of_cnots
            elif max_depth:
                return -5.
            else:
                return np.clip(-(loss - self.prev_loss)/abs(self.prev_loss - self.target_loss),-1,1)
        else:  # Default - simple negative loss
            return -loss

    def reset(self):
        """
        Returns randomly initialized state of environment.
        State is a torch Tensor of size (5 x number of layers)
        1st row [0, num of qubits-1] - denotes qubit with control gate in each layer
        2nd row [0, num of qubits-1] - denotes qubit with not gate in each layer
        3rd, 4th & 5th row - rotation qubit, rotation axis, angle
        !!! When some position in 1st or 3rd row has value 'num_qubits',
            then this means empty slot, gate does not exist (we do not
            append it in circuit creator)
        """   
        state = torch.zeros((self.num_layers, self.num_qubits+3+3, self.num_qubits))
        self.state = state
        
        
        if self.random_halt:
            statistics_generated = np.clip(np.random.negative_binomial(n=70,p=0.573, size=1),25,70)[0]
            
            self.halting_step = statistics_generated

        
        self.current_number_of_cnots = 0
        self.current_action = [self.num_qubits]*4
        self.illegal_actions = [[]]*self.num_qubits

        self.make_circuit()

        self.step_counter = -1

        
        self.moments = [0]*self.num_qubits
        #self.current_bond_distance = self.geometry[-3:]
        #self.curriculum = copy.deepcopy(self.curriculum_dict[str(self.current_bond_distance)])
        #self.done_threshold = copy.deepcopy(self.curriculum.get_current_threshold())
        #self.geometry = self.geometry[:-3] + str(self.current_bond_distance)
       # __ham = np.load(f"mol_data/{self.mol}_{self.num_qubits}q_geom_{self.geometry}_{self.ham_mapping}.npz")
        ##self.hamiltonian, self.weights,eigvals, self.energy_shift = __ham['hamiltonian'], __ham['weights'],__ham['eigvals'], __ham['energy_shift']
        #self.min_eig = self.fake_min_energy if self.fake_min_energy is not None else min(eigvals) + self.energy_shift
        #self.max_eig = max(eigvals)+self.energy_shift
        self.prev_energy, _ = self.get_loss() #get loss function
        if self.state_with_angles:
            return state.reshape(-1).to(self.device)
        else:
            state = state[:, :self.num_qubits+3]
            return state.reshape(-1).to(self.device)

    
    def make_circuit(self, thetas=None):
        """
        based on the angle of first rotation gate we decide if any rotation at
        a given qubit is present i.e.
        if thetas[0, i] == 0 then there is no rotation gate on the Control quibt
        if thetas[1, i] == 0 then there is no rotation gate on the NOT quibt
        CNOT gate have priority over rotations when both will be present in the given slot
        """
        state = self.state.clone()
        if thetas is None:
            thetas = state[:, self.num_qubits+3:]
        
        circuit = ParametricQuantumCircuit(self.num_qubits)
        
        for i in range(self.num_layers):
            
            cnot_pos = np.where(state[i][0:self.num_qubits] == 1)
            targ = cnot_pos[0]
            ctrl = cnot_pos[1]
            
            if len(ctrl) != 0:
                for r in range(len(ctrl)):
                    circuit.add_gate(CNOT(ctrl[r], targ[r]))
                    
            rot_pos = np.where(state[i][self.num_qubits: self.num_qubits+3] == 1)
            
            rot_direction_list, rot_qubit_list = rot_pos[0], rot_pos[1]
            
            if len(rot_qubit_list) != 0:
                for pos, r in enumerate(rot_direction_list):
                    rot_qubit = rot_qubit_list[pos]
                    if r == 0:
                        circuit.add_parametric_RX_gate(rot_qubit, thetas[i][0][rot_qubit]) 
                    elif r == 1:
                        circuit.add_parametric_RY_gate(rot_qubit, thetas[i][1][rot_qubit])
                    elif r == 2:
                        circuit.add_parametric_RZ_gate(rot_qubit, thetas[i][2][rot_qubit])
                    else:
                        print(f'rot-axis = {r} is in invalid')
                        
                        assert r >2
        self.ansatz = circuit
        return circuit
        
    def illegal_action_new(self):
        action = self.current_action
        illegal_action = self.illegal_actions
        ctrl, targ = action[0], (action[0] + action[1]) % self.num_qubits
        rot_qubit, rot_axis = action[2], action[3]

        if ctrl < self.num_qubits:
            are_you_empty = sum([sum(l) for l in illegal_action])
            
            if are_you_empty != 0:
                for ill_ac_no, ill_ac in enumerate(illegal_action):
                    
                    if len(ill_ac) != 0:
                        ill_ac_targ = ( ill_ac[0] + ill_ac[1] ) % self.num_qubits
                        
                        if ill_ac[2] == self.num_qubits:
                        
                            if ctrl == ill_ac[0] or ctrl == ill_ac_targ:
                                illegal_action[ill_ac_no] = []
                                for i in range(1, self.num_qubits):
                                    if len(illegal_action[i]) == 0:
                                        illegal_action[i] = action
                                        break

                            elif targ == ill_ac[0] or targ == ill_ac_targ:
                                illegal_action[ill_ac_no] = []
                                for i in range(1, self.num_qubits):
                                    if len(illegal_action[i]) == 0:
                                        illegal_action[i] = action
                                        break
                            
                            else:
                                for i in range(1, self.num_qubits):
                                    if len(illegal_action[i]) == 0:
                                        illegal_action[i] = action
                                        break
                        else:
                            if ctrl == ill_ac[2]:
                                illegal_action[ill_ac_no] = []
                                for i in range(1, self.num_qubits):
                                    if len(illegal_action[i]) == 0:
                                        illegal_action[i] = action
                                        break

                            elif targ == ill_ac[2]:
                                illegal_action[ill_ac_no] = []
                                for i in range(1, self.num_qubits):
                                    if len(illegal_action[i]) == 0:
                                        illegal_action[i] = action
                                        break
                            else:
                                for i in range(1, self.num_qubits):
                                    if len(illegal_action[i]) == 0:
                                        illegal_action[i] = action
                                        break                          
            else:
                illegal_action[0] = action

                            
        if rot_qubit < self.num_qubits:
            are_you_empty = sum([sum(l) for l in illegal_action])
            
            if are_you_empty != 0:
                for ill_ac_no, ill_ac in enumerate(illegal_action):
                    
                    if len(ill_ac) != 0:
                        ill_ac_targ = ( ill_ac[0] + ill_ac[1] ) % self.num_qubits
                        
                        if ill_ac[0] == self.num_qubits:
                            
                            if rot_qubit == ill_ac[2] and rot_axis != ill_ac[3]:
                                illegal_action[ill_ac_no] = []
                                for i in range(1, self.num_qubits):
                                    if len(illegal_action[i]) == 0:
                                        illegal_action[i] = action
                                        break
                            
                            elif rot_qubit != ill_ac[2]:
                                for i in range(1, self.num_qubits):
                                    if len(illegal_action[i]) == 0:
                                        illegal_action[i] = action
                                        break
                        else:
                            if rot_qubit == ill_ac[0]:
                                illegal_action[ill_ac_no] = []
                                for i in range(1, self.num_qubits):
                                    if len(illegal_action[i]) == 0:
                                        illegal_action[i] = action
                                        break
                                        
                            elif rot_qubit == ill_ac_targ:
                                illegal_action[ill_ac_no] = []
                                for i in range(1, self.num_qubits):
                                    if len(illegal_action[i]) == 0:
                                        illegal_action[i] = action
                                        break
                            
                            else:
                                for i in range(1, self.num_qubits):
                                    if len(illegal_action[i]) == 0:
                                        illegal_action[i] = action
                                        break 
            else:
                illegal_action[0] = action
        
        for indx in range(self.num_qubits):
            for jndx in range(indx+1, self.num_qubits):
                if illegal_action[indx] == illegal_action[jndx]:
                    if jndx != indx +1:
                        illegal_action[indx] = []
                    else:
                        illegal_action[jndx] = []
                    break
        
        for indx in range(self.num_qubits-1):
            if len(illegal_action[indx])==0:
                illegal_action[indx] = illegal_action[indx+1]
                illegal_action[indx+1] = []
        
        illegal_action_decode = []
        for key, contain in dictionary_of_actions(self.num_qubits).items():
            for ill_action in illegal_action:
                if ill_action == contain:
                    illegal_action_decode.append(key)
        self.illegal_actions = illegal_action
        return illegal_action_decode
if __name__ == "__main__":
    pass