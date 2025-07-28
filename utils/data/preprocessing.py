import openml
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


def get_data(openml_dataset_id):
    try:
        dataset = openml.datasets.get_dataset(openml_dataset_id)
        print(f"Dataset name: {dataset.name}")
    except Exception as e:
        print(f"Error fetching dataset {openml_dataset_id}: {e}")
        return None
    X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)
    X = pd.get_dummies(X)
    le = LabelEncoder()
    y = le.fit_transform(y)
    return X, y, le

def process_data(X, y):
    """Clean data: drop NaNs, duplicates, constant columns, and remove categoricals.
    Assumes X is DataFrame, y is ndarray."""
    y = pd.Series(y, index=X.index)

    total_features = X.shape[1]
    num_categoricals = X.select_dtypes(exclude=["number"]).shape[1]

    X = X.dropna()
    y = y.loc[X.index]

    X = X.drop_duplicates()
    y = y.loc[X.index]

    X = X.select_dtypes(include=["number"]) # dropping categorical features currently

    remaining_features = X.shape[1]

    print(f"Total features: {total_features}, Dropped Categorical features: {num_categoricals}, Remaining features: {remaining_features}")

    return pd.DataFrame(X), y.to_numpy()


def scale_data_classical(x_train, x_val, x_test):
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_val = scaler.transform(x_val)
    x_test = scaler.transform(x_test)

    return x_train, x_val, x_test

def scale_data_quantum(x_train, x_val, x_test):
    # scaler = MinMaxScaler() # MinMaxScaler(feature_range=(0, np.pi))
    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train)
    x_val = scaler.transform(x_val)
    x_test = scaler.transform(x_test)
    return x_train, x_val, x_test

def apply_pca(X, n_components=2, seed=42):
    pca = PCA(n_components=n_components, random_state=seed)
    X_pca = pca.fit_transform(X)
    return X_pca

def create_data_splits(X, y, batch_size=32, is_classical=True, do_pca=False, use_gpu=False, n_components=2, seed=42):

    is_multiclass = len(np.unique(y)) > 2

    if do_pca:
        X = apply_pca(X, n_components=n_components, seed=seed)
        print(f"{'Classical' if is_classical else 'Quantum'} data reduced to {X.shape[1]} dimensions using PCA.")

    x_train, x_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=seed, stratify=y) # 60/20/20
    x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=seed, stratify=y_temp)

    if is_classical:
        x_train, x_val, x_test = scale_data_classical(x_train, x_val, x_test)
    else:
        x_train, x_val, x_test = scale_data_quantum(x_train, x_val, x_test)

    x_train = torch.tensor(x_train, dtype=torch.float32)
    x_val = torch.tensor(x_val, dtype=torch.float32)
    x_test = torch.tensor(x_test, dtype=torch.float32)

    y_train = torch.tensor(y_train, dtype=torch.long)
    y_val = torch.tensor(y_val, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)

    # train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    # val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=batch_size)
    # test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=batch_size)

    if use_gpu and torch.cuda.is_available():
        print("Configuring DataLoader for GPU.")
        dataloader_kwargs = {
            'num_workers': 4,  # Use multiple subprocesses to load data
            'pin_memory': True # Speeds up host-to-device transfer
        }
    else:
        print("Configuring DataLoader for CPU.")
        dataloader_kwargs = {}

    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True, **dataloader_kwargs)
    val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=batch_size, **dataloader_kwargs)
    test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=batch_size, **dataloader_kwargs)
    
    input_dim = x_train.shape[1]
    output_dim = len(np.unique(y))

    return x_train, x_val, x_test, y_train, y_val, y_test, train_loader, val_loader, test_loader, input_dim, output_dim, is_multiclass

def data_pipeline(openml_dataset_id, batch_size=32, do_pca=False, use_gpu=False, n_components=2, seed=42):
    X, y, le = get_data(openml_dataset_id)
    if X is None:
        raise ValueError("Failed to fetch data. Exiting.")

    X, y = process_data(X, y)

    # --- Classical ---
    (
        x_train_c,
        x_val_c,
        x_test_c,
        y_train_c,
        y_val_c,
        y_test_c,
        train_loader_c,
        val_loader_c,
        test_loader_c,
        input_dim,
        output_dim,
        is_multiclass,
    ) = create_data_splits(X, y, batch_size=batch_size, is_classical=True, do_pca=do_pca, use_gpu=use_gpu, n_components=n_components, seed=seed)

    # --- Quantum ---
    (
        x_train_q,
        x_val_q,
        x_test_q,
        y_train_q,
        y_val_q,
        y_test_q,
        train_loader_q,
        val_loader_q,
        test_loader_q,
        _,
        _,
        _,
    ) = create_data_splits(X, y, batch_size=batch_size, is_classical=False, do_pca=do_pca, use_gpu=use_gpu, n_components=n_components, seed=seed)
    classical_data = (
        x_train_c,
        x_val_c,
        x_test_c,
        y_train_c,
        y_val_c,
        y_test_c,
        train_loader_c,
        val_loader_c,
        test_loader_c,
    )
    quantum_data = (
        x_train_q,
        x_val_q,
        x_test_q,
        y_train_q,
        y_val_q,
        y_test_q,
        train_loader_q,
        val_loader_q,
        test_loader_q,
    )
    return classical_data, quantum_data, input_dim, output_dim, is_multiclass