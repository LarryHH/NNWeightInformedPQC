import os
import sklearn.metrics
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import json

from .quantum_darts_model import QuantumDARTSModel
from .trainer import DARTSTrainer
from .utils import GATE_POOL, load_mnist_data, load_multiclass_data, load_openml_data


def plot_history(search_history, derived_history):
    """Plots the training loss and the derived circuit performance."""
    plt.figure(figsize=(12, 5))
    
    # Plot 1: Noisy Search Loss
    plt.subplot(1, 2, 1)
    plt.plot(search_history, label='Search Loss (Noisy)', alpha=0.8)
    plt.title("Per-Step Search Loss (Cross-Entropy)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()

    # Plot 2: Derived Circuit Performance (Accuracy)
    plt.subplot(1, 2, 2)
    if derived_history:
        epochs = [item['epoch'] for item in derived_history]
        performance = [item['performance'] for item in derived_history] # Now accuracy
        plt.plot(epochs, performance, 'o-', label='Derived Circuit Accuracy', color='C1')
        plt.title("Derived Circuit Accuracy Over Time")
        plt.xlabel("Epoch")
        plt.ylabel("Best Accuracy Achieved")
        plt.ylim(0, 1.05)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"{results_dir}/learning_curves.png")
    print(f"\nSaved learning curves to {results_dir}/learning_curves.png")



def main(dataset_id, n_qubits, n_layers, results_dir='./results'):
    """
    Main function to configure and run the QuantumDARTS search.
    """
    
    print("--- QuantumDARTS for Classification ---")
    print(f"Training for {EPOCHS} epochs.")
    print("-" * 55)

    # # --- Problem Definition (Machine Learning) ---
    # if DATASET == 'MNIST':
    #     print("Loading MNIST dataset for digits 0 and 1")
    #     X_train, y_train, X_val, y_val, X_test, y_test = load_mnist_data(n_features=N_FEATURES)
    #     n_classes = 2
    # elif DATASET == 'Multiclass':
    #     print(f"Generating synthetic multiclass dataset with {N_CLASSES} classes...")
    #     X_train, y_train, X_val, y_val, X_test, y_test = load_multiclass_data(n_features=N_FEATURES, n_classes=N_CLASSES)
    #     n_classes = N_CLASSES
    # else:
    #     raise ValueError("Unsupported DATASET. Choose 'MNIST' or 'Multiclass'.")

    X_train, y_train, X_val, y_val, X_test, y_test = load_openml_data(dataset_id, n_features=n_qubits, seed=42)
    n_classes = len(np.unique(y_train))
    print(f"Loaded dataset ID {dataset_id} with {n_classes} classes.")

    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # --- Model and Trainer Initialization ---
    model = QuantumDARTSModel(
        n_qubits=n_qubits,
        n_layers=n_layers,
        gate_pool=GATE_POOL,
        n_classes=n_classes
    )
    
    trainer = DARTSTrainer(
        model=model,
        train_loader=train_loader,
        validation_data=val_dataset,
        lr_arch=LEARNING_RATE_ARCH,
        lr_rot=LEARNING_RATE_ROT
)

    # --- Run the Search ---
    print("\nStarting architecture search...")
    trainer.search(num_epochs=EPOCHS, perform_eval=PERFORM_INTERMEDIATE_EVAL, eval_every=EVAL_EVERY)
    print("Search complete.")

    # --- Plot Learning History ---
    search_loss_history = [a_l + r_l for a_l, r_l in zip(trainer.search_arch_loss_history, trainer.search_rot_loss_history)]
    plot_history(search_loss_history, trainer.derived_circuit_performance_history)
    print("Performances")
    print(trainer.derived_circuit_performance_history)
    print("Search Losses")
    print("Arch Losses:")
    print(trainer.search_arch_loss_history)
    print("Rot Losses:")
    print(trainer.search_rot_loss_history)

    with open(f"{results_dir}/qas_history.json", "w") as f:
        json.dump({
            "search_arch_loss_history": trainer.search_arch_loss_history,
            "search_rot_loss_history": trainer.search_rot_loss_history,
            "derived_circuit_performance_history": trainer.derived_circuit_performance_history
        }, f, indent=2)

    # --- Derive, Evaluate, and Report the Final Circuit ---
    print("\nDeriving the best circuit from learned architecture weights...")
    final_circuit = trainer.derive_best_circuit()

    print("\n--- Final Discovered Variational Ansatz ---")
    print(final_circuit)

    # Save the circuit diagram
    try:
        circuit_diagram_path = f"{results_dir}/final_circuit.png"
        final_circuit.draw('mpl', filename=circuit_diagram_path)
        print(f"\nSaved circuit diagram to {circuit_diagram_path}")
    except Exception as e:
        print(f"\nCould not draw circuit: {e}")
    
    # --- Detailed Circuit and Performance Evaluation ---
    # Applicable Metrics:
    circuit_depth = final_circuit.depth()
    num_params = final_circuit.num_parameters
    
    print("\n--- Final Circuit Details ---")
    print(f"Number of Qubits: {model.n_qubits}")
    print(f"Number of Layers: {model.n_layers}")
    print(f"Circuit Depth: {circuit_depth}")
    print(f"Number of Parameters: {num_params}")

    # Not Applicable Metrics:
    print("\nNote on other metrics:")
    print("- 'preds': Predictions are generated and used to calculate accuracy/CM.")

    # Retrain and evaluate the final circuit on the validation/test data
    final_results = trainer.final_evaluation(train_loader, val_loader, test_loader, RETRAINING_STEPS, RETRAINING_LR)
    acc = final_results["accuracy"]
    cm = final_results["confusion_matrix"]
    model = final_results["model"]
    

    print(f"\n--- Final Test Performance ---")
    print(f"TEST ACCURACY = {acc*100:.2f}%")
    print("Confusion matrix (rows=true, cols=pred):")
    print(cm)
    
    # --- Save Final Results to JSON ---
    results_filepath = f"{results_dir}/final_results.json"
    print(f"\nSaving final results to {results_filepath}")
    
    # Gather all results into a dictionary
    output_data = {
        "accuracy": float(acc),
        "confusion_matrix": cm.tolist(), # Convert numpy array to list for JSON
        "test_size": len(val_dataset),
        "circuit_details": {
            "num_qubits": int(model.n_qubits),
            "depth": int(circuit_depth),
            "num_parameters": int(num_params)
        },
        "search_config": {
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE
        },
        "losses": {
            "train_losses": [float(loss) for loss in model.history['train_loss']],
            "val_losses": [float(loss) for loss in model.history['val_loss']]
        },
    }
    
    with open(results_filepath, "w") as f:
        json.dump(output_data, f, indent=2)

if __name__ == "__main__":
    print("Running QuantumDARTS...")

    N_LAYERS = 8  # A slightly deeper circuit for a more complex task
    # Training hyperparameters
    EPOCHS = 20 # Adjust as needed for the larger dataset
    BATCH_SIZE = 32
    LEARNING_RATE_ARCH = 0.02
    LEARNING_RATE_ROT = 0.02

    PERFORM_INTERMEDIATE_EVAL = True
    EVAL_EVERY = 2

    RETRAINING_STEPS = 50
    RETRAINING_LR = 0.01

    DATASETS = {
        "iris": (61, 4),
        "wine": (187, 13),
        "diabetes": (37, 8),
    }
    N_QUBITS = [2,4,6] # [2,4,6,8]
    N_LAYERS = [int(24/q) for q in N_QUBITS]  # Keep depth inversely proportional to qubits
    for n_qubits, n_layers in zip(N_QUBITS, N_LAYERS):
        for dataset, (_, n_features) in DATASETS.items():
            if n_qubits > n_features:
                print(f"Skipping dataset {dataset} with {n_features} features for {n_qubits} qubits.")
                continue
            print(f"\n\n=== Running dataset: {dataset} ===")
            dataset_id = DATASETS[dataset][0]
            results_dir = f"/Users/larryhh/Documents/PhD/Projects/weight_matrix_informed_circuit_design/utils/benchmarks/QuantumDARTS/results/{dataset}_{n_qubits}qubits"
            os.makedirs(results_dir, exist_ok=True)
            main(dataset_id, n_qubits, n_layers, results_dir)

