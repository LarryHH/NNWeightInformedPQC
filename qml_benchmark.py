# qml_benchmark.py
import os
import sys
import subprocess
import platform
import time
import threading
import numpy as np
import pandas as pd

# --- PyTorch and Qiskit Imports ---
import torch
import torch.nn as nn
import qiskit
from qiskit import QuantumCircuit
from qiskit.circuit.library import ZZFeatureMap
from qiskit.primitives import Sampler # Base Sampler for CPU
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.connectors import TorchConnector
from qiskit_machine_learning.gradients import ParamShiftSamplerGradient
from qiskit.transpiler import PassManager
from qiskit_machine_learning.utils import algorithm_globals


from utils.nn.QuantumNN import QuantumNN
from utils.ansatze.HEA import HEA

# Conditional import for AerSampler (GPU)
try:
    from qiskit_aer.primitives import SamplerV2 as AerSampler
    _aer_sampler_available = True
except ImportError:
    _aer_sampler_available = False
    print("Warning: qiskit_aer.primitives.SamplerV2 not found. GPU benchmarking will be skipped.")

# --- NVML Imports and Helpers ---
try:
    import pynvml
    _pynvml_available = True
    pynvml.nvmlInit() # Initialize NVML once for the main process
except Exception as e:
    _pynvml_available = False
    print("Warning: pynvml not found or failed to initialize. GPU utilization monitoring will be skipped.")

# Define NVML helper functions (from previous script)
def _get_gpu_stats():
    if not _pynvml_available:
        return None, None
    try:
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        mem = pynvml.nvmlDeviceGetMemoryInfo(h)
        util = pynvml.nvmlDeviceGetUtilizationRates(h)
        return mem.used / 2**20, util.gpu          # MB, %
    except pynvml.NVMLError as error:
        # print(f"NVML Error: {error}") # Suppress frequent errors during monitoring
        return None, None
    except Exception as e:
        # print(f"Error getting NVML stats: {e}") # Suppress frequent errors during monitoring
        return None, None

def _monitor_gpu_in_background_local(stop_event, interval, gpu_util_history, mem_util_history):
    # This function runs in a separate thread to monitor GPU
    try:
        try:
            pynvml.nvmlInit()
        except pynvml.NVMLError_LibraryNotFound:
            print("NVML library not found in monitoring thread. Skipping monitoring.")
            return
        except pynvml.NVMLError_DriverNotLoaded:
            print("NVML driver not loaded in monitoring thread. Skipping monitoring.")
            return
        except pynvml.NVMLError_AlreadyInitialized:
            pass

        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        while not stop_event.is_set():
            m, u = _get_gpu_stats()
            if m is not None and u is not None:
                gpu_util_history.append(u)
                mem_util_history.append(m)
            time.sleep(interval)
    except Exception as e:
        print(f"Monitor thread unexpected error: {e}. Monitoring stopped.")
    finally:
        try:
            pynvml.nvmlShutdown()
        except: pass


# --- Main Benchmark Logic ---
print("\n=== System Information ===")
print(f"OS: {platform.system()} {platform.release()}")
print(f"Python Version: {sys.version}")

# --- NVCC and nvidia-smi Checks (from previous script) ---
print("\n--- NVCC (CUDA Compiler) Check ---")
try:
    result = subprocess.run(["nvcc", "--version"], capture_output=True, text=True, check=True)
    print("nvcc --version output:")
    print(result.stdout)
    if "release" in result.stdout:
        print("NVCC found. CUDA Toolkit is likely installed in the container.")
    else:
        print("NVCC output looks unusual.")
except FileNotFoundError:
    print("NVCC not found in PATH. This means the CUDA Toolkit 'devel' tools are not fully configured or missing.")
except subprocess.CalledProcessError as e:
    print(f"Error running nvcc: {e}\n{e.stderr}")

print("\n--- nvidia-smi Check (Host GPU visibility) ---")
try:
    result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, check=True)
    print("nvidia-smi output:")
    print(result.stdout)
    if "NVIDIA-SMI" in result.stdout:
        print("nvidia-smi successfully executed. GPU visible to container.")
    else:
        print("nvidia-smi output looks unusual.")
except FileNotFoundError:
    print("nvidia-smi not found. Container does not have NVIDIA GPU tools.")
    print("Ensure you run docker with --gpus all and use an NVIDIA base image.")
except subprocess.CalledProcessError as e:
    print(f"Error running nvidia-smi: {e}\n{e.stderr}")

print("\n--- Environment Variables Check ---")
print(f"LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH', 'Not set')}")

print("\n=== PyTorch CUDA Check ===")
try:
    print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"PyTorch CUDA device count: {torch.cuda.device_count()}")
        if torch.cuda.device_count() > 0:
            print(f"PyTorch CUDA current device: {torch.cuda.current_device()}")
            print(f"PyTorch CUDA device name: {torch.cuda.get_device_name(0)}")
            print(f"PyTorch CUDA version: {torch.version.cuda}")
            if not torch.version.cuda.startswith('12'):
                print(f"WARNING: PyTorch CUDA version {torch.version.cuda} does not start with 12. Expecting 12.x.")
        else:
            print("PyTorch CUDA available but no devices found. (Unusual)")
    else:
        print("PyTorch CUDA NOT available. This is a critical issue for GPU usage.")
except ImportError:
    print("PyTorch not installed. Skipping PyTorch CUDA check.")
except Exception as e:
    print(f"Error during PyTorch CUDA check: {e}")


# --- Main QML Benchmark Loop ---
print("\n=== QML Benchmark (Iterating Qubit Counts) ===")

# List to store results for a DataFrame
results_data = []
depth = 2 # HEA depth for the ansatz - you can adjust this
num_classes = 2 # Fixed for binary classification - adjust if your task is different
batch_size = 32 # Representative batch size for QML training - adjust as needed
default_shots = 1024 # Sampler shots - adjust as needed for your specific QML task

# Qubit counts to iterate through
for n_qubits in range(2, 8, 2): # Range from 2 to 28, step 2
    print(f"\n{'='*20} TESTING N_QUBITS = {n_qubits} {'='*20}")

    # --- Prepare dummy input data ---
    xb = torch.randn(batch_size, n_qubits, dtype=torch.float)
    yb = torch.randint(0, num_classes, (batch_size,), dtype=torch.long)

    # --- Initialize HEA Ansatz ---
    ansatz_instance = HEA(n_qubits, depth)

    # --- Benchmark CPU QNN ---
    cpu_qnn_status = "Not run"
    cpu_forward_t, cpu_backward_t = None, None
    try:
        qnn_cpu = QuantumNN(
            ansatz=ansatz_instance,
            n_qubits=n_qubits,
            num_classes=num_classes,
            use_gpu=False,
            default_shots=default_shots,
            gradient_method="param_shift"
        )
        print(f"CPU QNN initialized. Sampler: {qnn_cpu.sampler_device_str}")

        # Time forward pass
        t0_forward = time.perf_counter()
        _ = qnn_cpu(xb) # Run forward pass, ignore output
        t1_forward = time.perf_counter()
        cpu_forward_t = t1_forward - t0_forward

        # Time backward pass (requires a dummy loss and optimizer)
        # Note: qnn_cpu(xb) has already been called for forward timing.
        # Calling it again for backward would add redundant forward computation to backward time.
        # Instead, we just take the last computed log_probs from the previous forward pass
        # for loss calculation, and then time its backward.
        # However, to be fully representative of a training step, we'll run a full _train_batch
        # which includes forward and backward, and extract backward from it if needed,
        # or just time the _train_batch as a whole.
        # For simplicity in this benchmark, let's explicitly run a separate forward and backward pass.
        # To get the backward pass only time, you usually need to separate it from the forward pass.
        # Given how TorchConnector works, the backward pass triggers subsequent Sampler calls.
        # Let's time a full _train_batch which includes both.

        # For clearer separation of Forward vs. Backward as per column names,
        # we'll time forward and then time only the .backward() + optimizer.step() portion.
        # This means the loss calculation needs to be re-done if log_probs are not saved.
        # Let's re-run forward for backward timing if it simplifies the code, or ensure log_probs are available.
        
        # --- Recalculate log_probs if not stored or for clean timing of backward ---
        # Or even better, time the full _train_batch call for more realistic training step timing
        # and then also individual forward for comparison.
        
        # Simpler approach: Time `_train_batch` to represent a full training step (forward + backward)
        # and also time the `forward` explicitly.
        
        optimizer_cpu = torch.optim.SGD(qnn_cpu.model.parameters(), lr=0.1)
        
        t0_full_batch = time.perf_counter()
        # _train_batch includes forward, loss, backward, optimizer.step()
        _ = qnn_cpu._train_batch(xb, yb, optimizer_cpu)
        t1_full_batch = time.perf_counter()
        cpu_full_batch_t = t1_full_batch - t0_full_batch

        # For comparison with GPU's separate forward/backward times, we'll try to estimate.
        # The backward time is the full batch time minus the forward time. This is an approximation.
        cpu_backward_t = cpu_full_batch_t - cpu_forward_t
        if cpu_backward_t < 0: cpu_backward_t = 0 # Ensure non-negative

        print(f"CPU QNN Forward time: {cpu_forward_t:7.3f} s")
        print(f"CPU QNN Backward time (approx): {cpu_backward_t:7.3f} s")
        print(f"CPU QNN Full Batch (Fwd+Bwd+Opt) time: {cpu_full_batch_t:7.3f} s")
        cpu_qnn_status = "Success"

    except Exception as e:
        cpu_qnn_status = f"ERROR: {e}"
        print(f"Error benchmarking CPU QNN for N={n_qubits}: {e}")


    # --- Benchmark GPU QNN ---
    gpu_qnn_status = "Not run"
    gpu_forward_t, gpu_backward_t, avg_gpu_util, avg_mem_used = None, None, None, None
    if torch.cuda.is_available() and _aer_sampler_available:
        try:
            qnn_gpu = QuantumNN(
                ansatz=ansatz_instance,
                n_qubits=n_qubits,
                num_classes=num_classes,
                use_gpu=True,
                default_shots=default_shots,
                gradient_method="param_shift"
            )
            print(f"\nGPU QNN initialized. Sampler: {qnn_gpu.sampler_device_str}")

            # Prepare input on GPU
            xb_gpu = xb.to(qnn_gpu.device)
            yb_gpu = yb.to(qnn_gpu.device)

            gpu_util_history = []
            mem_util_history = []
            stop_event = threading.Event()
            monitor_thread = threading.Thread(
                target=_monitor_gpu_in_background_local,
                args=(stop_event, 0.1, gpu_util_history, mem_util_history)
            )
            monitor_thread.daemon = True
            monitor_thread.start()

            # Time forward pass
            t0_forward = time.perf_counter()
            _ = qnn_gpu(xb_gpu) # Run forward pass, ignore output
            t1_forward = time.perf_counter()
            gpu_forward_t = t1_forward - t0_forward
            
            # Time backward pass (using full _train_batch and estimating backward portion)
            optimizer_gpu = torch.optim.SGD(qnn_gpu.model.parameters(), lr=0.1)

            t0_full_batch = time.perf_counter()
            _ = qnn_gpu._train_batch(xb_gpu, yb_gpu, optimizer_gpu)
            t1_full_batch = time.perf_counter()
            gpu_full_batch_t = t1_full_batch - t0_full_batch

            gpu_backward_t = gpu_full_batch_t - gpu_forward_t
            if gpu_backward_t < 0: gpu_backward_t = 0 # Ensure non-negative

            stop_event.set()
            monitor_thread.join(timeout=10)
            if monitor_thread.is_alive():
                print("Warning: GPU monitoring thread did not terminate gracefully.")

            if gpu_util_history:
                avg_gpu_util = sum(gpu_util_history) / len(gpu_util_history)
                max_gpu_util = max(gpu_util_history)
                print(f"[NVML In-Run] Avg GPU Util: {avg_gpu_util:.1f}%, Max GPU Util: {max_gpu_util:.1f}%")
            if mem_util_history:
                avg_mem_used = sum(mem_util_history) / len(mem_util_history)
                print(f"[NVML In-Run] Avg Mem Used: {avg_mem_used:,.0f} MB")

            print(f"GPU QNN Forward time: {gpu_forward_t:7.3f} s")
            print(f"GPU QNN Backward time (approx): {gpu_backward_t:7.3f} s")
            print(f"GPU QNN Full Batch (Fwd+Bwd+Opt) time: {gpu_full_batch_t:7.3f} s")
            gpu_qnn_status = "Success"

        except Exception as e:
            gpu_qnn_status = f"ERROR: {e}"
            print(f"Error benchmarking GPU QNN for N={n_qubits}: {e}")
    else:
        gpu_qnn_status = "Skipped (CUDA/AerSampler not available)"
        print(f"\nGPU QNN skipped for N={n_qubits}: CUDA not available or AerSampler not found.")


    # --- Summary for current qubit count ---
    forward_speedup = None
    backward_speedup = None
    if gpu_forward_t is not None and cpu_forward_t is not None and cpu_forward_t > 0:
        forward_speedup = cpu_forward_t / gpu_forward_t
    if gpu_backward_t is not None and cpu_backward_t is not None and cpu_backward_t > 0:
        backward_speedup = cpu_backward_t / gpu_backward_t

    print(f"\n=== Summary for N={n_qubits} (QML Forward/Backward) ===")
    print(f"Forward: CPU {cpu_forward_t:7.3f} s | GPU {gpu_forward_t:7.3f} s | Speedup x{forward_speedup:.2f}")
    print(f"Backward: CPU {cpu_backward_t:7.3f} s | GPU {gpu_backward_t:7.3f} s | Speedup x{backward_speedup:.2f}")

    # Store results for DataFrame
    results_data.append({
        'N_Qubits': n_qubits,
        'CPU_Forward_s': cpu_forward_t,
        'CPU_Backward_s': cpu_backward_t,
        'GPU_Forward_s': gpu_forward_t,
        'GPU_Backward_s': gpu_backward_t,
        'Forward_Speedup': forward_speedup,
        'Backward_Speedup': backward_speedup,
        'Avg_GPU_Util_percent': avg_gpu_util,
        'Avg_Mem_Used_MB': avg_mem_used,
        'CPU_QNN_Status': cpu_qnn_status,
        'GPU_QNN_Status': gpu_qnn_status
    })

print("\n"+"="*50)
print(" QML BENCHMARK COMPLETE - SUMMARY TABLE")
print("="*50)

# Print results in a nice table
try:
    results_df = pd.DataFrame(results_data)
    # Order columns nicely
    results_df = results_df[[
        'N_Qubits', 'CPU_Forward_s', 'GPU_Forward_s', 'Forward_Speedup',
        'CPU_Backward_s', 'GPU_Backward_s', 'Backward_Speedup',
        'Avg_GPU_Util_percent', 'Avg_Mem_Used_MB', 'CPU_QNN_Status', 'GPU_QNN_Status'
    ]]
    print(results_df.to_markdown(index=False, floatfmt=".3f"))
    # Save to CSV for easy analysis outside
    results_df.to_csv("qml_benchmark_results.csv", index=False)
    print("\nResults also saved to qml_benchmark_results.csv")
except Exception as e:
    print(f"Error generating summary table: {e}")

print("\n"+"="*50)
print(" END OF SCRIPT")
print("="*50)