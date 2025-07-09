# verify_gpu_setup.py
import os
import sys
import subprocess
import platform
import time
import pynvml
import threading

print("="*50)
print(" VERIFYING GPU SETUP INSIDE CONTAINER")
print("="*50)

# --- System Checks ---
print("\n--- System Information ---")
print(f"OS: {platform.system()} {platform.release()}")
print(f"Python Version: {sys.version}")

# --- NVCC (CUDA Compiler) Check ---
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
    print("While PyTorch/Aer might still run (if runtime libs are there), this is a good sign for comprehensive setup.")
except subprocess.CalledProcessError as e:
    print(f"Error running nvcc: {e}\n{e.stderr}")

# --- nvidia-smi Check ---
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

# --- Environment Variables Check ---
print("\n--- Environment Variables Check ---")
print(f"LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH', 'Not set')}")
# Expect this to contain paths like /usr/local/cuda/lib64

# --- PyTorch CUDA Check (Your existing excellent check) ---
print("\n=== PyTorch CUDA Check ===")
try:
    import torch
    print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"PyTorch CUDA device count: {torch.cuda.device_count()}")
        if torch.cuda.device_count() > 0:
            print(f"PyTorch CUDA current device: {torch.cuda.current_device()}")
            print(f"PyTorch CUDA device name: {torch.cuda.get_device_name(0)}")
            print(f"PyTorch CUDA version: {torch.version.cuda}")
            # Verify PyTorch CUDA version matches expected 12.x
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

# --- Qiskit Aer GPU Backend Initialization Check ---
print("\n=== Qiskit Aer GPU Backend Check ===")
try:
    from qiskit_aer import AerSimulator, AerError
    print("Attempting to initialize AerSimulator with device='GPU'...")
    sim_gpu = AerSimulator(method="statevector", device="GPU")
    print(f"AerSimulator GPU backend initialized successfully: {sim_gpu.status}")
    print(f"Aer version: {sim_gpu.version}")

    # Small test run to trigger actual GPU usage and monitor utilization
    print("\n--- Running a small Qiskit Aer GPU test simulation (N=28, D=40) ---")
    from qiskit.circuit.random import random_circuit
    from qiskit import transpile

    n_qubits, depth = 28, 40
    qc = random_circuit(n_qubits, depth, measure=False, seed=0)
    circ = transpile(qc, sim_gpu)

    # In-run GPU monitoring
    stop_event = threading.Event() # Define stop_event here
    gpu_util_history = []
    mem_util_history = []

    def _monitor_gpu_in_background_local(stop_event, interval=0.1):
        try:
            pynvml.nvmlInit() # Re-initialize for this thread
            h = pynvml.nvmlDeviceGetHandleByIndex(0)
            while not stop_event.is_set():
                mem = pynvml.nvmlDeviceGetMemoryInfo(h)
                util = pynvml.nvmlDeviceGetUtilizationRates(h)
                gpu_util_history.append(util.gpu)
                mem_util_history.append(mem.used / 2**20)
                time.sleep(interval)
        except pynvml.NVMLError as error:
            print(f"NVML Monitor Thread Error: {error}")
        except Exception as e:
            print(f"Monitor thread unexpected error: {e}")
        finally:
            try:
                pynvml.nvmlShutdown() # Shutdown for this thread
            except: pass # Ignore if already shutdown

    import threading
    monitor_thread = threading.Thread(
        target=_monitor_gpu_in_background_local, args=(stop_event,)
    )
    monitor_thread.start()

    t0 = time.perf_counter()
    job = sim_gpu.run(circ)
    res = job.result() # Crucial to wait for results
    dt = time.perf_counter() - t0
    stop_event.set()
    monitor_thread.join()

    print(f"Test run elapsed: {dt:.3f} s")
    if gpu_util_history:
        avg_gpu_util = sum(gpu_util_history) / len(gpu_util_history)
        max_gpu_util = max(gpu_util_history)
        print(f"[NVML In-Run] Avg GPU Util: {avg_gpu_util:.1f}%, Max GPU Util: {max_gpu_util:.1f}%")
    if mem_util_history:
        avg_mem_used = sum(mem_util_history) / len(mem_util_history)
        print(f"[NVML In-Run] Avg Mem Used: {avg_mem_used:,.0f} MB")

    if avg_gpu_util > 50.0: # Threshold for 'good' utilization
        print("STATUS: GPU utilization is GOOD for test run. Aer GPU backend appears functional.")
    else:
        print("STATUS: GPU utilization is LOW for test run. Aer GPU backend may not be engaging GPU effectively.")

except AerError as err:
    print(f"Aer GPU backend initialization FAILED: {err}")
    print("This indicates a severe problem with the qiskit-aer-gpu installation/compatibility.")
except ImportError:
    print("Qiskit Aer not installed. Skipping Aer checks.")
except Exception as e:
    print(f"An unexpected error occurred during Aer GPU check: {e}")

print("="*50)
print(" VERIFICATION COMPLETE")
print("="*50)