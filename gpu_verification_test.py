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

# --- PyTorch CUDA Check ---
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

# --- Qiskit Aer Backend Initialization and Timing Check ---
print("\n=== Qiskit Aer Backend Check and Timing ===")

# Define a helper function to run and time simulations
def run_and_time_sim(sim_instance, label, n_qubits_val, depth_val):
    from qiskit.circuit.random import random_circuit
    from qiskit import transpile

    print(f"\n--- Running {label} (N={n_qubits_val}, D={depth_val}) ---")

    qc_local = random_circuit(n_qubits_val, depth_val, measure=False, seed=0)

    # --- MEASURE TRANSPILE TIME ---
    t_transpile_start = time.perf_counter()
    # Adding optimization_level=0 to keep transpilation simple and fast for benchmarking simulation
    circ = transpile(qc_local, sim_instance, optimization_level=0)
    # Add save_statevector to ensure the statevector is explicitly computed and stored
    circ.save_statevector()
    t_transpile_end = time.perf_counter()
    dt_transpile = t_transpile_end - t_transpile_start
    print(f"{label:4s} transpile time: {dt_transpile:7.3f} s")

    gpu_util_history = []
    mem_util_history = []
    stop_event = threading.Event()
    monitor_thread = None

    if label == "GPU": # Only monitor GPU when running GPU backend
        def _monitor_gpu_in_background_local(stop_event, interval=0.1):
            try:
                pynvml.nvmlInit()
                h = pynvml.nvmlDeviceGetHandleByIndex(0)
                while not stop_event.is_set():
                    mem = pynvml.nvmlDeviceGetMemoryInfo(h)
                    util = pynvml.nvmlDeviceGetUtilizationRates(h)
                    gpu_util_history.append(util.gpu)
                    mem_util_history.append(mem.used / 2**20) # Convert to MB
                    time.sleep(interval)
            except pynvml.NVMLError as error:
                print(f"NVML Monitor Thread Error: {error}")
            except Exception as e:
                print(f"Monitor thread unexpected error: {e}")
            finally:
                try:
                    pynvml.nvmlShutdown()
                except: pass

        monitor_thread = threading.Thread(
            target=_monitor_gpu_in_background_local, args=(stop_event,)
        )
        monitor_thread.start()

    # --- MEASURE RUN (SIMULATION) TIME ---
    t_run_start = time.perf_counter()
    job = sim_instance.run(circ)
    res = job.result() # Crucial to wait for results
    t_run_end = time.perf_counter()
    dt_run = t_run_end - t_run_start

    if monitor_thread:
        stop_event.set()
        monitor_thread.join()
        if gpu_util_history:
            avg_gpu_util = sum(gpu_util_history) / len(gpu_util_history)
            max_gpu_util = max(gpu_util_history)
            print(f"[NVML In-Run] Avg GPU Util: {avg_gpu_util:.1f}%, Max GPU Util: {max_gpu_util:.1f}%")
        if mem_util_history:
            avg_mem_used = sum(mem_util_history) / len(mem_util_history)
            print(f"[NVML In-Run] Avg Mem Used: {avg_mem_used:,.0f} MB")

    print(f"{label:4s} run time    : {dt_run:7.3f} s   backend={job.backend().name}")
    return dt_run, dt_transpile # Return both for detailed summary

try:
    from qiskit_aer import AerSimulator, AerError

    # Set N_QUBITS to a value that should benefit from GPU and fit in memory
    # 30 qubits: ~34 GB statevector, fits 80GB A100
    # 31 qubits: ~68 GB statevector, also fits, might show even more GPU benefit
    n_qubits, depth = 24, 40 # Changed from 24 to 30

    # --- Initialize CPU Simulator ---
    sim_cpu = AerSimulator(method="statevector", device="CPU")
    print(f"CPU backend initialized: {sim_cpu.status}")
    cpu_run_t, cpu_transpile_t = run_and_time_sim(sim_cpu, "CPU", n_qubits, depth)

    # --- Initialize GPU Simulator ---
    sim_gpu = None
    try:
        sim_gpu = AerSimulator(method="statevector", device="GPU")
        print(f"\nGPU backend initialized: {sim_gpu.status}")
        print(f"Aer default device for statevector: {sim_gpu.default_options().device}")
        gpu_run_t, gpu_transpile_t = run_and_time_sim(sim_gpu, "GPU", n_qubits, depth)

        # --- Summary ---
        if gpu_run_t is not None:
            speedup = cpu_run_t / gpu_run_t
            print(f"\n=== Summary (Simulation Run Time Only) ===\nCPU Run: {cpu_run_t:7.3f} s   GPU Run: {gpu_run_t:7.3f} s   → speed-up ×{speedup:.2f}")
            print(f"CPU Transpile: {cpu_transpile_t:7.3f} s   GPU Transpile: {gpu_transpile_t:7.3f} s")

            if speedup > 1.0:
                print("STATUS: GPU Run is FASTER than CPU Run. GPU backend is working for simulation.")
            elif speedup > 0.5:
                print("STATUS: GPU Run is SLIGHTLY SLOWER or similar to CPU Run. Overhead may still be present.")
            else:
                print("STATUS: GPU Run is SIGNIFICANTLY SLOWER than CPU Run. This is unexpected for high N and high GPU util.")
        else:
            print(f"\n=== Summary ===\nCPU Run: {cpu_run_t:7.3f} s   GPU backend not available for comparison.")

        # Final check on GPU utilization status
        # Note: avg_gpu_util is only available if GPU run was attempted
        if 'avg_gpu_util' in locals() and avg_gpu_util > 50.0 and gpu_run_t is not None and speedup < 1.0:
             print("FINAL DIAGNOSIS: GPU backend initialized, utilized, but still slower than CPU for run time.")
             print("This suggests a performance bottleneck within Qiskit Aer's GPU kernels for this N or circuit type.")
        elif 'avg_gpu_util' in locals() and avg_gpu_util <= 50.0 and gpu_run_t is not None:
             print("FINAL DIAGNOSIS: GPU backend initialized, but GPU utilization was low.")
             print("This indicates that Qiskit Aer is not effectively using the GPU for computation for this N.")


    except AerError as err:
        print(f"Aer GPU backend initialization FAILED: {err}")
        print("This indicates a severe problem with the qiskit-aer-gpu installation/compatibility.")
        print(f"\n=== Summary ===\nCPU Run: {cpu_run_t:7.3f} s   GPU backend not available.")
    except Exception as e:
        print(f"An unexpected error occurred during Aer GPU test: {e}")
        print(f"\n=== Summary ===\nCPU Run: {cpu_run_t:7.3f} s   GPU test incomplete.")

except ImportError:
    print("Qiskit Aer not installed. Skipping Aer checks and timing.")
except Exception as e:
    print(f"An unexpected error occurred during initial Qiskit Aer setup: {e}")


print("="*50)
print(" VERIFICATION COMPLETE")
print("="*50)