# verify_gpu_setup.py
import os
import sys
import subprocess
import platform
import time
import pynvml
import threading
import pandas as pd # Added for storing results in a table

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
print("\n=== Qiskit Aer Backend Check and Timing (Iterating Qubit Counts) ===")

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
                # NVML can sometimes fail if device goes away or driver issues
                print(f"NVML Monitor Thread Error: {error}. Monitoring stopped.")
                break # Exit loop on error
            except Exception as e:
                print(f"Monitor thread unexpected error: {e}. Monitoring stopped.")
                break # Exit loop on error
            finally:
                # Ensure shutdown is called on thread exit
                try:
                    pynvml.nvmlShutdown()
                except: pass # Ignore if already shutdown or not initialized

        monitor_thread = threading.Thread(
            target=_monitor_gpu_in_background_local, args=(stop_event,)
        )
        monitor_thread.daemon = True # Allow main program to exit even if thread is running
        monitor_thread.start()

    # --- MEASURE RUN (SIMULATION) TIME ---
    t_run_start = time.perf_counter()
    job = sim_instance.run(circ)
    res = job.result() # Crucial to wait for results
    t_run_end = time.perf_counter()
    dt_run = t_run_end - t_run_start

    if monitor_thread:
        stop_event.set()
        monitor_thread.join(timeout=10) # Give thread some time to finish gracefully
        if monitor_thread.is_alive():
            print("Warning: GPU monitoring thread did not terminate gracefully.")
        if gpu_util_history:
            avg_gpu_util = sum(gpu_util_history) / len(gpu_util_history)
            max_gpu_util = max(gpu_util_history)
            print(f"[NVML In-Run] Avg GPU Util: {avg_gpu_util:.1f}%, Max GPU Util: {max_gpu_util:.1f}%")
        if mem_util_history:
            avg_mem_used = sum(mem_util_history) / len(mem_util_history)
            print(f"[NVML In-Run] Avg Mem Used: {avg_mem_used:,.0f} MB")

    print(f"{label:4s} run time    : {dt_run:7.3f} s   backend={job.backend().name}")
    # Return both times for detailed summary, along with GPU metrics
    return dt_run, dt_transpile, avg_gpu_util if 'avg_gpu_util' in locals() else None, avg_mem_used if 'avg_mem_used' in locals() else None

# List to store results for a DataFrame
results_data = []
depth = 40 # Constant depth for all qubit counts

# Qubit counts to iterate through
# Adjusted to be inclusive of 2 and 28, step 2
for n_qubits in range(2, 29, 2):
    print(f"\n{'='*20} TESTING N_QUBITS = {n_qubits} {'='*20}")

    try:
        from qiskit_aer import AerSimulator, AerError

        # --- Initialize CPU Simulator ---
        sim_cpu = AerSimulator(method="statevector", device="CPU")
        print(f"CPU backend initialized: {sim_cpu.status}")
        cpu_run_t, cpu_transpile_t, _, _ = run_and_time_sim(sim_cpu, "CPU", n_qubits, depth)

        # --- Initialize GPU Simulator ---
        sim_gpu_status = "Not run"
        gpu_run_t, gpu_transpile_t, avg_gpu_util, avg_mem_used = None, None, None, None
        try:
            sim_gpu = AerSimulator(method="statevector", device="GPU")
            print(f"\nGPU backend initialized: {sim_gpu.status}")
            gpu_run_t, gpu_transpile_t, avg_gpu_util, avg_mem_used = run_and_time_sim(sim_gpu, "GPU", n_qubits, depth)
            sim_gpu_status = "Success"

        except AerError as err:
            sim_gpu_status = f"FAILED: {err}"
            print(f"Aer GPU backend initialization FAILED for N={n_qubits}: {err}")
            print("This indicates a severe problem with the qiskit-aer-gpu installation/compatibility.")
        except Exception as e:
            sim_gpu_status = f"ERROR: {e}"
            print(f"An unexpected error occurred during Aer GPU test for N={n_qubits}: {e}")

        # --- Summary for current qubit count ---
        speedup = None
        if gpu_run_t is not None and cpu_run_t is not None:
            speedup = cpu_run_t / gpu_run_t
            print(f"\n=== Summary for N={n_qubits} (Simulation Run Time Only) ===")
            print(f"CPU Run: {cpu_run_t:7.3f} s   GPU Run: {gpu_run_t:7.3f} s   → speed-up ×{speedup:.2f}")
            print(f"CPU Transpile: {cpu_transpile_t:7.3f} s   GPU Transpile: {gpu_transpile_t:7.3f} s")

            if speedup > 1.0:
                print("STATUS: GPU Run is FASTER than CPU Run. GPU backend is working for simulation.")
            elif speedup > 0.5:
                print("STATUS: GPU Run is SLIGHTLY SLOWER or similar to CPU Run. Overhead may still be present.")
            else:
                print("STATUS: GPU Run is SIGNIFICANTLY SLOWER than CPU Run. This is unexpected for high N and high GPU util.")
        else:
            print(f"\n=== Summary for N={n_qubits} ===\nCPU Run: {cpu_run_t:7.3f} s   GPU backend not available for comparison.")

        # Store results for DataFrame
        results_data.append({
            'N_Qubits': n_qubits,
            'CPU_Run_Time_s': cpu_run_t,
            'CPU_Transpile_Time_s': cpu_transpile_t,
            'GPU_Run_Time_s': gpu_run_t,
            'GPU_Transpile_Time_s': gpu_transpile_t,
            'Speedup_Factor': speedup,
            'Avg_GPU_Util_percent': avg_gpu_util,
            'Avg_Mem_Used_MB': avg_mem_used,
            'GPU_Status': sim_gpu_status
        })

    except ImportError:
        print(f"Qiskit Aer not installed for N={n_qubits}. Skipping this iteration.")
        results_data.append({'N_Qubits': n_qubits, 'GPU_Status': 'Qiskit Aer not installed'})
    except Exception as e:
        print(f"An unexpected error occurred during N={n_qubits} setup: {e}")
        results_data.append({'N_Qubits': n_qubits, 'GPU_Status': f'Unexpected Error: {e}'})

print("\n"+"="*50)
print(" VERIFICATION COMPLETE - SUMMARY TABLE")
print("="*50)

# Print results in a nice table
try:
    results_df = pd.DataFrame(results_data)
    # Order columns nicely
    results_df = results_df[[
        'N_Qubits', 'CPU_Run_Time_s', 'GPU_Run_Time_s', 'Speedup_Factor',
        'CPU_Transpile_Time_s', 'GPU_Transpile_Time_s',
        'Avg_GPU_Util_percent', 'Avg_Mem_Used_MB', 'GPU_Status'
    ]]
    print(results_df.to_markdown(index=False, floatfmt=".3f"))
    # Save to CSV for easy analysis outside
    results_df.to_csv("gpu_benchmark_results.csv", index=False)
    print("\nResults also saved to gpu_benchmark_results.csv")
except Exception as e:
    print(f"Error generating summary table: {e}")

print("\n"+"="*50)
print(" END OF SCRIPT")
print("="*50)