#!/usr/bin/env python3
"""
GPU-vs-CPU sanity check for Qiskit Aer – with profiling.
Modified to capture in-run GPU utilization for remote diagnosis.
"""
import os, time, cProfile, pstats, contextlib, subprocess, threading
from qiskit import transpile
from qiskit.circuit.random import random_circuit
from qiskit_aer import AerSimulator, AerError

# Attempt to import pynvml, handle gracefully if not present
try:
    import pynvml

    _pynvml_available = True
    pynvml.nvmlInit()  # Initialize NVML once
except (ModuleNotFoundError, AttributeError):
    _pynvml_available = False
    print(
        "Warning: pynvml not found or failed to initialize. GPU utilization monitoring will be skipped."
    )


# ------------------------------------------------------------
# 1.  NVML helpers (gracefully skipped if pynvml absent)
# ------------------------------------------------------------
def _get_gpu_stats():
    if not _pynvml_available:
        return None, None
    try:
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        mem = pynvml.nvmlDeviceGetMemoryInfo(h)
        util = pynvml.nvmlDeviceGetUtilizationRates(h)
        return mem.used / 2**20, util.gpu  # MB, %
    except pynvml.NVMLError as error:
        print(f"NVML Error: {error}")
        return None, None
    except Exception as e:
        print(f"Error getting NVML stats: {e}")
        return None, None


@contextlib.contextmanager
def gpu_monitor_initial_final(label):
    # This context manager is for before/after snapshots
    m0, u0 = _get_gpu_stats()
    if m0 is not None:
        print(f"[NVML] {label} – before: {m0:,.0f} MB, util {u0}%")
    yield
    m1, u1 = _get_gpu_stats()
    if m1 is not None:
        print(f"[NVML] {label} – after : {m1:,.0f} MB, util {u1}%\n")


# New: Function to monitor GPU during execution
def _monitor_gpu_in_background(stop_event, interval=0.1):
    gpu_util_history = []
    mem_util_history = []
    print("\n[NVML Monitor] Starting in-run monitoring...")
    while not stop_event.is_set():
        m, u = _get_gpu_stats()
        if m is not None and u is not None:
            gpu_util_history.append(u)
            mem_util_history.append(m)
        time.sleep(interval)
    print("[NVML Monitor] Stopping in-run monitoring.")
    return gpu_util_history, mem_util_history


# ------------------------------------------------------------
# 2.  Check installation
# ------------------------------------------------------------
print("\n=== Aer installation check ===")

sim_cpu = AerSimulator(method="statevector", device="CPU")
sim_gpu = None  # Initialize to None
try:
    sim_gpu = AerSimulator(method="statevector", device="GPU")
    print(f"GPU Backend found: {sim_gpu.status}")
    print(f"Aer version: {sim_gpu.version}")
    # Also check if Aer explicitly says it's using CUDA
    try:
        from qiskit_aer.backends.aer_simulator import AerSimulator

        print(
            f"Aer default device for statevector: {AerSimulator.default_options().device}"
        )
    except Exception:
        pass  # Older Aer versions might not have default_options
except AerError as err:
    print(f"⚠️  GPU backend NOT found: {err}")
    print("   Please ensure qiskit-aer-gpu (or -cu11) is installed correctly.")
    print("   pip install --force-reinstall qiskit-aer-gpu   # CUDA 12.x")
    print("   pip install --force-reinstall qiskit-aer-gpu-cu11  # CUDA 11.x")
    # Don't exit, allow CPU run to continue if GPU fails
except Exception as e:
    print(f"An unexpected error occurred during GPU backend initialization: {e}")
    # Don't exit, allow CPU run to continue if GPU fails

# Add a check for PyTorch CUDA visibility
try:
    import torch

    print(f"\n=== PyTorch CUDA Check ===")
    print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"PyTorch CUDA device count: {torch.cuda.device_count()}")
        if torch.cuda.device_count() > 0:
            print(f"PyTorch CUDA current device: {torch.cuda.current_device()}")
            print(f"PyTorch CUDA device name: {torch.cuda.get_device_name(0)}")
            print(f"PyTorch CUDA version: {torch.version.cuda}")
    else:
        print(
            "PyTorch CUDA not available. This might indicate a system-level CUDA issue."
        )
except ImportError:
    print("PyTorch not installed. Skipping PyTorch CUDA check.")

# ------------------------------------------------------------
# 3.  Build an n-qubit test circuit
# ------------------------------------------------------------
n_qubits, depth = 28, 40  # Changed to 28 for testing GPU sweet spot
qc = random_circuit(n_qubits, depth, measure=False, seed=0)
# print(qc.draw(idle_wires=False, fold=-1))


def run_and_time(sim, label, n_qubits_val, depth_val):
    qc_local = random_circuit(n_qubits_val, depth_val, measure=False, seed=0)
    circ = transpile(qc_local, sim)

    print(f"\n--- Running {label} (N={n_qubits_val}, D={depth_val}) ---")

    gpu_util_history = []
    mem_util_history = []
    stop_event = threading.Event()
    monitor_thread = None

    if (
        _pynvml_available and label == "GPU"
    ):  # Only monitor GPU when running GPU backend
        monitor_thread = threading.Thread(
            target=lambda: (
                gpu_util_history.extend(_monitor_gpu_in_background(stop_event)[0]),
                mem_util_history.extend(_monitor_gpu_in_background(stop_event)[1]),
            )
        )
        monitor_thread.start()

    with gpu_monitor_initial_final(
        label
    ):  # This context handles initial/final NVML snapshot
        pr = cProfile.Profile()
        pr.enable()
        t0 = time.perf_counter()
        job = sim.run(circ)
        res = job.result()  # Make sure to get the result to ensure full computation
        dt = time.perf_counter() - t0
        pr.disable()

    if monitor_thread:
        stop_event.set()
        monitor_thread.join()
        # Process and print in-run utilization
        if gpu_util_history:
            avg_gpu_util = sum(gpu_util_history) / len(gpu_util_history)
            max_gpu_util = max(gpu_util_history)
            print(
                f"[NVML In-Run] Avg GPU Util: {avg_gpu_util:.1f}%, Max GPU Util: {max_gpu_util:.1f}%"
            )
        if mem_util_history:
            avg_mem_used = sum(mem_util_history) / len(mem_util_history)
            print(f"[NVML In-Run] Avg Mem Used: {avg_mem_used:,.0f} MB")

    print(f"{label:4s} elapsed : {dt:7.3f} s   backend={job.backend().name}")
    pstats.Stats(pr).strip_dirs().sort_stats("cumtime").print_stats(6)
    return dt


print("\n=== Timing & profiling ===")
cpu_t = run_and_time(sim_cpu, "CPU", n_qubits, depth)
if sim_gpu:  # Only run GPU if it initialized successfully
    gpu_t = run_and_time(sim_gpu, "GPU", n_qubits, depth)
else:
    gpu_t = None

# ------------------------------------------------------------
# 4.  Summary
# ------------------------------------------------------------
if gpu_t is not None:
    speedup = cpu_t / gpu_t
    print(
        f"\n=== Summary ===\nCPU {cpu_t:7.3f} s   GPU {gpu_t:7.3f} s   → speed-up ×{speedup:.2f}"
    )
else:
    print(f"\n=== Summary ===\nCPU {cpu_t:7.3f} s   GPU backend not available.")

print("\n--- Troubleshooting Guide ---")
print(
    "1. If 'Avg GPU Util' is low (<50-60%) during the GPU run, the GPU is not being effectively used."
)
print("2. Check 'Aer default device for statevector:' above. It should be 'GPU'.")
print("3. Ensure 'PyTorch CUDA available' is True and device name matches your V100.")
print(
    "4. Verify your system's CUDA Toolkit version matches the qiskit-aer-gpu package (e.g., cu11 or default for CUDA 12.x)."
)
print(
    "5. For N=8-24, overhead might dominate. For N=30+, memory limits can occur on 32GB V100."
)
print("   Try N_qubits = 28 or 29 to test the optimal GPU sweet spot on a 32GB V100.")

# Final NVML shutdown
if _pynvml_available:
    pynvml.nvmlShutdown()
