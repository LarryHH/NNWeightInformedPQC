#!/usr/bin/env python3
"""
GPU-vs-CPU sanity check for Qiskit Aer – with profiling.
"""
import os, time, cProfile, pstats, contextlib, subprocess
from qiskit import transpile
from qiskit.circuit.random import random_circuit
from qiskit_aer import AerSimulator, AerError

# ------------------------------------------------------------
# 0.  OPTIONAL ─ watch nvidia-smi in another terminal:
#       watch -n 1 nvidia-smi
# ------------------------------------------------------------

# ------------------------------------------------------------
# 1.  NVML helpers (gracefully skipped if pynvml absent)
# ------------------------------------------------------------
def _gpu_stats():
    try:
        import pynvml
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        mem = pynvml.nvmlDeviceGetMemoryInfo(h)
        util = pynvml.nvmlDeviceGetUtilizationRates(h)
        return mem.used / 2**20, util.gpu          # MB, %
    except (ModuleNotFoundError, AttributeError):
        print("⚠️  pynvml not installed – GPU stats unavailable.")
        return None, None

@contextlib.contextmanager
def gpu_monitor(label):
    m0, u0 = _gpu_stats()
    if m0 is not None:
        print(f"[NVML] {label} – before: {m0:,.0f} MB, util {u0}%")
    yield
    m1, u1 = _gpu_stats()
    if m1 is not None:
        print(f"[NVML] {label} – after : {m1:,.0f} MB, util {u1}%\n")

# ------------------------------------------------------------
# 2.  Check installation
# ------------------------------------------------------------
print("\n=== Aer installation check ===")

sim_cpu = AerSimulator(method="statevector", device="CPU")
try:
    sim_gpu = AerSimulator(method="statevector", device="GPU")   # ⇐ forces GPU path :contentReference[oaicite:0]{index=0}
except AerError as err:
    sim_gpu = None
    print("⚠️  GPU backend NOT found – reinstall with:")
    print("   pip install --force-reinstall qiskit-aer-gpu   # CUDA 12.x")  # :contentReference[oaicite:1]{index=1}
    print("   pip install --force-reinstall qiskit-aer-gpu-cu11  # CUDA 11.x")
    exit(1)

# ------------------------------------------------------------
# 3.  Build an n-qubit test circuit
# ------------------------------------------------------------
n_qubits, depth = 35, 40
qc = random_circuit(n_qubits, depth, measure=False, seed=0)
# print(qc.draw(idle_wires=False, fold=-1))

def run_and_time(sim, label):
    circ = transpile(qc, sim)
    with gpu_monitor(label):            # NVML snapshot
        pr = cProfile.Profile()
        pr.enable()
        t0 = time.perf_counter()
        job = sim.run(circ)
        res = job.result()
        dt = time.perf_counter() - t0
        pr.disable()
    print(f"{label:4s} elapsed : {dt:7.3f} s   backend={job.backend().name}")
    pstats.Stats(pr).strip_dirs().sort_stats("cumtime").print_stats(6)
    return dt

print("\n=== Timing & profiling ===")
cpu_t = run_and_time(sim_cpu, "CPU")
gpu_t = run_and_time(sim_gpu, "GPU")

# ------------------------------------------------------------
# 4.  Summary
# ------------------------------------------------------------
speedup = cpu_t / gpu_t
print(f"\n=== Summary ===\nCPU {cpu_t:7.3f} s   GPU {gpu_t:7.3f} s   → speed-up ×{speedup:.2f}")
print("If NVML shows zero util or the backend falls back to CPU, the GPU build is missing.")

