from qiskit import transpile
from qiskit.circuit.random import random_circuit
from qiskit_aer import AerSimulator
import time, numpy as np

n_qubits   = 24          # state-vector: 2**24 complex64 ≈ 268 MB
depth      = 40
circuit    = random_circuit(n_qubits, depth, measure=False, seed=0)

sim_cpu = AerSimulator(method="statevector", device="CPU")
sim_gpu = AerSimulator(method="statevector", device="GPU")  # needs qiskit-aer-gpu

c_cpu  = transpile(circuit, sim_cpu)
c_gpu  = transpile(circuit, sim_gpu)      # same map, avoids extra transpile cost

for name, sim, circ in [("CPU", sim_cpu, c_cpu), ("GPU", sim_gpu, c_gpu)]:
    t0 = time.perf_counter()
    sim.run(circ).result()                # one shot; no classical read-out cost
    print(f"{name} elapsed: {time.perf_counter()-t0:7.3f} s")