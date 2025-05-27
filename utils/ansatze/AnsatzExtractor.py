import os
import json
from pathlib import Path
from typing import Tuple

from qiskit import QuantumCircuit, qpy

try:
    from .GenericAnsatz import GenericAnsatz
except ImportError:
    from GenericAnsatz import GenericAnsatz

def extract_and_store_model_schema(ansatz: GenericAnsatz, model, circuit_fp, params_fp):
    """Extract and store model schema in a text file."""

    circuit = ansatz.get_ansatz()
    num_qubits = ansatz.get_num_qubits()
    tensor = next(model.model.parameters())
    param_values = tensor.detach().cpu().tolist()

    gates = dict(circuit.get_ansatz().count_ops())
    entangling_gates = ["cx", "cz", "swap", "ccx", "cswap"]
    num_parameterized_gates = len(param_values)
    num_entangling_gates = sum(gates[gate] for gate in entangling_gates if gate in gates)

    with open(circuit_fp, "wb") as file:
        qpy.dump(circuit, file)
    print(f"Model schema saved to {circuit_fp}")

    metadata = {
        "ansatz_name": ansatz.get_name(),
        "num_qubits": num_qubits,
        "parameters": param_values,
        "num_parameterized_gates": num_parameterized_gates,
        "num_entangling_gates": num_entangling_gates,
    }

    with open(params_fp, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Model parameters saved to {params_fp}")


def load_ansatz_from_model_schema(
    circuit_fp: str, params_fp: str
) -> Tuple[QuantumCircuit, dict]:
    """Load ansatz from a file."""

    with open(circuit_fp, "rb") as handle:
        qc = qpy.load(handle)
    with open(params_fp, "r") as f:
        metadata = json.load(f)

    ansatz_circ = QuantumCircuit(metadata["num_qubits"])
    ansatz_circ.compose(qc[0], inplace=True)
    return ansatz_circ, metadata
