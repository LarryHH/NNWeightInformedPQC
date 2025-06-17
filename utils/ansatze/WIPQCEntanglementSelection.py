"""
Weight‑informed entangler selection
===================================
Utility functions to convert a classical NN weight matrix into a set of
qubit–pair entanglers for a parametrised quantum circuit (PQC).

Pipeline (all optional & composable):

0.  **Correlation matrix** – choose how to measure relationship between
    weight columns (``|cosine|`` or ``|WᵀW|``).
1.  **Edge selection** – decide which qubit pairs will be entangled:
    original greedy, maximum‑weight spanning tree (MWST) or top‑*k* edges.
2.  **Direction rule** – if using directional gates (e.g. CNOT), orient
    each pair via column‑norm or correlation centrality.

Every step is pure NumPy; import only the pieces you need.
"""

from __future__ import annotations

from typing import List, Tuple, Literal, Optional
import numpy as np

# ---------------------------------------------------------------------------
# Correlation matrices
# ---------------------------------------------------------------------------

def _cosine_corr(W: np.ndarray) -> np.ndarray:
    """Return the matrix of absolute cosine similarities of **columns**.

    Parameters
    ----------
    W : ndarray, shape (d, n)
        Weight matrix whose *columns* correspond to qubits.

    Returns
    -------
    C : ndarray, shape (n, n)
        ``C[i, k]`` = ``|cos( W_·,i , W_·,k )|`` with a zero diagonal.
    """
    norm = np.linalg.norm(W, axis=0, keepdims=True) + 1e-12
    C = np.abs(W.T @ W) / (norm.T @ norm)
    np.fill_diagonal(C, 0.0)
    return C


def _gram_abs(W: np.ndarray) -> np.ndarray:
    """Absolute Gram matrix ``|WᵀW|`` with a zero diagonal."""
    C = np.abs(W.T @ W)
    np.fill_diagonal(C, 0.0)
    return C

# ---------------------------------------------------------------------------
# Lightweight Union–Find for Kruskal MST
# ---------------------------------------------------------------------------

def _union_find(n: int):
    """Return ``find`` and ``union`` closures for *n* disjoint‑set items."""

    parent = list(range(n))
    rank = [0] * n

    def find(x: int) -> int:  # path‑compression
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> bool:
        """Merge the sets of *a* and *b*; return ``False`` if already same."""
        ra, rb = find(a), find(b)
        if ra == rb:
            return False  # would form a cycle
        if rank[ra] < rank[rb]:
            parent[ra] = rb
        elif rank[ra] > rank[rb]:
            parent[rb] = ra
        else:
            parent[rb] = ra
            rank[ra] += 1
        return True

    return find, union

# ---------------------------------------------------------------------------
# Pair‑selection strategies
# ---------------------------------------------------------------------------

def _pairs_greedy(W: np.ndarray, top_k: int | None) -> List[Tuple[int, int]]:
    """Greedy heuristic. Picks up to *n* strongest off‑diagonal entries of ``|WᵀW|``.
    Each edge weight is printed for transparency.
    """
    n = W.shape[1]
    M = _gram_abs(W)
    flat = sorted(
        [(M[i, k], i, k) for i in range(n) for k in range(i + 1, n)],
        key=lambda e: e[0],
        reverse=True,
    )
    if top_k is None:             
        top_k = min(n, len(flat))
    for w, i, k_ in flat[:top_k]:
        print(f"Entangler pair ({i},{k_}) strength {w:.3f}")
    return [(i, k_) for _, i, k_ in flat[:top_k]]


def _pairs_mwst(C: np.ndarray) -> List[Tuple[int, int]]:
    """Maximum‑weight spanning tree on the complete graph with weights *C*.

    Guarantees acyclic connectivity with the strongest possible links, with exactly ``n_qubits-1`` edges. (implication is connectivity is good)
    """
    n = C.shape[0]
    edges = sorted(
        [(C[i, k], i, k) for i in range(n) for k in range(i + 1, n)],
        key=lambda e: e[0],
        reverse=True,
    )
    find, union = _union_find(n)
    tree: List[Tuple[int, int]] = []
    for w, i, k in edges:
        if union(i, k):
            tree.append((i, k))
            if len(tree) == n - 1:
                break
    return tree



# ---------------------------------------------------------------------------
# Direction heuristics (for CX)
# ---------------------------------------------------------------------------

def _direction_norm(W: np.ndarray, pairs: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Orient pairs so the column with larger L2‑norm becomes **control**."""
    s = np.linalg.norm(W, axis=0)
    return [(i, k) if s[i] >= s[k] else (k, i) for i, k in pairs]


def _direction_centrality(C: np.ndarray, pairs: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Orient pairs using node centrality ``∑|C_{ij}|`` as the score."""
    d = C.sum(axis=1)
    return [(i, k) if d[i] >= d[k] else (k, i) for i, k in pairs]

# ---------------------------------------------------------------------------
# User‑facing front‑end
# ---------------------------------------------------------------------------

Correlation   = Literal["gram", "cosine"]
Selection     = Literal["greedy", "mwst"]
DirectionRule = Optional[Literal["norm", "centrality"]]


def select_entanglers(
    W: np.ndarray,
    *,
    correlation: Correlation = "gram",
    selection: Selection = "greedy",
    top_k: int | None = None,
    direction: DirectionRule = "norm",
) -> List[Tuple[int, int]]:
    """Return a list of entangler pairs derived from weight matrix *W*.

    Parameters
    ----------
    W : ndarray, shape (d, n)
        Classical weight matrix (columns correspond to qubits).
    correlation : {"cosine", "gram"}, default "cosine"
        Metric used to build the qubit‑correlation graph.
    selection : {"greedy", "mwst", "topk"}, default "mwst"
        Strategy for deciding *which* pairs to keep.
    top_k : int, default 4
        Number of edges to keep when *selection* == "topk".
    direction : None or {"norm", "centrality"}, default "norm"
        Heuristic for orienting each pair (only for directional gates).
        ``None`` returns undirected pairs suitable for CZ/RZZ.

    Returns
    -------
    list of tuple(int, int)
        Ordered pairs (control, target) if *direction* is not ``None``.
        Otherwise, unordered pairs (i, k).
    """
    # -- build correlation matrix ------------------------------------------------
    if correlation == "cosine":
        C = _cosine_corr(W)
    elif correlation == "gram":
        C = _gram_abs(W)
    else:
        raise ValueError("correlation must be 'cosine' or 'gram'")

    # -- select edges ------------------------------------------------------------
    if selection == "greedy":
        pairs = _pairs_greedy(W, top_k=top_k)
    elif selection == "mwst":
        pairs = _pairs_mwst(C)
    else:
        raise ValueError("selection must be 'greedy' or 'mwst'")

    # -- optionally orient -------------------------------------------------------
    if direction is None:
        return pairs
    if direction == "norm":
        return _direction_norm(W, pairs)
    if direction == "centrality":
        return _direction_centrality(C, pairs)
    raise ValueError("direction must be None, 'norm' or 'centrality'")


# ---------------------------------------------------------------------------
# Example usage (delete or comment‑out in production)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    W_demo = np.array(
        [[1, 2, 3, 4],
         [2, 5, 6, 7],
         [3, 6, 8, 9],
         [4, 7, 9, 10]],
        dtype=np.float32,
    )

    print("MWST + norm direction:")
    print(select_entanglers(W_demo))

    print("Top‑3 undirected:")
    print(select_entanglers(W_demo, selection="mwst", top_k=3, direction=None))
