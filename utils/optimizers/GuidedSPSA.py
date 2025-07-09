import numpy as np
from qiskit_machine_learning.gradients import (
    BaseSamplerGradient,
    ParamShiftSamplerGradient,
    SPSASamplerGradient,
    SamplerGradientResult,
)
from qiskit.circuit import Parameter    # only for typing

def _sample_norm(sample_grad):
    """Euclidean norm of one sample (list[dict])."""
    s = 0.0
    for pdict in sample_grad:
        for v in pdict.values():
            s += float(v) ** 2
    return np.sqrt(s)

def _scale_sample(sample_grad, factor):
    """Multiply every entry by <factor> in-place."""
    for pdict in sample_grad:
        for k in pdict:
            pdict[k] *= factor


class GuidedSPSASamplerGradient(BaseSamplerGradient):
    """
    Guided-SPSA gradient estimator (arXiv:2405.16734, Alg. 2).
    Splits each mini-batch: a τ-fraction is evaluated with parameter-shift,
    the rest with SPSA whose batch-size k is scheduled epoch-wise.
    """
    # ─── public API ──────────────────────────────────────────────────────
    def __init__(
        self,
        sampler,
        *,
        N_epochs: int = 100,
        tau: float = 0.5,
        epsilon: float = 1.0,
        k_min_ratio: float = 0.10,
        k_max_factor: float = 1.5,
        seed: int | None = None,
    ):
        super().__init__(sampler)
        self._ps_grad   = ParamShiftSamplerGradient(sampler)
        self._spsa_grad = SPSASamplerGradient(sampler, batch_size=1, seed=seed)

        self.N_epochs     = N_epochs
        self.tau          = tau
        self.epsilon      = epsilon
        self.k_min_ratio  = k_min_ratio
        self.k_max_factor = k_max_factor

        self.rng    = np.random.default_rng(seed)
        self._epoch = 0

    def step_epoch(self) -> None:
        """Call exactly once at the end of each training epoch."""
        self._epoch += 1

    # ─── BaseSamplerGradient interface ───────────────────────────────────
    def _run(self, circuits, parameter_values, parameter_sets, **options):
        B = len(circuits)                         # mini-batch size
        P = len(next(iter(parameter_sets)))       # |θ|

        # 1 ▪ split indices
        idx = self.rng.permutation(B)
        split = int(np.round(self.tau * B))
        idx_ps, idx_spsa = idx[:split], idx[split:]

        grads_full = [None] * B                   # final container

        # ----- param-shift part -----------------------------------------
        if idx_ps.size:
            ps_res = self._ps_grad._run(
                [circuits[i] for i in idx_ps],
                [parameter_values[i] for i in idx_ps],
                [parameter_sets[i]  for i in idx_ps],
            )
            for i, g in zip(idx_ps, ps_res.gradients):
                grads_full[i] = g

        # ----- SPSA part -------------------------------------------------
        if idx_spsa.size:
            # epoch-wise k-schedule
            k_min = max(1, int(self.k_min_ratio * P))
            k_max = int(P * min(1.0, self.k_max_factor - self.tau))
            gamma = (k_max - k_min) / max(1, self.N_epochs)
            k_now = int(np.floor(k_min + self._epoch * gamma))
            self._spsa_grad.batch_size = k_now

            spsa_res = self._spsa_grad._run(
                [circuits[i] for i in idx_spsa],
                [parameter_values[i] for i in idx_spsa],
                [parameter_sets[i]  for i in idx_spsa],
            )

            # -- scale (Alg-2 step-21) -----------------------------------
            if idx_ps.size:
                sigma = np.mean([_sample_norm(grads_full[i]) for i in idx_ps]) + 1e-12
            else:
                sigma = 1.0

            for i, g in zip(idx_spsa, spsa_res.gradients):
                norm_g = _sample_norm(g) + 1e-12
                factor = self.epsilon * sigma / norm_g
                _scale_sample(g, factor)
                grads_full[i] = g

        # all slots filled?
        assert all(g is not None for g in grads_full), "internal bug"

        metadata = [{"parameters": list(ps)} for ps in parameter_sets]
        return SamplerGradientResult(
            gradients=grads_full,
            metadata=metadata,
            options=self._get_local_options(options),
        )