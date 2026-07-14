"""
COSMO‑SAC 2002 (MVP, pure‑Python) — ThermoBox‑ready
====================================================
Minimal, readable implementation following Lin & Sandler (2002), eqs. (7), (10)-(12), (14),
with a pragmatic SG combinatorial term. Focus: correctness + clarity first; optimization later.

What this file does now
-----------------------
- read `.sigma` with two columns (sigma, dA) and normalize to p_i(σ) = dA/A_i
- build p_mix(σ) by the exact area‑weighted formula
- compute ΔW(σm,σn) (misfit + H‑bond) and solve lnΓ(σ) via the fixed‑point iteration
- compute lnγ_i,res from eq. (12) with n_i = A_i/aeff
- compute lnγ_i,SG (Staverman‑Guggenheim) with a safe default r_i≈q_i (can be replaced later)
- return lnγ_i = lnγ_i,res + lnγ_i,SG

Notes
-----
• Units follow the paper: energies in kcal/mol, σ in e/Å², areas in Å². R (gas const) in kcal/mol/K.
• a_eff = 7.50 Å², σ_hb = 0.0084 e/Å², c_hb = 85580 (kcal/mol·Å⁴/e²).
• R′ for misfit: Rprime = (0.64 * 0.3 * a_eff**1.5) / eps0, eps0 = 2.395e-4 (e²·mol)/(kcal·Å).
• The SG term uses q_i = A_i/79.53 and r_i ≈ q_i if V_i unknown; pass actual V_i to improve.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Iterable, Optional, Tuple, Dict
import numpy as np
import math, os

# ------------------ constants (paper values) ------------------
AEFF = 7.50                 # Å^2
SIGMA_HB = 0.0084           # e/Å^2
CHB = 85580.0               # kcal/mol * Å^4 / e^2
# EPS0 = 2.395e-4             # (e^2 mol)/(kcal Å)
R_KCAL = 0.00198720425864   # kcal/mol/K
# Misfit prefactor R′ (kcal/mol * Å^4 / e^2) — derived from a_eff and eps0
RPRIME = 16466.72  # kcal·Å^4/(mol·e^2)

# ---- Canonical sigma grid (51 bins, -0.025..+0.025) ----
SIGMA_MIN, SIGMA_MAX, NBINS = -0.025, 0.025, 51
SIGMA_GRID = np.linspace(SIGMA_MIN, SIGMA_MAX, NBINS)

def _build_deltaW():
    s = SIGMA_GRID
    sm = s.reshape(-1, 1)
    sn = s.reshape(1, -1)
    sigma_acc = np.maximum(sm, sn)
    sigma_don = np.minimum(sm, sn)
    misfit = 0.5 * RPRIME * (sm + sn) ** 2
    hb = CHB * np.maximum(0.0, sigma_acc - SIGMA_HB) * np.minimum(0.0, sigma_don + SIGMA_HB)
    return misfit + hb

_DELTAW = _build_deltaW()

# ------------------ data structures ------------------
@dataclass
class SigmaProfile:
    name: str
    sigma: np.ndarray   # (nbins,) in e/Å^2
    dA: np.ndarray      # (nbins,) in Å^2
    A_total: Optional[float] = None  # if None, computed from dA
    V_vdw: Optional[float] = None    # Å^3 (optional, for SG term)

    def __post_init__(self):
        self.sigma = np.asarray(self.sigma, float)
        self.dA = np.asarray(self.dA, float)
        assert self.sigma.shape == self.dA.shape, "sigma and dA must match"

    @property
    def A(self) -> float:
        return float(self.A_total if self.A_total is not None else self.dA.sum())

    @property
    def p(self) -> np.ndarray:
        A = self.A
        if A <= 0: raise ValueError(f"Profile {self.name} has zero/negative area")
        return self.dA / A


# ------------------ I/O ------------------
def read_sigma_file(path: str, name: Optional[str] = None) -> SigmaProfile:
    sig, dA = [], []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith('#') or s.startswith(';'):
                continue
            parts = s.replace(',', ' ').split()
            if len(parts) < 2: continue
            try:
                sig.append(float(parts[0])); dA.append(float(parts[1]))
            except ValueError:
                continue
    if not sig: raise ValueError(f"No numeric data read from {path}")
    return SigmaProfile(name or os.path.basename(path), np.array(sig), np.array(dA))


# ------------------ core math ------------------
def _deltaW_matrix(sigma: np.ndarray) -> np.ndarray:
    """Build ΔW(σm,σn) for all m,n (kcal/mol). Vectorized.
    ΔW = 1/2 R′ (σm+σn)^2 + CHB*max(0, σ_acc-σ_hb)*min(0, σ_don+σ_hb)
    """
    s = sigma.reshape(-1,1)
    sp = s + s.T
    misfit = 0.5 * RPRIME * (sp ** 2)
    # HB term needs σ_acc (max) and σ_don (min) per pair
    sigma_acc = np.maximum(s, s.T)
    sigma_don = np.minimum(s, s.T)
    hb = CHB * np.maximum(0.0, sigma_acc - SIGMA_HB) * np.minimum(0.0, sigma_don + SIGMA_HB)
    return misfit + hb


def _solve_lnGamma(p_sigma: np.ndarray, T: float,
                   *, max_iter: int = 500, tol: float = 1e-11, damp: float = 0.5) -> np.ndarray:
    p = np.asarray(p_sigma, float); p = p / p.sum()
    lnG = np.zeros_like(p)
    beta = 1.0 / (R_KCAL * T)
    # Precompute logK once (numeric stability)
    logK = -beta * _DELTAW
    logp = np.log(p + 1e-300)

    for _ in range(max_iter):
        # ln Γ_m = - ln Σ_n p_n Γ_n exp(-ΔW_mn/RT)
        logsum = np.empty_like(p)
        for m in range(p.size):
            a = logp + lnG + logK[m]
            a_max = a.max()
            logsum[m] = a_max + np.log(np.exp(a - a_max).sum())
        lnG_new = -logsum
        delta = lnG_new - lnG
        lnG += damp * delta
        if np.max(np.abs(delta)) < tol:
            lnG = lnG_new
            break
    return lnG


def _p_mix(profiles: List[SigmaProfile], x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return (sigma_grid, p_mix). Assumes all profiles share same σ grid; if not, warns.
    p_mix(σ) = [Σ_i x_i dA_i(σ)] / [Σ_i x_i A_i]
    """
    sig0 = profiles[0].sigma
    # (Optionally rebin here if grids differ — omitted for MVP)
    num = np.zeros_like(sig0)
    denom = 0.0
    for xi, pi in zip(x, profiles):
        num += xi * pi.dA
        denom += xi * pi.A
    if denom <= 0: raise ValueError("Mixture denominator zero")
    p_mix = num / denom
    return sig0, p_mix


def _ln_gamma_residual(profiles: List[SigmaProfile], x: np.ndarray, T: float) -> np.ndarray:
    sigma, pS = _p_mix(profiles, x)
    lnG_S = _solve_lnGamma(pS, T)

    lng_res = np.zeros(len(profiles))
    for i, prof in enumerate(profiles):
        p_i = prof.p
        lnG_i = _solve_lnGamma(p_i, T)
        n_i = prof.A / AEFF
        lng_res[i] = n_i * float((p_i * (lnG_S - lnG_i)).sum())
    return lng_res


def _ln_gamma_SG(profiles: List[SigmaProfile], x: np.ndarray) -> np.ndarray:
    """Staverman‑Guggenheim combinatorial term (safe default).
    q_i = A_i/79.53 ; r_i ≈ q_i if V_i unknown; z=10.0
    """
    A_STD = 79.53
    V_STD = 66.69
    z = 10.0
    x = x / x.sum()

    q = np.array([p.A / A_STD for p in profiles])
    r = np.array([(p.V_vdw / V_STD) if (p.V_vdw is not None) else (p.A / A_STD) for p in profiles])

    theta = x * q / max(1e-300, float((x * q).sum()))
    phi   = x * r / max(1e-300, float((x * r).sum()))

    li = (z/2.0) * (r - q) - (r - 1.0)
    # SG formula (eq. 3): ln γ^SG = ln(φ_i/x_i) + (z/2) q_i ln(θ_i/φ_i) + l_i - (φ_i/x_i) Σ x_j l_j
    eps = 1e-300
    term = np.log((phi + eps) / (x + eps)) + (z/2.0)*q * np.log((theta + eps)/(phi + eps)) + li
    lij = float((x * li).sum())
    return term - (phi / (x + eps)) * lij


# ------------------ public API ------------------
def lngamma_2002_from_paths(paths: Iterable[str], x: Iterable[float], T: float,
                            names: Optional[Iterable[str]] = None,
                            volumes: Optional[Dict[str, float]] = None) -> Tuple[np.ndarray, List[SigmaProfile]]:
    """Convenience: pass `.sigma` paths and mole fractions, get lnγ array.
       volumes: optional dict of V_vdw (Å^3) keyed by name to improve SG term.
    """
    paths = list(paths)
    names = list(names) if names is not None else [os.path.basename(p) for p in paths]
    x = np.asarray(list(x), float)
    assert len(paths) == len(x) == len(names)

    profs = [read_sigma_file(p, n) for p, n in zip(paths, names)]
    if volumes:
        for p in profs:
            if p.name in volumes: p.V_vdw = float(volumes[p.name])

    lng_res = _ln_gamma_residual(profs, x, T)
    lng_sg  = _ln_gamma_SG(profs, x)
    return lng_res + lng_sg


def lngamma_2002(profiles: List[SigmaProfile], x: Iterable[float], T: float) -> np.ndarray:
    """Main entry if you already have SigmaProfile objects."""
    x = np.asarray(list(x), float)
    if x.sum() <= 0: raise ValueError("x sums to zero")
    return _ln_gamma_residual(profiles, x, T) + _ln_gamma_SG(profiles, x)


# ------------------ quick self‑test ------------------
if __name__ == "__main__":
    # Drop a couple of `.sigma` files in the working dir to try this quickly
    # e.g. WATER.sigma and ACETONE.sigma sharing the same σ grid
    try:
        paths = [f for f in os.listdir('.') if f.lower().endswith('.sigma')][:2]
        if len(paths) == 2:
            lng, profs = lngamma_2002_from_paths(paths, x=[0.3, 0.7], T=298.15)
            print("components:", [p.name for p in profs])
            print("lngamma:", lng)
        else:
            print("Place two .sigma files here to run the demo.")
    except Exception as e:
        print("demo error:", e)
