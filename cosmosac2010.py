# -*- coding: utf-8 -*-
"""
COSMO-SAC 2010 (Hsieh–Lin–Sandler) — ThermoBox clean implementation (NIST-compatible)
- Typed Γ fixed-point (NHB, OH, OT)
- Residual + SG combinatorial (no dispersion)
- Canonical 51-bin σ grid; profiles are resampled with area preservation
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Iterable, Optional, Tuple, Dict
import numpy as np
import os, json

# ---------- constants (2010) ----------
R_KCAL = 0.00198720425864   # kcal/mol/K
AEFF    = 7.25              # Å^2
A_STD   = 79.53             # Å^2 (SG)
V_STD   = 66.69             # Å^3 (SG)
Z_SG    = 10.0

# electrostatic parameter (T-dependent)
A_ES = 6525.69              # kcal·Å^4/(mol·e^2)
B_ES = 1.4859e8             # kcal·Å^4·K^2/(mol·e^2)

# HB strengths (kcal·Å^4/(mol·e^2))
C_OH_OH = 4013.78
C_OT_OT =  932.31
C_OH_OT = 3016.43

# canonical σ grid (NIST/COSMO3)
SIGMA_MIN, SIGMA_MAX, NBINS = -0.025, 0.025, 51
SIGMA_GRID = np.linspace(SIGMA_MIN, SIGMA_MAX, NBINS)

# ---------- data ----------
@dataclass
class SigmaProfile2010:
    name: str
    sigma: np.ndarray            # (nbins,)
    dA_nhb: np.ndarray           # (nbins,) area in Å^2
    dA_oh:  np.ndarray           # (nbins,)
    dA_ot:  np.ndarray           # (nbins,)
    A_total: Optional[float] = None
    V_vdw:   Optional[float] = None

    def __post_init__(self):
        self.sigma  = np.asarray(self.sigma , float)
        self.dA_nhb = np.asarray(self.dA_nhb, float)
        self.dA_oh  = np.asarray(self.dA_oh , float)
        self.dA_ot  = np.asarray(self.dA_ot , float)
        n = self.sigma.size
        assert self.dA_nhb.size == n and self.dA_oh.size == n and self.dA_ot.size == n

    @property
    def A(self) -> float:
        # usa a soma dos blocos (robusto aos arquivos sem meta)
        return float(self.dA_nhb.sum() + self.dA_oh.sum() + self.dA_ot.sum())

    # typed p(σ), normalizados pela área total (sem renorm. por tipo)
    @property
    def p_nhb(self) -> np.ndarray: return self.dA_nhb / max(self.A, 1e-300)
    @property
    def p_oh (self) -> np.ndarray: return self.dA_oh  / max(self.A, 1e-300)
    @property
    def p_ot (self) -> np.ndarray: return self.dA_ot  / max(self.A, 1e-300)

# ---------- I/O ----------
def _resample_to_canonical(grid_in: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Conservative, area-preserving remap from (grid_in, y) to (SIGMA_GRID, y_new).
    Treat y as area-per-bin on grid_in with *uniform* bin width inferred from grid_in.
    We redistribute into the canonical 51 uniform bins by overlap length (1D conservative remap).
    """
    grid_in = np.asarray(grid_in, float)
    y = np.asarray(y, float)

    # Fast path: already canonical
    if grid_in.size == SIGMA_GRID.size and np.allclose(grid_in, SIGMA_GRID):
        return y.copy()

    # Build source bin edges from centers (grid_in) assuming uniform spacing near edges
    if grid_in.size < 2:
        raise ValueError("Cannot resample from <2 points")
    dx_in = np.diff(grid_in)
    dx = np.median(dx_in)
    edges_in = np.empty(grid_in.size + 1)
    edges_in[1:-1] = 0.5 * (grid_in[:-1] + grid_in[1:])
    edges_in[0] = grid_in[0] - 0.5 * dx
    edges_in[-1] = grid_in[-1] + 0.5 * dx

    # Target (canonical) bin edges
    dx_out = (SIGMA_MAX - SIGMA_MIN) / (NBINS - 1)
    edges_out = np.empty(NBINS + 1)
    edges_out[1:-1] = 0.5 * (SIGMA_GRID[:-1] + SIGMA_GRID[1:])
    edges_out[0] = SIGMA_GRID[0] - 0.5 * dx_out
    edges_out[-1] = SIGMA_GRID[-1] + 0.5 * dx_out

    # Interpret y as "area contained in each source bin"
    # Distribute to target bins by 1D overlap of intervals.
    y_new = np.zeros(NBINS, dtype=float)
    i = j = 0
    while i < grid_in.size and j < NBINS:
        L = max(edges_in[i], edges_out[j])
        R = min(edges_in[i+1], edges_out[j+1])
        if R > L:  # there is overlap
            frac = (R - L) / (edges_in[i+1] - edges_in[i])
            y_new[j] += y[i] * frac
        # advance whichever bin closes first
        if edges_in[i+1] <= edges_out[j+1] + 1e-15:
            i += 1
        else:
            j += 1

    # No renormalization: area is conserved by construction
    return y_new

def read_sigma3_file(path: str, name: Optional[str] = None) -> SigmaProfile2010:
    """
    Lê `.sigma3` com 3 blocos consecutivos (NHB, OH, OT) ou 1 bloco (NHB).
    Cada linha de dados: 'sigma  pSigmaA' (pSigmaA em Å^2 por bin).
    Linha '# meta: {...}' pode conter 'name', 'area [A^2]', 'volume [A^3]'.
    Reamostra cada bloco para a grade canônica (51 bins) preservando área.
    """
    sig, vals = [], []
    meta: Dict[str, float] = {}
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.startswith('#'):
                if s.startswith('# meta'):
                    js = s.split(':', 1)[1].strip()
                    i, j = js.find('{'), js.rfind('}')
                    if i >= 0 and j > i:
                        try:
                            meta = json.loads(js[i:j+1])
                        except Exception:
                            meta = {}
                continue
            parts = s.split()
            if len(parts) >= 2:
                sig.append(float(parts[0])); vals.append(float(parts[1]))
    if not sig:
        raise ValueError(f"No numeric data in {path}")

    nrows = len(vals)
    nb = NBINS
    if nrows % nb != 0:
        raise ValueError(f"Unexpected row count in {path}: {nrows}")
    nblocks = nrows // nb
    if nblocks not in (1, 3):
        raise ValueError(f"Expected 1 or 3 blocks; got {nblocks} in {path}")

    grid_in = np.array(sig[:nb])
    arr = np.array(vals, float).reshape(nblocks, nb)

    if nblocks == 1:
        dA_nhb_in, dA_oh_in, dA_ot_in = arr[0], np.zeros(nb), np.zeros(nb)
    else:
        dA_nhb_in, dA_oh_in, dA_ot_in = arr[0], arr[1], arr[2]

    # reamostra para a grade canônica
    dA_nhb = _resample_to_canonical(grid_in, dA_nhb_in)
    dA_oh  = _resample_to_canonical(grid_in, dA_oh_in)
    dA_ot  = _resample_to_canonical(grid_in, dA_ot_in)

    prof = SigmaProfile2010(
        name or meta.get('name') or os.path.basename(path),
        SIGMA_GRID.copy(), dA_nhb, dA_oh, dA_ot,
        A_total=float(meta.get('area [A^2]', 0.0)) or None,
        V_vdw=float(meta.get('volume [A^3]', 0.0)) or None,
    )
    return prof

# ---------- core math ----------
def _cES(T: float) -> float:
    return A_ES + B_ES / (T*T)

def _deltaW_blocks(sigma: np.ndarray, T: float) -> np.ndarray:
    """
    Constrói os nove blocos ΔW^{t,s}_{m,n} (t linhas: NHB/OH/OT, s colunas: NHB/OH/OT).
    ΔW^{t,s} = c_ES(T)*(σm+σn)^2 − c_HB^{t,s}*1_{σmσn<0}*(σm−σn)^2
    HB somente quando o ALVO t é OH ou OT (NHB nunca recebe HB).
    """
    # usa exatamente a grade canônica (paridade com NIST)
    s_grid = SIGMA_GRID
    s  = s_grid.reshape(-1, 1)               # (51,1)
    sp = s + s.T                             # (51,51)
    sm = s - s.T                             # (51,51)
    base = _cES(T) * (sp**2)
    opp  = (s * s.T) < 0.0

    def hb(coeff: float) -> np.ndarray:
        return base - coeff * (sm**2) * opp

    DW = np.empty((3,3, NBINS, NBINS), dtype=float)
    # t = NHB → sem HB, qualquer que seja a fonte
    DW[0,0] = base; DW[0,1] = base;         DW[0,2] = base
    # t = OH
    DW[1,0] = base; DW[1,1] = hb(C_OH_OH);  DW[1,2] = hb(C_OH_OT)
    # t = OT
    DW[2,0] = base; DW[2,1] = hb(C_OH_OT);  DW[2,2] = hb(C_OT_OT)
    return DW

def _solve_lnGamma_typed(p_s: List[np.ndarray], sigma: np.ndarray, T: float,
                                     *, max_iter: int = 400, tol: float = 1e-11, damp: float = 0.5
                                     ) -> List[np.ndarray]:
    """
    Variant of the COSMO-SAC 2010 segment fixed-point using Γ^s on the RHS:

        ln Γ^t_m = - ln sum_s sum_n [ p_mix^s(n) * Γ^s(n) * exp(-ΔW^{t,s}_{m,n}/RT) ]

    • p_s: [p^NHB, p^OH, p^OT], each shape (nbins,)
    • sigma: bin centers (unused here except to get nbins)
    • Returns [lnΓ^NHB, lnΓ^OH, lnΓ^OT], each (nbins,)
    """
    nb = int(sigma.size)

    # Trust upstream normalization: global area normalization already applied
    p_s = [np.asarray(ps, dtype=np.float64) for ps in p_s]

    beta = 1.0 / (R_KCAL * T)
    A = np.exp(-beta * _deltaW_blocks(sigma, T))   # shape (3,3,nb,nb)

    # Initialize segment fields
    lnG = [np.zeros(nb, dtype=np.float64) for _ in range(3)]

    for _ in range(max_iter):
        # Use Γ^s from the *previous* iteration for all s (Jacobi update)
        G_prev = [np.exp(lnG[s]) for s in range(3)]

        lnG_new = [None, None, None]
        maxchg = 0.0

        for t in (0, 1, 2):
            # acc_m = Σ_s (A^{t,s} @ (p^s ⊙ Γ^s))  for each m
            acc = np.zeros(nb, dtype=np.float64)
            for s in (0, 1, 2):
                acc += A[t, s] @ (p_s[s] * G_prev[s])

            # guard against underflow; zeros mean “no contribution”
            acc = np.where(acc > 0.0, acc, 1e-300)
            lnG_new[t] = -np.log(acc)

        # damped Jacobi update of all three at once
        for t in (0, 1, 2):
            d = lnG_new[t] - lnG[t]
            lnG[t] += damp * d
            maxchg = max(maxchg, float(np.max(np.abs(d))))

        if maxchg < tol:
            break

    return lnG

# def _solve_lnGamma_typed(p_s, sigma, T, *, max_iter=200, tol=1e-11, damp=0.5):
#     nb = int(sigma.size)
#     # DO NOT renormalize here; trust p_s coming from mixer (already area-normalized)
#     p_s = [np.asarray(ps, float) for ps in p_s]

#     beta = 1.0 / (R_KCAL * T)
#     DW = _deltaW_blocks(sigma, T)      # (3,3,nb,nb)
#     A  = np.exp(-beta * DW)            # (3,3,nb,nb)

#     lnG = [np.zeros(nb) for _ in range(3)]
#     for _ in range(max_iter):
#         maxchg = 0.0
#         for t in (0,1,2):
#             Gt  = np.exp(lnG[t])       # Γ^t
#             acc = np.zeros(nb)
#             # z^{t} = Σ_s A^{t,s} @ ( p^s ⊙ Γ^t )
#             for s in (0,1,2):
#                 acc += A[t, s] @ (p_s[s] * Gt)
#             acc = np.where(acc > 0.0, acc, 1e-300)
#             lnG_new = -np.log(acc)
#             d = lnG_new - lnG[t]
#             lnG[t] += damp * d
#             maxchg = max(maxchg, float(np.max(np.abs(d))))
#         if maxchg < tol:
#             break
#     return lnG

def _pmix_typed(profs: List[SigmaProfile2010], x: np.ndarray) -> List[np.ndarray]:
    """p_mix^s(σ) = [Σ_i x_i dA_i^s(σ)] / [Σ_i x_i A_i],  s ∈ {NHB,OH,OT}"""
    nb = profs[0].sigma.size
    num_nhb = np.zeros(nb); num_oh = np.zeros(nb); num_ot = np.zeros(nb)
    denom = 0.0
    for xi, p in zip(x, profs):
        num_nhb += xi * p.dA_nhb
        num_oh  += xi * p.dA_oh
        num_ot  += xi * p.dA_ot
        denom   += xi * p.A
    if denom <= 0:
        raise ValueError("Mixture denominator is zero.")
    return [num_nhb/denom, num_oh/denom, num_ot/denom]

def _ln_gamma_residual_2010(profs: List[SigmaProfile2010], x: np.ndarray, T: float) -> np.ndarray:
    """Eq. (12) com Γ tipado."""
    pmix = _pmix_typed(profs, x)
    sigma = profs[0].sigma
    lnG_mix = _solve_lnGamma_typed(pmix, sigma, T)

    lng_res = np.zeros(len(profs))
    for i, p in enumerate(profs):
        pi = [p.p_nhb, p.p_oh, p.p_ot]
        lnG_i = _solve_lnGamma_typed(pi, sigma, T)
        n_i = p.A / AEFF
        accum = 0.0
        for s in range(3):
            accum += float((pi[s] * (lnG_mix[s] - lnG_i[s])).sum())
        lng_res[i] = n_i * accum
    return lng_res

def _ln_gamma_SG(profs: List[SigmaProfile2010], x: np.ndarray) -> np.ndarray:
    """Staverman–Guggenheim combinatorial (mesmo de 2002)."""
    x = np.asarray(x, float)
    # corta traços quase-zero para evitar blow-up numérico e renormaliza
    epsx = 1e-12
    x = np.clip(x, epsx, 1.0)
    x = x / x.sum()

    q = np.array([p.A / A_STD for p in profs])
    r = np.array([(p.V_vdw / V_STD) if (p.V_vdw is not None) else (p.A / A_STD) for p in profs])

    theta = x*q / max(1e-300, float((x*q).sum()))
    phi   = x*r / max(1e-300, float((x*r).sum()))
    l = (Z_SG/2.0)*(r - q) - (r - 1.0)

    eps = 1e-300
    term = np.log((phi + eps)/(x + eps)) + (Z_SG/2.0)*q*np.log((theta + eps)/(phi + eps)) + l
    lij = float((x*l).sum())
    return term - (phi/(x + eps))*lij

# ---------- public API ----------
def lngamma_2010(profiles: List[SigmaProfile2010], x: Iterable[float], T: float,
                 *, return_terms: bool = False) -> np.ndarray | Tuple[np.ndarray,np.ndarray,np.ndarray]:
    x = np.asarray(list(x), float)
    lng_res = _ln_gamma_residual_2010(profiles, x, T)
    lng_sg  = _ln_gamma_SG(profiles, x)
    lng_tot = lng_res + lng_sg
    return (lng_tot, lng_res, lng_sg) if return_terms else lng_tot

def lngamma_2010_from_paths(paths: Iterable[str], x: Iterable[float], T: float,
                            names: Optional[Iterable[str]] = None,
                            volumes: Optional[Dict[str, float]] = None
                            ) -> Tuple[np.ndarray, List[SigmaProfile2010]]:
    """
    Lê perfis ('.sigma3' ou '.sigma' simples), monta objetos de perfil,
    aplica volumes se fornecidos, e calcula lnγ total.
    """
    paths = list(paths)
    names = list(names) if names is not None else [os.path.basename(p) for p in paths]
    x = list(x)
    assert len(paths) == len(names) == len(x)

    profs: List[SigmaProfile2010] = []
    for pth, nm in zip(paths, names):
        if pth.lower().endswith('.sigma3'):
            prof = read_sigma3_file(pth, nm)
        else:
            # fallback: arquivo '.sigma' (único bloco → NHB)
            sig, dA = [], []
            with open(pth, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    s = line.strip()
                    if not s or s.startswith(('#',';')): continue
                    pr = s.replace(',', ' ').split()
                    if len(pr) >= 2:
                        try:
                            sig.append(float(pr[0])); dA.append(float(pr[1]))
                        except ValueError:
                            pass
            sigma_in = np.array(sig, float); dA_in = np.array(dA, float)
            dA_res = _resample_to_canonical(sigma_in, dA_in)
            prof = SigmaProfile2010(nm, SIGMA_GRID.copy(), dA_res,
                                    np.zeros_like(dA_res), np.zeros_like(dA_res))
        profs.append(prof)

    if volumes:
        for p in profs:
            if p.name in volumes:
                p.V_vdw = float(volumes[p.name])

    lng = lngamma_2010(profs, x, T)
    return lng
