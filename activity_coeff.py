import os
import unicodedata
import inspect

from unifac import unifac_lngamma
from cosmosac2002 import lngamma_2002_from_paths
from cosmosac2010 import lngamma_2010_from_paths

METHODS = {
    'unifac': unifac_lngamma,
    'unifac-lle': unifac_lngamma,
    'cosmosac2002': lngamma_2002_from_paths,
    'cosmosac2010': lngamma_2010_from_paths,
}

_COSMO_SIG_EXT = {
    "cosmosac2002": ".sigma",
    "cosmosac2010": ".sigma3",
    "cosmosacdsp":  ".sigma3",
    "cosmosac2013": ".sigma13",
}

_KNOWN_SIG_EXTS = (".sigma", ".sigma3", ".sigma13")

def filter_kwargs(func, kwargs: dict) -> dict:
    """Keep only kwargs that `func` accepts."""
    sig = inspect.signature(func)
    return {k: v for k, v in kwargs.items() if k in sig.parameters}

def _normalize_comp_name(name: str) -> str:
    if name.lower().endswith(('.sigma', '.sigma3', '.sigma13')):
        return name.replace('\\', '/')
    nfkd = unicodedata.normalize('NFKD', name)
    ascii_only = ''.join(c for c in nfkd if not unicodedata.combining(c))
    return ascii_only.strip().upper().replace(' ', '_').replace('/', '_').replace('\\', '_')

# def _build_cosmosac_paths(components, method, profiles_dir="profiles"):
#     ext = '.sigma' if method == 'cosmosac2002' else '.sigma3'
#     paths, missing = [], []
#     for comp in components:
#         base = _normalize_comp_name(comp)
#         path = base if base.lower().endswith(('.sigma', '.sigma3')) else os.path.join(profiles_dir, f"{base}{ext}")
#         if not os.path.exists(path):
#             missing.append(path)
#         paths.append(path)
#     if missing:
#         raise FileNotFoundError(
#             "COSMO-SAC profile files not found:\n  - " + "\n  - ".join(missing) +
#             f"\nChecked using method='{method}' (expected '{ext}') in profiles_dir='{profiles_dir}'."
#         )
#     return paths
def _build_cosmosac_paths(components, method, profiles_dir="profiles", *, allow_fallback=True):
    m = (method or "").lower()
    if m not in _COSMO_SIG_EXT:
        raise ValueError(f"Unknown COSMO-SAC method '{method}'. Expected one of: {list(_COSMO_SIG_EXT)}")

    expected_ext = _COSMO_SIG_EXT[m]

    paths, missing = [], []
    for comp in components:
        base = _normalize_comp_name(comp)

        # If user already provided an explicit profile filename, honor it.
        if base.lower().endswith(_KNOWN_SIG_EXTS):
            candidates = [base] if os.path.isabs(base) else [os.path.join(profiles_dir, base)]
        else:
            # Default expected file for this method
            candidates = [os.path.join(profiles_dir, f"{base}{expected_ext}")]

            # Optional fallback: try other known extensions (useful during transition/debug)
            if allow_fallback:
                for ext in _KNOWN_SIG_EXTS:
                    if ext == expected_ext:
                        continue
                    candidates.append(os.path.join(profiles_dir, f"{base}{ext}"))

        found = next((p for p in candidates if os.path.exists(p)), None)
        if found is None:
            missing.append(candidates[0])   # report the expected one
            paths.append(candidates[0])
        else:
            paths.append(found)

    if missing:
        raise FileNotFoundError(
            "COSMO-SAC profile files not found:\n  - " + "\n  - ".join(missing) +
            f"\nChecked using method='{method}' (expected '{expected_ext}') in profiles_dir='{profiles_dir}'."
        )

    return paths

def calculate_activity_coefficients(x, N, T, NC, method, comp, params):
    """
    x: array-like
    N: array-like (not used by COSMO-SAC but kept for API compatibility)
    T: float
    NC: int
    method: str
    comp: list[str]
    params: dict
    """
    if method not in METHODS:
        raise ValueError(f"Unknown method: {method}. Available: {list(METHODS.keys())}")

    params = dict(params or {})
    params.pop("groups", None)  # UNIFAC-only artifact

    # Common kwargs we might want to pass to many methods
    common_kw = dict(x=x, N=N, T=T, NC=NC, **params)

    if method == "unifac-debye-huckel":
        # Pass only what the function accepts
        func = METHODS[method]
        kw = dict(comp=comp, **filter_kwargs(func, common_kw))
        return func(**kw)

    if method in ("cosmosac2002", "cosmosac2010", "cosmosacpdh", "cosmosacdsp", "cosmosac2013"):
        if len(comp) != NC:
            raise ValueError(f"Length of 'comp' ({len(comp)}) must match NC ({NC}) for COSMO-SAC.")
        profiles_dir = params.pop("profiles_dir", "profiles")
        paths = _build_cosmosac_paths(comp, method, profiles_dir=profiles_dir)

        # COSMO-SAC functions typically have signature: (paths, x, T, ...)
        func = METHODS[method]
        # Start with mandatory args; then filter any remaining params to what func accepts
        base_kw = dict(paths=paths, x=x, T=T)
        extra_kw = filter_kwargs(func, params)  # strips NG, Q, R, etc.
        return func(**base_kw, **extra_kw)

    # All other methods (UNIFAC variants, etc.)
    func = METHODS[method]
    kw = filter_kwargs(func, common_kw)
    return func(**kw)
