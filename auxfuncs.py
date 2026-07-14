import numpy as np
import constants

def _rho_g_L(component: str, T_K: float | None, densities: dict | None = None) -> float:
    """Return liquid density in g/L.

    Priority order (when T_K is provided):
      1) constants.rho_g_L(component, T_K)  [DIPPR-aware + fallback]
      2) caller-provided `densities` dict   [override / legacy]
      3) constants.densities[component]     [constant fallback]

    If T_K is None:
      - cannot evaluate temperature-dependent models; then:
        1) caller-provided `densities`
        2) constants.densities
    """
    # If no temperature, only constant sources make sense
    if T_K is None:
        if densities is not None and component in densities and densities[component] is not None:
            return float(densities[component])
        try:
            import constants
            rho = getattr(constants, "densities", {}).get(component)
            if rho is None:
                raise KeyError(f"Missing density for {component} in constants.densities")
            return float(rho)
        except ModuleNotFoundError:
            pass
        raise KeyError(f"Missing density for {component} (T_K=None).")

    # T_K provided: prefer constants.rho_g_L (DIPPR-aware)
    try:
        import constants
        if hasattr(constants, "rho_g_L"):
            return float(constants.rho_g_L(component, float(T_K)))
    except Exception:
        # Any failure -> fallback to legacy dict behavior
        pass

    # Legacy override
    if densities is not None and component in densities and densities[component] is not None:
        return float(densities[component])

    # Final fallback: constants.densities
    try:
        import constants
        rho = getattr(constants, "densities", {}).get(component)
        if rho is None:
            raise KeyError(f"Missing density for {component} in constants.densities")
        return float(rho)
    except ModuleNotFoundError:
        pass

    raise KeyError(f"Missing density for {component}.")



def _mw_g_mol(component: str, molecular_weights: dict | None = None) -> float:
    """Return molecular weight in g/mol (supports legacy dict or constants.py)."""
    if molecular_weights is not None and component in molecular_weights and molecular_weights[component] is not None:
        return float(molecular_weights[component])
    try:
        import constants
        mw = getattr(constants, "molecular_weights", {}).get(component)
        if mw is None:
            raise KeyError(f"Missing molecular weight for {component} in constants.molecular_weights")
        return float(mw)
    except ModuleNotFoundError:
        pass
    raise KeyError(
        f"Missing molecular weight for {component}. Provide `molecular_weights` or define it in constants.py."
    )


# Function to calculate molar concentration
def calculate_molar_concentration(X, densities, molecular_weights, components, T_K=298.15, density_method="const"):
    """Mixture molar concentration (mol/L).

    Legacy-compatible signature (densities/molecular_weights dicts) with an optional
    temperature `T_K` to enable temperature-dependent density via constants.rho_g_L().

    Formula:
        C = (Σ x_i ρ_i) / (Σ x_i MW_i)
    where ρ in g/L and MW in g/mol, so C is mol/L.
    """
    import constants
    X = np.asarray(X, float)
    numerator = 0.0
    denominator = 0.0
    for i, comp in enumerate(components):
        # rho_i = _rho_g_L(comp, T_K, densities)
        rho_i = constants.rho_g_L(comp, T_K, density_method=density_method)
        mw_i = _mw_g_mol(comp, molecular_weights)
        numerator += X[i] * rho_i
        denominator += X[i] * mw_i
    return numerator / denominator

def calculate_phase_volume(N, comp, molecular_weights, densities, *, T_K: float | None = None):
    """
    Calculate the total volume of a phase.

    Parameters:
    - N: List of mole numbers for components in the phase.
    - comp: List of component names.
    - molecular_weights: Dictionary of molecular weights (g/mol).
    - densities: Dictionary of densities (g/L).

    Returns:
    - Total volume of the phase (L).
    """
    volume = 0
    for i, component in enumerate(comp):
        MW = _mw_g_mol(component, molecular_weights)  # g/mol
        rho = _rho_g_L(component, T_K, densities)     # g/L
        volume += float(N[i]) * MW / rho  # L
    return volume

def save_results(filename, data):
    np.savetxt(filename, data, fmt='%.6f', delimiter='\t')
    print(f"Results saved to {filename}")

# In auxfuncs.py

import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

def show_volume_plot(parent, volume_I, volume_II, xI, xII, N_I, N_II, components):
    popup = tk.Toplevel(parent)
    popup.title("Volume Visualization")
    popup.geometry("600x700")

    fig, ax = plt.subplots(figsize=(4, 8))
    total_volume = volume_I + volume_II
    phase_1_fraction = volume_I / total_volume
    phase_2_fraction = volume_II / total_volume

    # Draw the tank as stacked bars
    ax.bar([0.5], [phase_1_fraction],
           color="blue", edgecolor="black", label="Phase 1")
    ax.bar([0.5], [phase_2_fraction],
           bottom=[phase_1_fraction],
           color="green", edgecolor="black", label="Phase 2")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Tank Volume Distribution", fontsize=14)

    xI = [float(frac) for frac in xI] if isinstance(xI, (list, tuple)) else xI
    xII = [float(frac) for frac in xII] if isinstance(xII, (list, tuple)) else xII

    phase_1_text = (
        f"Phase 1\nVolume: {volume_I:.2f} L\n"
        f"Moles: [{', '.join(f'{ni:.4f}' for ni in N_I)}]\n"
        "Composition:\n" +
        "\n".join(f"  {comp}: {float(frac):.2f}" for comp, frac in zip(components, xI))
    )
    phase_2_text = (
        f"Phase 2\nVolume: {volume_II:.2f} L\n"
        f"Moles: [{', '.join(f'{ni:.4f}' for ni in N_II)}]\n"
        "Composition:\n" +
        "\n".join(f"  {comp}: {float(frac):.2f}" for comp, frac in zip(components, xII))
    )

    ax.text(0.2, phase_1_fraction / 2, phase_1_text,
            fontsize=10, color="white", ha="center", va="center",
            bbox=dict(facecolor="blue", alpha=0.8, edgecolor="none"))
    ax.text(0.2, phase_1_fraction + phase_2_fraction / 2, phase_2_text,
            fontsize=10, color="white", ha="center", va="center",
            bbox=dict(facecolor="green", alpha=0.8, edgecolor="none"))

    component_text = "Components:\n" + ", ".join(components)
    ax.text(0.5, -0.1, component_text, fontsize=10, color="black",
            ha="center", va="center", transform=ax.transAxes)

    canvas = FigureCanvasTkAgg(fig, master=popup)
    canvas.draw()
    canvas.get_tk_widget().pack(padx=10, pady=10)

    close_button = ttk.Button(popup, text="Close", command=popup.destroy)
    close_button.pack(pady=10)