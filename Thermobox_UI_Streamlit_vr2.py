# Thermobox_UI_Streamlit_vr2.py

import numpy as np
import streamlit as st

import constants
from auxfuncs import show_volume_plot  # still here if you want to reuse later

from lle_calcs import lle_calc_volume, lle_calc, kow_calc
import vle_calcs

# -----------------------------
# Parameter + database loaders
# -----------------------------

def get_parameters(method, comp, solvents=None):
    """Fetch thermodynamic parameters, mirroring Thermobox_UI_vr2_3 logic."""
    if method == "unifac-lle":
        from get_unifaclle_parameters import get_unifac_parameters
        params = get_unifac_parameters(comp)
    elif method == "unifac":
        from get_unifac_parameters import get_unifac_parameters
        params = get_unifac_parameters(comp)
    elif method == "unifac-debye-huckel":
        from get_unifacllepdh_parameters import get_unifac_parameters
        params = get_unifac_parameters(comp)
        params["solvents"] = solvents
    elif method == "unifac-vle-debye-huckel":
        from get_unifacpdh_parameters import get_unifac_parameters
        params = get_unifac_parameters(comp)
        params["solvents"] = solvents
    # COSMO-SAC models currently reuse the UNIFAC-VLE parameter loader
    elif method in ("cosmosac-2002", "cosmosac-2010", "cosmosac-dsp"):
        from get_unifac_parameters import get_unifac_parameters
        params = get_unifac_parameters(comp)
    else:
        # default to unifac
        from get_unifac_parameters import get_unifac_parameters
        params = get_unifac_parameters(comp)

    return params


def get_component_database(model: str):
    """Return the component dictionary for the chosen model.

    For COSMO-SAC, we reuse the UNIFAC LLE database so you have the same
    list of component names; COSMO-SAC itself will look for sigma profiles
    using those keys.
    """
    if model == "unifac-vle":
        from unifac_database import unifac_molecules as db_molecules
    elif model == "unifac-lle":
        from unifac_lle_database import unifac_molecules as db_molecules
    elif model == "unifac-debye-huckel":
        from unifac_llepdh_database import unifac_molecules as db_molecules
    elif model == "unifac-vle-debye-huckel":
        from unifac_pdh_database import unifac_molecules as db_molecules
    elif model in ("cosmosac-2002", "cosmosac-2010", "cosmosac-dsp"):
        # share the same component list as UNIFAC-LLE
        from unifac_lle_database import unifac_molecules as db_molecules
    else:
        from unifac_database import unifac_molecules as db_molecules
    return db_molecules


# -----------------------------
# VLE helper: prepare globals
# -----------------------------

def prepare_vle_globals(comp, z, params):
    """
    [Inference] Prepare globals expected by vle_calcs:
    comp, params, n, NC, A, B, C.

    - z: array of total moles for each component.
    - Antoine coefficients are taken from vle_calcs.db (indexed by 'comp').
    """
    n = np.array(z, float)
    NC = len(comp)

    A = np.zeros(NC)
    B = np.zeros(NC)
    C = np.zeros(NC)

    # Assumes antoine_coeff.csv has columns 'comp','A','B','C'
    for i, cname in enumerate(comp):
        row = vle_calcs_joao_marcelo.db.loc[cname]
        A[i] = row["A"]
        B[i] = row["B"]
        C[i] = row["C"]

    # Attach as module globals (this matches how your original script works)
    vle_calcs_joao_marcelo.comp = comp
    vle_calcs_joao_marcelo.params = params
    vle_calcs_joao_marcelo.n = n
    vle_calcs_joao_marcelo.NC = NC
    vle_calcs_joao_marcelo.A = A
    vle_calcs_joao_marcelo.B = B
    vle_calcs_joao_marcelo.C = C


def run_vle_dew_T(comp, z, T_K, params):
    """
    VLE: dew point at known temperature (T known, solve for P and liquid x).

    - T_K is Kelvin from UI; vle_calcs expects °C and internally adds 273.15.
    """
    prepare_vle_globals(comp, z, params)
    z_mf = np.array(z, float) / np.sum(z)
    T_C = T_K - 273.15
    return vle_calcs_joao_marcelo.dew_point_T_known(z_mf, T_C)


def run_vle_bubble_T(comp, z, T_K, params):
    """
    VLE: bubble point at known temperature (T known, solve for P and vapor y).
    """
    prepare_vle_globals(comp, z, params)
    z_mf = np.array(z, float) / np.sum(z)
    T_C = T_K - 273.15
    return vle_calcs_joao_marcelo.bubble_point_T_known(z_mf, T_C)


# -----------------------------
# Calculation dispatcher
# -----------------------------

def run_thermobox_calculation(calc_type, comp, z, T, method, params, extra=None):
    """
    Wrapper to call your existing calculation functions.
    """
    if calc_type == "LLE Volume":
        return lle_calc_volume(comp, np.array(z, float), T, method, params)

    elif calc_type == "LLE":
        return lle_calc(comp, np.array(z, float), T, method, params)

    elif calc_type == "Kow":
        pH = extra["pH"]
        solute = extra["solute"]
        return kow_calc(comp, np.array(z, float), pH, solute, T, method, params)

    elif calc_type == "VLE – Dew point (T)":
        if method != "unifac-vle":
            raise ValueError(
                "VLE currently implemented only for 'unifac-vle' in vle_calcs.py."
            )
        return run_vle_dew_T(comp, z, T, params)

    elif calc_type == "VLE – Bubble point (T)":
        if method != "unifac-vle":
            raise ValueError(
                "VLE currently implemented only for 'unifac-vle' in vle_calcs.py."
            )
        return run_vle_bubble_T(comp, z, T, params)

    else:
        raise ValueError(f"Unknown calc_type: {calc_type}")


# -----------------------------
# Streamlit UI
# -----------------------------

st.set_page_config(
    page_title="ThermoBox",
    layout="wide",
)

st.title("ThermoBox – A Toolbox for Thermodynamics Calculations")

with st.sidebar:
    st.header("Setup")

    # Thermodynamic model
    thermo_model = st.selectbox(
        "Thermodynamic Model",
        [
            "unifac-vle",
            "unifac-lle",
            "unifac-debye-huckel",
            "unifac-vle-debye-huckel",
            "cosmosac-2002",
            "cosmosac-2010",
            "cosmosac-dsp",
        ],
        index=1,  # unifac-lle by default
    )

    # Calculation type
    calc_type = st.selectbox(
        "Calculation Type",
        [
            "LLE Volume",
            "LLE",
            "Kow",
            "VLE – Dew point (T)",
            "VLE – Bubble point (T)",
        ],
        index=0,
    )

    # Small guard: VLE only implemented with unifac-vle for now
    if calc_type.startswith("VLE") and thermo_model != "unifac-vle":
        st.warning(
            "VLE calculations (dew/bubble) are currently implemented only for "
            "'unifac-vle'. Please select that model for VLE, or choose LLE/Kow "
            "for the other models."
        )

    # Number of components
    max_nc = 10
    NC = st.number_input(
        "Number of components (NC)",
        min_value=1,
        max_value=max_nc,
        value=3,
        step=1,
    )

    # Temperature (used by all calcs; VLE wrappers convert to °C internally)
    T = st.number_input(
        "Temperature [K]",
        min_value=200.0,
        max_value=800.0,
        value=298.15,
        step=1.0,
    )

    st.markdown("---")
    st.caption(
        "LLE & Kow use T in Kelvin directly. "
        "VLE dew/bubble use T (K) but are converted to °C inside vle_calcs."
    )

# Title vertical position
st.markdown(
    """
    <style>
    h1 {
        margin-top: 1.8rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Calculation description block
calc_help = {
    "LLE Volume": (
        "Given overall moles of each component (feed) and temperature T (K), "
        "calculates liquid–liquid equilibrium and returns phase compositions, "
        "mole splits, and phase volumes."
    ),
    "LLE": (
        "Given overall moles and temperature T (K), calculates liquid–liquid "
        "equilibrium (no explicit volume calculation)."
    ),
    "Kow": (
        "Given a biphasic aqueous/organic system with a chosen solute, "
        "temperature T (K) and pH, performs LLE and computes Kow, distribution "
        "coefficient D, and related quantities."
    ),
    "VLE – Dew point (T)": (
        "Given vapor-phase overall composition (feed) and temperature T, "
        "solves the dew-point problem: equilibrium liquid composition x and "
        "system pressure P."
    ),
    "VLE – Bubble point (T)": (
        "Given liquid-phase overall composition (feed) and temperature T, "
        "solves the bubble-point problem: equilibrium vapor composition y and "
        "system pressure P."
    ),
}

# Load database for this model
db_molecules = get_component_database(thermo_model)
component_options = sorted(db_molecules.keys())

st.subheader("Components and Feed Composition")
st.info(calc_help.get(calc_type, ""), icon="ℹ️")

if not component_options:
    st.error("No components found in the database for this model.")
    st.stop()

# --- small CSS tweaks: tighten main width & right-align numbers a bit ---
st.markdown(
    """
    <style>
    /* make the main content a bit narrower on very wide screens */
    .block-container {
        max-width: 1300px;
        padding-top: 1rem;
    }
    /* right-align plain text numbers in our table cells */
    .thermobox-num {
        text-align: right;
        padding-top: 0.6rem;  /* vertical alignment with input boxes */
        font-variant-numeric: tabular-nums;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header row – narrower columns
cols_header = st.columns([2.5, 1.3, 1.0, 1.0, 1.0])
cols_header[0].markdown("**Component**")
cols_header[1].markdown("**Moles (input)**")
cols_header[2].markdown("**Mole fraction**")
cols_header[3].markdown("**Mass [g]**")
cols_header[4].markdown("**Conc. [g/L]**")

# First pass: inputs + placeholders on same row
comp_list = []
n_list = []
x_placeholders = []
m_placeholders = []
c_placeholders = []

for i in range(NC):
    with st.container():
        c0, c1, c2, c3, c4 = st.columns([2.5, 1.3, 1.0, 1.0, 1.0])

        comp_i = c0.selectbox(
            f"Component {i+1}",
            component_options,
            index=i if i < len(component_options) else 0,
            key=f"comp_{i}",
        )
        n_i = c1.number_input(
            f"n_{i+1} [mol]",
            min_value=0.0,
            value=1.0,
            step=0.1,
            key=f"n_{i}",
        )

        # Placeholders to be filled after totals are known
        x_ph = c2.empty()
        m_ph = c3.empty()
        c_ph = c4.empty()

        comp_list.append(comp_i)
        n_list.append(n_i)
        x_placeholders.append(x_ph)
        m_placeholders.append(m_ph)
        c_placeholders.append(c_ph)

# Compute totals
z = np.array(n_list, float)
total_moles = z.sum()

total_volume = 0.0
volumes = []
for comp_i, n_i in zip(comp_list, z):
    MW = float(constants.molecular_weights.get(comp_i, 0.0))
    rho = float(constants.densities.get(comp_i, 1.0))  # g/L
    vol = (n_i * MW) / rho if rho > 0 else 0.0  # L
    volumes.append(vol)
    total_volume += vol

# Fill placeholders in the SAME rows (aligned with inputs)
for i in range(NC):
    comp_i = comp_list[i]
    n_i = z[i]
    MW = float(constants.molecular_weights.get(comp_i, 0.0))

    xi = n_i / total_moles if total_moles > 0 else 0.0
    mass_i = n_i * MW
    conc_i = (mass_i / total_volume) if total_volume > 0 else 0.0

    x_placeholders[i].markdown(
        f"<div class='thermobox-num'>{xi:.4f}</div>",
        unsafe_allow_html=True,
    )
    m_placeholders[i].markdown(
        f"<div class='thermobox-num'>{mass_i:.2f}</div>",
        unsafe_allow_html=True,
    )
    c_placeholders[i].markdown(
        f"<div class='thermobox-num'>{conc_i:.2f}</div>",
        unsafe_allow_html=True,
    )

# Optional: choose solvents when using Debye-Hückel
solvents = None
if thermo_model == "unifac-debye-huckel":
    st.subheader("Select Solvents (up to NC-1)")
    selected_solvents = st.multiselect(
        "Solvent components",
        options=comp_list,
        default=comp_list[: max(0, NC - 1)],
    )
    if len(selected_solvents) > (NC - 1):
        st.error("You can select at most NC-1 solvents.")
        st.stop()
    solvents = selected_solvents

# Extra inputs for Kow
extra_inputs = {}
if calc_type == "Kow":
    st.subheader("Extra Inputs for Kow")
    col1, col2 = st.columns(2)
    pH = col1.number_input(
        "pH", min_value=0.0, max_value=14.0, value=7.0, step=0.1
    )
    # solute must be one of the components
    solute = col2.selectbox("Solute", comp_list, index=0)
    extra_inputs["pH"] = pH
    extra_inputs["solute"] = solute

# -----------------------------
# Run calculation
# -----------------------------

st.markdown("---")
run_button = st.button("Run Calculation")

if run_button:
    # Basic validations
    if any(c is None or c == "" for c in comp_list):
        st.error("Please select all components.")
        st.stop()
    if total_moles <= 0:
        st.error("Total moles must be > 0.")
        st.stop()

    method = thermo_model  # keep same naming convention as your code

    try:
        params = get_parameters(method, comp_list, solvents)
    except Exception as e:
        st.exception(f"Error getting parameters: {e}")
        st.stop()

    # -----------------------------
    # Build extra inputs for Kow and VLE
    # -----------------------------

    extra_inputs = {}

    # 1. Kow extra inputs
    if calc_type == "Kow":
        extra_inputs["pH"] = pH
        extra_inputs["solute"] = solute

    # 2. VLE extra inputs
    if calc_type == "VLE":
        # Convert UI strings to clean internal flags
        extra_inputs["known_var"] = "T" if known_var.startswith("Temperature") else "P"
        extra_inputs["feed_phase"] = "x" if feed_phase.startswith("Liquid") else "y"
        extra_inputs["P_known"] = P_known   # May be None if known_var == "T"
    # -------------------------
    # Run calculation
    try:
        result = run_thermobox_calculation(
            calc_type, comp_list, z, T, method, params, extra_inputs
        )
    except Exception as e:
        st.exception(f"Calculation failed: {e}")
        st.stop()

    # -------------------------
    # Display results
    # -------------------------
    st.subheader("Results")

    if calc_type == "LLE Volume":
        N_I, N_II, xI, xII, volume_I, volume_II = result

        colA, colB = st.columns(2)
        with colA:
            st.markdown("**Phase I**")
            st.write("Mole fractions:")
            st.write(dict(zip(comp_list, [float(x) for x in xI])))
            st.write("Moles:")
            st.write(dict(zip(comp_list, [float(n) for n in N_I])))
            st.write(f"Volume: {float(volume_I):.6f} L")
        with colB:
            st.markdown("**Phase II**")
            st.write("Mole fractions:")
            st.write(dict(zip(comp_list, [float(x) for x in xII])))
            st.write("Moles:")
            st.write(dict(zip(comp_list, [float(n) for n in N_II])))
            st.write(f"Volume: {float(volume_II):.6f} L")

        # Simple bar plot of volumes
        st.bar_chart(
            {
                "Phase": ["I", "II"],
                "Volume [L]": [float(volume_I), float(volume_II)],
            },
            x="Phase",
            y="Volume [L]",
        )

    elif calc_type == "LLE":
        N_I, N_II, xI, xII, vv = result

        st.write("**Feed moles**")
        st.write(dict(zip(comp_list, [float(ni) for ni in z])))

        st.markdown("**Phase I**")
        st.write("Mole fractions:")
        st.write(dict(zip(comp_list, [float(x) for x in xI])))
        st.write("Moles:")
        st.write(dict(zip(comp_list, [float(n) for n in N_I])))

        st.markdown("**Phase II**")
        st.write("Mole fractions:")
        st.write(dict(zip(comp_list, [float(x) for x in xII])))
        st.write("Moles:")
        st.write(dict(zip(comp_list, [float(n) for n in N_II])))

        st.write(f"Flash variable vv: {float(vv):.6f}")

    elif calc_type == "Kow":
        (
            N_I,
            N_II,
            xI,
            xII,
            vv,
            Kow,
            D,
            CO,
            CW,
        ) = result  # if you switch to kowIDAC, adjust unpacking

        st.write(f"Model: `{method}`")
        st.write(f"pH: {extra_inputs['pH']:.2f}")
        st.write(f"Solute: `{extra_inputs['solute']}`")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Phase I**")
            st.write("Mole fractions:")
            st.write(dict(zip(comp_list, [float(x) for x in xI])))
            st.write("Moles:")
            st.write(dict(zip(comp_list, [float(n) for n in N_I])))
        with col2:
            st.markdown("**Phase II**")
            st.write("Mole fractions:")
            st.write(dict(zip(comp_list, [float(x) for x in xII])))
            st.write("Moles:")
            st.write(dict(zip(comp_list, [float(n) for n in N_II])))

        st.markdown("**Kow and related quantities**")
        st.write(f"CW (aqueous total conc.): {float(CW):.6f}")
        st.write(f"CO (organic total conc.): {float(CO):.6f}")
        st.write(f"Kow: {float(Kow):.6f}")
        st.write(f"log10(Kow): {np.log10(Kow):.6f}")
        st.write(f"D (distribution coefficient): {float(D):.6f}")

        st.bar_chart(
            {
                "Quantity": ["Kow", "D"],
                "Value": [float(Kow), float(D)],
            },
            x="Quantity",
            y="Value",
        )

    elif calc_type == "VLE – Dew point (T)":
        x, P_sat, P, gamma = result

        st.write(f"T = {T:.2f} K  ({T - 273.15:.2f} °C)")
        st.write(f"P (raw units): {float(P):.4f}")
        st.write(f"P ≈ {float(P) / 750.0:.4f} bar  [assuming P/750 → bar]")

        st.markdown("**Liquid composition x (dew point)**")
        st.write(dict(zip(comp_list, [float(xi) for xi in x])))

        st.markdown("**Saturation pressures (P_sat)**")
        st.write(dict(zip(comp_list, [float(p) for p in P_sat])))

        st.markdown("**Activity coefficients γ (UNIFAC)**")
        st.write(dict(zip(comp_list, [float(g) for g in gamma])))

    elif calc_type == "VLE – Bubble point (T)":
        y, P_sat, P, gamma = result

        st.write(f"T = {T:.2f} K  ({T - 273.15:.2f} °C)")
        st.write(f"P (raw units): {float(P):.4f}")
        st.write(f"P ≈ {float(P) / 750.0:.4f} bar  [assuming P/750 → bar]")

        st.markdown("**Vapor composition y (bubble point)**")
        st.write(dict(zip(comp_list, [float(yi) for yi in y])))

        st.markdown("**Saturation pressures (P_sat)**")
        st.write(dict(zip(comp_list, [float(p) for p in P_sat])))

        st.markdown("**Activity coefficients γ (UNIFAC)**")
        st.write(dict(zip(comp_list, [float(g) for g in gamma])))
