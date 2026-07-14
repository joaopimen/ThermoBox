from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional, Iterable, Mapping, Tuple
import math
from pathlib import Path
import csv

# ============================================================
# Fundamental constants
# ============================================================

N_A = 6.02214076e23  # 1/mol (SI definition)
CM3_TO_A3 = 1.0e24   # 1 cm^3 = (1e8 Å)^3 = 1e24 Å^3

_DIPPR_LIQDEN_CACHE = None

@dataclass(frozen=True)
class DipprLiqDenRow:
    name_key: str
    dippr_id: int
    A1: float
    A2: float
    A3: float
    A4: float
    MW: float
    Tc_K: float
    Dc: float

def _load_dippr_liqden_db():
    global _DIPPR_LIQDEN_CACHE
    if _DIPPR_LIQDEN_CACHE is not None:
        return _DIPPR_LIQDEN_CACHE

    # ajuste o caminho conforme onde você salvou
    csv_path = Path(__file__).with_name("LiqDen_by_name.csv")

    db = {}
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name_key"].strip()
            dippr_id = int(float(row["DIPPR_ID"]))  # às vezes vem como "105.0"
            db[name] = DipprLiqDenRow(
                name_key=name,
                dippr_id=dippr_id,
                MW=float(row["MW"]) if row["MW"] else 0.0,
                Tc_K=float(row["Tc_K"]) if row["Tc_K"] else 0.0,
                Dc=float(row["Dc"]) if row["Dc"] else 0.0,
                A1=float(row["A1"]) if row["A1"] else 0.0,
                A2=float(row["A2"]) if row["A2"] else 0.0,
                A3=float(row["A3"]) if row["A3"] else 0.0,
                A4=float(row["A4"]) if row["A4"] else 0.0,
            )

    _DIPPR_LIQDEN_CACHE = db
    return db

# All data taken from pubchem
# Molecular weights dictionary [g/mol]
molecular_weights: Dict[str, Optional[float]] = {
    # Alkanes (1–10 carbons)
    "METHANE": 16.04,
    "ETHANE": 30.07,
    "PROPANE": 44.10,
    "BUTANE": 58.12,
    "PENTANE": 72.15,
    "HEXANE": 86.18,
    'N-HEPTANE': 100.21,
    "N-OCTANE": 114.23,
    'ISOOCTANE': 114.23,
    "NONANE": 128.26,
    "DECANE": 142.29,
    "UNDECANE": 156.31,
    "DODECANE": 170.33,
    "TRIDECANE": 184.36,
    "TETRADECANE": 198.39,
    "PENTADECANE": 212.41,
    "HEXADECANE": 226.44,
    "HEPTADECANE": 240.47,
    "OCTADECANE": 254.49,
    'TETRACHLOROMETHANE': 153.8,

    "2,2,4-TRIMETHYLPENTANE": 114.23,
    "1-METHYLNAPHTHALENE": 142.20,
    "BENZENE": 78.11,


    # Alcohols
    "METHANOL": 32.04,
    "ETHANOL": 46.07,
    "1-PROPANOL": 60.10,
    "1-BUTANOL": 74.12,
    "1-PENTANOL": 88.15,
    "1-HEXANOL": 102.17,
    "1-HEPTANOL": 116.20,
    '1-OCTANOL': 130.23,
    "1-NONANOL": 144.25,
    "1-DECANOL": 158.27,
    "ISOPROPANOL": 60.1,
    "THYMOL": 150.22,
    "MENTHOL": 156.26,
    "1-TETRADECANOL": 214.39,
    "1-HEXADECANOL": 242.44,
    "1-OCTADECANOL": 270.5,

    # Solvents
    'WATER': 18.01528,
    'TOLUENE': 92.14,
    'PENTACHLOROPHENOL': 266.34,
    "ACETONITRILE": 41.05,

    # Acids
    'HYDROCHLORIC_ACID': 36.46,
    'HCL': 36.46,
    'HCl': 36.46,
    'BENZOIC_ACID': 122.12,
    'CYCLOHEXANEACETIC_ACID': 142.20,
    '4-HEPTYLBENZOIC_ACID': 220.31,
    'ACETIC_ACID': 60.052,
    'OXALIC_ACID': 90.03,
    'GLYCOLIC_ACID': 76.05,
    'MALIC_ACID': 134.09,
    'SUCCINIC_ACID': 118.09,
    'VALERIC_ACID': 102.13,
    'CYCLOHEXANOACETIC_ACID': 142.20,
    'CYCLOHEXANECARBOXILIC_ACID': 128.17,
    'HEPTANOIC_ACID':130.18,
    '1,4_CYCLOHEXANEDICARBOXYLIC_ACID':172.18,
    '1-METHYLCYCLOHEXANECARBOXYLIC_ACID':142.20,
    'DECANOIC_ACID':172.26,
    '5-PHENYLVALERIC_ACID':178.23,
    '2-NAPHTHOIC_ACID':172.18,
    'CYCLOHEXANEPENTANOIC_ACID':184.27,
    'LAURIC_ACID':200.32,
    'DIPHENYLACETIC_ACID':212.24,
    'DIPHENYLVALERIC_ACID':254.32,
    "MYRISTIC_ACID":228.37,
    "PALMITIC_ACID":256.42,
    "HEXADECANOIC_ACID": 256.42,
    "PALMITOLEIC_ACID":254.41,
    "STEARIC_ACID":284.5,
    "OCTADECANOIC_ACID":284.5,
    "OLEIC_ACID":282.5,
    "LINOLEIC_ACID":280.4,
    "LINOLENIC_ACID":278.4,
    "ARACHIDIC_ACID":312.5,
    "BEHENIC_ACID":340.6,
    "LIGNOCERIC_ACID":368.6,
    "FORMIC_ACID": 46.025,
    "MALONIC_ACID": 104.06,
    "BUTYRIC_ACID": 88.11,
    "TARTARIC_ACID": 150.09,
    "LACTIC_ACID": 90.08,
    "CITRIC_ACID": 192.12,
    "OCTANOIC_ACID": 144.21,
    "LEVULINIC_ACID": 116.11,

    # Bases and salts
    'SODIUM_HYDROXIDE': 39.997,
    'NAOH': 39.997,
    'NACL': 58.44,
    'SODIUM_CHLORIDE': 58.44,
    'QUINOLINE': 129.16,
    "1,2-DICHLOROETHANE": 98.9592,
    
    # FAMEs
    "METHYL_PALMITATE": 270.5,
    # 9,12-Octadecadienoic acid (Z,Z)-, methyl ester
    "METHYL_LINOLEATE": 294.5,
    # 9-Octadecenoic acid (Z)-, methyl ester
    "METHYL_OLEATE": 296.48,
    # Octadececanoic acid, methyl ester
    "METHYL_STEARATE": 298.47,
    # "HEXADECANOIC_ACID": 270.45,    # "9,12-OCTADECADIENOIC_ACID": 280.40,
    # "9-OCTADECANOIC_ACID": 282.50,
    # "OCTADECANOIC_ACID": 284.50,

    # Acetates
    "BUTYL_ACETATE": 116.16,
    'ISOBUTYL_ACETATE': 116.16,
    "ETHYL_ACETATE": 88.11,
    "BENZYL_ACETATE": 150.17,

    # Amines
    "N,N-DIETHYLANILINE": 149.23,
    # Ketones
    "ACETONE": 58.08,
    "2-BUTANONE": 72.11,
    "METHYL_ETHYL_KETONE": 72.11,

    # Terpenes
    "P-CYMENE": 134.22,
    "TRIOLEIN": 885.45,

    "CCl4": 153.82,
    
    # Ions
    'Na+': 22.99,
    'NA+': 22.99,
    'Cl-': 35.45,
    'CL-': 35.45,
    'Ca2+':40.08,
    "Li+": 6.94,
    "K+": 39.10,
    "Ba2+": 137.33,
    "Sr2+": 87.62,
    "Cu2+": 63.55,
    "Ni2+": 58.69,
    "Hg2+": 200.59,
    "F-": 19.00,
    "Br-": 79.90,
    "I-": 126.90,
    "NO3-": 62.00,
    "CH3COO-": 59.04,
    "OH-": 17.01,

    # Various
    "TERT-BUTYL_CHLORIDE": 92.57,
    }
# Densities dictionary [g/L]
densities : Dict[str, Optional[float]] = {
    # Alkanes (liquid phase approximations)
    "METHANE": 422,
    "ETHANE": 550,
    "PROPANE": 580,
    "BUTANE": 600,
    "PENTANE": 620,
    "HEXANE": 660,
    'N-HEPTANE': 679.5,
    "N-OCTANE": 703,
    'ISOOCTANE': 691.94,
    "NONANE": 730,
    "DECANE": 730,
    "DODECANE": 748.7,
    'TETRACHLOROMETHANE': 1594,

    "2,2,4-TRIMETHYLPENTANE": 691.94,
    "1-METHYLNAPHTHALENE": 1020.2,
    "BENZENE": 875.6, 
    
    # Alcohols
    "METHANOL": 792,
    "ETHANOL": 789,
    "1-PROPANOL": 803,
    "1-BUTANOL": 810,
    "1-PENTANOL": 815,
    "1-HEXANOL": 820,
    "1-HEPTANOL": 827,
    '1-OCTANOL': 827,
    "1-NONANOL": 830,
    "1-DECANOL": 835,
    "THYMOL": 969.9,
    "MENTHOL": 904,
    "1-TETRADECANOL": 824,
    "1-HEXADECANOL": 818.7,
    "1-OCTADECANOL": 812.4,

    # Solvents
    'WATER': 1000,
    'TOLUENE': 862.3,
    'PENTACHLOROPHENOL': 1978,

    # Acids
    'HYDROCHLORIC_ACID': 1.639,
    'HCL': 1.639,
    'HCl': 1.639,
    'BENZOIC_ACID': 1270,
    'CYCLOHEXANEACETIC_ACID': 1007,
    '4-HEPTYLBENZOIC_ACID': 1013, # https://www.molbase.com/supplier/789290-product-7798010.html
    'ACETIC_ACID': 1050,
    'OXALIC_ACID': 1900,
    'GLYCOLIC_ACID': 1490,
    'MALIC_ACID': 1601,
    'SUCCINIC_ACID': 1572,
    'VALERIC_ACID': 939,
    'CYCLOHEXANOACETIC_ACID': 1001,
    'CYCLOHEXANECARBOXILIC_ACID': 1029,
    'HEPTANOIC_ACID':918.1,
    '1,4_CYCLOHEXANEDICARBOXYLIC_ACID':1360, # https://en.wikipedia.org/wiki/1,4-Cyclohexanedicarboxylic_acid
    '1-METHYLCYCLOHEXANECARBOXYLIC_ACID':1003.7, # https://www.chemicalbook.com/ChemicalProductProperty_EN_CB9365795.htm
    'DECANOIC_ACID':893,
    '5-PHENYLVALERIC_ACID':1029.2, # https://www.chemicalbook.com/ChemicalProductProperty_EN_CB7109829.htm
    '2-NAPHTHOIC_ACID':1080, # https://www.chemicalbook.com/ChemicalProductProperty_EN_CB2363217.htm
    'CYCLOHEXANEPENTANOIC_ACID':1027.4,
    'LAURIC_ACID':883,
    'DIPHENYLACETIC_ACID':1257, # https://www.chemicalbook.com/ChemicalProductProperty_EN_CB0153782.htm
    'DIPHENYLVALERIC_ACID': 1200, # Não foi encontrada densidade para este ácido
    "MYRISTIC_ACID":862.2,
    "PALMITIC_ACID":852.7,
    "HEXADECANOIC_ACID":852.7,
    "STEARIC_ACID":940.8,
    "OCTADECANOIC_ACID":940.8,
    "OLEIC_ACID":895,
    "LINOLEIC_ACID":900.7,
    "LINOLENIC_ACID":916.4,
    "BEHENIC_ACID":822.1,
    "FORMIC_ACID": 1220,
    "MALONIC_ACID": 1619,
    "BUTYRIC_ACID": 959,
    "TARTARIC_ACID": 1790,
    "LACTIC_ACID": 1206,
    "CITRIC_ACID": 1665,
    'OCTANOIC_ACID': 910,
    "LEVULINIC_ACID": 1136,

    # Bases and salts
    'SODIUM_HYDROXIDE': 1000, # water's for simplicity
    'NAOH': 1000,
    'NACL': 1000,
    'SODIUM_CHLORIDE': 1000,
    "1,2-DICHLOROETHANE": 1245,
    
    # FAMEs
    "METHYL_PALMITATE": 852,    # 97% solution sigma aldrich
    # 9,12-Octadecadienoic acid (Z,Z)-, methyl ester
    "METHYL_LINOLEATE": 889,  # 98% solution sigma aldrich
    # 9-Octadecenoic acid (Z)-, methyl ester
    "METHYL_OLEATE": 873.9, # pubchem
    # Octadececanoic acid, methyl ester
    "METHYL_STEARATE": 849.8,

    # Acetates
    'ISOBUTYL_ACETATE': 871,
    "BENZYL_ACETATE": 1050, # pubchem
    # Amines
    "N,N-DIETHYLANILINE": 930.7, # pubchem
    # Ketones
    "ACETONE": 784.5, # pubchem
    "2-BUTANONE": 806,
    "METHYL_ETHYL_KETONE": 806,

    # Terpenes
    "P-CYMENE": 857.3,

    "TRIOLEIN": 910.00,
    # Salts
    'QUINOLINE': 1095,
    # Ions
    'Na+': 1000, # water's for simplicity
    'NA+': 1000,
    'Cl-': 1000,
    'CL-': 1000,
    'Ca2+': 1000,  
    "Li+": 1000,
    "K+": 1000,
    "Ba2+": 1000,
    "Sr2+": 1000,
    "Cu2+": 1000,
    "Ni2+": 1000,
    "Hg2+": 1000,
    "F-": 1000,
    "Br-": 1000,
    "I-": 1000,
    "NO3-": 1000,
    "CH3COO-": 1000,
    'OH-': 1000,

    # Various
    "TERT-BUTYL_CHLORIDE": 851,
    }

# ALGUMAS SUBSTÂNCIAS TÊM pKa VARIANDO TAMBÉM EM FUNÇÃO DO pH
# TENTAR CRIAR UM BANCO DE DADOS PARA ISSO TAMBÉM
pKa_list : Dict[str, float] = {'WATER': 14,
    '1-OCTANOL': 16.84,
    'BENZOIC_ACID': 4.19,
    'N-HEPTANE': 14,
    'TOLUENE': 41,
    'CYCLOHEXANEACETIC_ACID': 4.94,
    'PENTACHLOROPHENOL': 7.35,
    'ACETIC_ACID': 4.76,
    'OXALIC_ACID': 4.4,
    'GLYCOLIC_ACID': 3.83,
    'MALIC_ACID': 5.03,
    'SUCCINIC_ACID': 4.21,
    'VALERIC_ACID': 4.84,
    'CYCLOHEXANOACETIC_ACID': 4.89,
    'CYCLOHEXANECARBOXILIC_ACID': 4.89,
    'HEPTANOIC_ACID':4.4,
    '1,4_CYCLOHEXANEDICARBOXYLIC_ACID':4.38, # https://www.chemicalbook.com/ChemicalProductProperty_EN_CB9128589.htm
    '1-METHYLCYCLOHEXANECARBOXYLIC_ACID':5.13, # https://www.chemicalbook.com/ChemicalProductProperty_EN_CB9365795.
    "4-HEPTYLBENZOIC_ACID":4.36, # https://chemdad.com/index.php?c=article&id=50117
    'DECANOIC_ACID':4.9,
    '5-PHENYLVALERIC_ACID':4.56,
    '2-NAPHTHOIC_ACID':4.17, # https://www.chemicalbook.com/ChemicalProductProperty_EN_CB2363217.htm
    'CYCLOHEXANEPENTANOIC_ACID': 4.9, # pka indisponível (valor fictício). Ideia: modelo de pKa
    'LAURIC_ACID':5.3,
    'DIPHENYLACETIC_ACID':3.94, # https://www.chemicalbook.com/ChemicalProductProperty_EN_CB0153782.htm
    'DIPHENYLVALERIC_ACID': 4.9, # Não foi encontrado o pKa para este ácido.
    "MYRISTIC_ACID":4.9,
    "PALMITIC_ACID":4.95,
    "HEXADECANOIC_ACID": 4.95,
    "STEARIC_ACID":4.75,
    "OCTADECANOIC_ACID": 4.75,
    "OLEIC_ACID":5.02,
    "LINOLEIC_ACID":4.77,
    "BEHENIC_ACID":4.7,
    'TETRACHLOROMETHANE': 0,
    "FORMIC_ACID": 3.742,
    "MALONIC_ACID": 2.85,
    "BUTYRIC_ACID": 4.82,
    "LACTIC_ACID": 3.86,
    "CITRIC_ACID": 2.79,
    "TARTARIC_ACID": 2.98,
    "P-CYMENE": 0,
    'HYDROCHLORIC_ACID': -6.3,
    'HCL': -6.3,
    'HCl': -6.3,
    }
pKa = pKa_list # Para ir substituindo aos poucos 'pKa_list' por 'pKa' no código

dielectric_constants : Dict[str, float] = {'WATER': 78.54,
    'BENZOIC_ACID': 6.6,            # estimated
    'N-HEPTANE': 1.92,
    'TOLUENE': 2.38,
    'CYCLOHEXANEACETIC_ACID': 4.0,   # estimated
    'Na+': 78.54, # water for simplicity
    'Cl-': 78.54, # water for simplicity
    'Ca2+': 78.54, # water for simplicity
    'HYDROCHLORIC_ACID': 78.54, # water for simplicity 
    'HCL': 78.54, # water for simplicity
    'HCl': 78.54, # water for simplicity
    'OH-': 78.54, # water for simplicity
    }

# Ionic Radii values references:
# [1] YIZHAK MARCUS - IONIC RADII IN AQUEOUS SOLUTIONS - Chem. Rev. 1988, 88, 1475–1498
# [2] Revised Effective Ionic Radii and Systematic Studies of Interatomie Distances
# in Halides and Chaleogenides - R. D. Shannon - Acta Cryst. (1976). A32, 751-767
ion_radii : Dict[str, float] = {
    'Na+': 0.102,  # nm (Shannon radii for hydrated ions)
    'Cl-': 0.181,  # nm
    'Ca2+': 0.100,  # nm
    'OH-': 0.132,  # nm
}

pdh_charge : Dict[str, int] = {
    'WATER': 0,
    'SODIUM': +1,
    'Na+': +1,
    'CHLORIDE': -1,
    'Cl-': -1,
    'Na+': +1,
    'Cl-': -1,
    'Ca2+': +2,
    'OH-': -1,
}

@dataclass(frozen=True)
class DensityConstant:
    rho_g_L: float
    T_ref_K: Optional[float] = None
    source: str = "pubchem"

    def rho(self, T_K: float) -> float:
        return self.rho_g_L

@dataclass(frozen=True)
class DensityDIPPR105:
    A: float
    B: float
    C: float
    D: float
    Tmin_K: float
    Tmax_K: float
    source: str = "dippr"

    def rho(self, T_K: float) -> float:
        # fórmula DIPPR 105 (LiqDen.sac)
        # return ...
        raise NotImplementedError

# constants.py
def rho_g_L(name: str, T_K: float, density_method: str = "const", fallback_to_const: bool = True) -> float:
    """
    density_method:
      - "const": usa densities[name] (g/L)
      - "dippr": usa LiqDen_by_name.csv + Thermo.c (105/116/-1)
      - "auto": tenta dippr; se não tiver, cai no const
    """
    density_method = density_method.lower().strip()

    def _rho_const() -> float:
        rho = densities.get(name)
        if rho is None:
            raise KeyError(f"Missing constant density for {name}")
        return float(rho)

    if density_method in ("const", "constant"):
        rho = densities.get(name)
        if rho is None:
            raise KeyError(f"Missing constant density for {name}")
        return float(rho)

    if density_method in ("dippr", "dippr_liqden"):
        db = _load_dippr_liqden_db()
        row = db.get(name)
        if row is None:
            # raise KeyError(f"Missing DIPPR LiqDen entry for {name}")
            if fallback_to_const:
                return _rho_const()
            raise KeyError(f"Missing DIPPR LiqDen entry for {name}")
        
        den_kmol_m3 = rho_kmol_m3_from_dippr(T_K, row)
        # rho(kg/m3)=den(kmol/m3)*MW(kg/kmol); numericamente == g/L
        return den_kmol_m3 * row.MW

    if density_method == "auto":
        db = _load_dippr_liqden_db()
        if name in db:
            row = db[name]
            den_kmol_m3 = rho_kmol_m3_from_dippr(T_K, row)
            return den_kmol_m3 * row.MW
        rho = densities.get(name)
        if rho is None:
            raise KeyError(f"Missing density for {name} (auto tried dippr then const)")
        return float(rho)

    raise ValueError(f"Unknown density_method={density_method!r}")


# constants.py
def rho_kmol_m3_from_dippr(T_K: float, row: DipprLiqDenRow) -> float:
    if row.dippr_id == 105:
        tau = 1.0 - T_K / row.A3
        den = row.A1 / (row.A2 ** (1.0 + (tau ** row.A4)))  # kmol/m3
        return den

    if row.dippr_id == 116:
        tau = 1.0 - T_K / row.Tc_K
        den = (
            row.Dc
            + row.A1 * (tau ** 0.35)
            + row.A2 * (tau ** (2.0 / 3.0))
            + row.A3 * tau
            + row.A4 * (tau ** (4.0 / 3.0))
        )  # kmol/m3
        return den

    if row.dippr_id == -1:
        # Thermo.c: den = A1/MW *1E3   (kmol/m3)
        # (A1 ali está em g/L ≡ kg/m3)
        den = (row.A1 / row.MW) * 1.0e3
        return den

    raise ValueError(f"Unsupported DIPPR_ID={row.dippr_id} for {row.name_key}")

def mw_g_mol(name: str) -> float:
    mw = molecular_weights.get(name)
    if mw is None or not math.isfinite(float(mw)):
        raise KeyError(f"Missing molecular weight for {name}")
    return float(mw)


def molar_volume_cm3_mol(name: str, T_K: float) -> float:
    """
    Molar volume from density:
        rho in g/L  -> rho_g_cm3 = rho/1000
        Vm (cm3/mol) = MW (g/mol) / rho_g_cm3
                     = 1000 * MW / rho_g_L
    """
    rho = rho_g_L(name, T_K)  # g/L
    if rho is None or (isinstance(rho, float) and not math.isfinite(rho)) or float(rho) <= 0.0:
        raise ValueError(f"Invalid density for {name}: {rho}")
    MW = mw_g_mol(name)
    return 1000.0 * MW / float(rho)


def molecular_volume_A3(name: str, T_K: float) -> float:
    """
    Molecular volume (Å^3 / molecule) from Vm (cm3/mol):
        V_molecule = Vm * (1e24 Å^3 / cm^3) / N_A
    """
    Vm_cm3_mol = molar_volume_cm3_mol(name, T_K)
    return Vm_cm3_mol * CM3_TO_A3 / N_A


def Vmix_molar_cm3_mol(components: Iterable[str], x: Iterable[float], T_K: float) -> float:
    """
    Ideal mixing molar volume:
        Vmix = sum_i x_i * Vm_i
    """
    comps = list(components)
    xs = list(x)
    if len(comps) != len(xs):
        raise ValueError("components and x must have same length")
    s = 0.0
    for ci, xi in zip(comps, xs):
        s += float(xi) * molar_volume_cm3_mol(ci, T_K)
    return s


def Vmix_molecular_A3(components: Iterable[str], x: Iterable[float], T_K: float) -> float:
    """
    Same as Vmix_molar_cm3_mol, but expressed per molecule in Å^3.
    """
    return Vmix_molar_cm3_mol(components, x, T_K) * CM3_TO_A3 / N_A
