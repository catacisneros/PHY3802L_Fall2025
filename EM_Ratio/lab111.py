import LT.box as B
import numpy as np

# -----------------------------
# Experiment constants
# -----------------------------
cm2m = 0.01
N   = 132.0
dN  = 0.0
Dc  = 30.5 * cm2m
dDc = 1.0 * cm2m
R   = Dc/2.0
dR  = dDc/2.0
mu0 = 4.0*np.pi*1.0e-7
fac = (4.0/5.0)**1.5

dV_inst = 1.0     # volts
dI_inst = 0.01    # amps

EOM_ACCEPTED = 1.75882001076e11  # C/kg

datasets = [
    ("emV200.data", 200.0),
    ("emV250.data", 250.0),
    ("emV300.data", 300.0),
    ("emV400.data", 400.0),
]

# -----------------------------
# Helper functions
# -----------------------------
def k_helmholtz():
    k = mu0 * N / R * fac
    dk = fac * mu0 * (N / R**2) * dR  # simplified uncertainty
    return k, dk

def e_over_m_from_point(V, I, r, dV, dI, dr):
    k, dk = k_helmholtz()
    Bfield = k * I
    eom = 2.0 * V / (Bfield**2 * r**2)
    return eom

def weighted_mean(values, sigmas):
    w = 1.0 / (sigmas**2)
    mean = np.sum(w * values) / np.sum(w)
    s_mean = np.sqrt(1.0 / np.sum(w))
    return mean, s_mean

def percent_diff(x, ref):
    return 100.0 * np.abs(x - ref) / ref

# -----------------------------
# Main loop
# -----------------------------
all_eom = []
all_seom = []

for data_file, V in datasets:
    md = B.get_file(data_file)

    I  = md['I']
    D  = md['D'] * cm2m
    dD = md['dD'] * cm2m

    r  = 0.5 * D
    dr = 0.5 * dD

    dI = np.full_like(I, dI_inst, dtype=float)
    dV = np.full_like(I, dV_inst, dtype=float)

    eom_i = []
    for idx in range(len(I)):
        eom_i.append(e_over_m_from_point(V, float(I[idx]), float(r[idx]),
                                         float(dV[idx]), float(dI[idx]), float(dr[idx])))

    eom_i = np.array(eom_i)
    seom_i = np.full_like(eom_i, np.std(eom_i)/np.sqrt(len(eom_i)))  # rough errors

    # weighted mean
    eom_w, seom_w = weighted_mean(eom_i, seom_i)

    all_eom.extend(eom_i)
    all_seom.extend(seom_i)

    # Plot e/m vs I
    B.plot_exp(I, eom_i, seom_i)
    B.pl.xlabel("Coil Current I [A]")
    B.pl.ylabel("e/m [C/kg]")
    B.pl.title(f"e/m vs I at V = {V:.0f} V")

    print(f"V = {V:.0f} V: e/m = {eom_w:.3e} ± {seom_w:.3e} C/kg "
          f"({percent_diff(eom_w, EOM_ACCEPTED):.1f}% diff)")

# -----------------------------
# Overall mean
# -----------------------------
all_eom = np.array(all_eom)
all_seom = np.array(all_seom)

eom_all_w, seom_all_w = weighted_mean(all_eom, all_seom)
print(f"Overall weighted mean e/m = {eom_all_w:.3e} ± {seom_all_w:.3e} C/kg "
      f"({percent_diff(eom_all_w, EOM_ACCEPTED):.1f}% diff)")


