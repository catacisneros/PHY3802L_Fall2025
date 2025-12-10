# =========================================
# Imports & Setup
# =========================================
import numpy as np
import LT.box as B
import matplotlib.pyplot as plt

# =========================================
# Load Data
# =========================================
b_data = B.get_file('Data_blue.data')
g_data = B.get_file('Data_green.data')
y_data = B.get_file('Data_yellow.data')
v_data = B.get_file('Data_violet.data')
uv_data = B.get_file('Data_uv.data')     # make sure the file name matches exactly

# =========================================
# Arrays
# =========================================
b_v, b_I = b_data['V'], b_data['I']
g_v, g_I = g_data['V'], g_data['I']
y_v, y_I = y_data['V'], y_data['I']
v_v, v_I = v_data['V'], v_data['I']
uv_v, uv_I = uv_data['V'], uv_data['I']

# =========================================
# Uncertainties
# =========================================
s_v = 0.0005
s_I = 0.0005
dy = 0.0005
y_dy = np.full_like(y_I, dy)
g_dy = np.full_like(g_I, dy)
b_dy = np.full_like(b_I, dy)
v_dy = np.full_like(v_I, dy)
uv_dy = np.full_like(uv_I, dy)


# =========================================
# Collected Data: Raw Measurements
# =========================================
# --- 1. Overview Plot (All Wavelengths) ---
B.pl.figure()
B.plot_exp(x=y_v, y=y_I, color='orange', dy=y_dy, label='Yellow (578 nm)')
B.plot_exp(x=g_v, y=g_I, color='green', dy=g_dy, label='Green (546 nm)')
B.plot_exp(x=b_v, y=b_I, color='blue', dy=b_dy, label='Blue (436 nm)')
B.plot_exp(x=v_v, y=v_I, color='purple', dy=v_dy, label='Violet (405 nm)')
B.pl.xlabel('Voltage (V)')
B.pl.ylabel('Current (mA)')
B.pl.title('Voltage vs. Current All Wavelengths')
B.pl.legend()
B.pl.show()

# --- 2. Yellow Light (578 nm) ---
B.pl.figure()
B.plot_exp(x=y_v, y=y_I, dy=y_dy, color='orange')
B.pl.xlabel('Voltage (V)')
B.pl.ylabel('Current (mA)')
B.pl.title('Voltage vs. Current Yellow (578 nm)')
B.pl.ylim(-0.1, None)
B.pl.show()

# --- 3. Green Light (546 nm) ---
B.pl.figure()
B.plot_exp(x=g_v, y=g_I, dy=g_dy, color='green')
B.pl.xlabel('Voltage (V)')
B.pl.ylabel('Current (mA)')
B.pl.title('Voltage vs. Current Green (546 nm)')
B.pl.ylim(-0.25, None)
B.pl.show()

# --- 4. Blue Light (436 nm) ---
B.pl.figure()
B.plot_exp(x=b_v, y=b_I, dy=b_dy, color='blue')
B.pl.xlabel('Voltage (V)')
B.pl.ylabel('Current (mA)')
B.pl.title('Voltage vs. Current Blue (436 nm)')
B.pl.ylim(-0.45, None)
B.pl.show()

# --- 5. Violet Light (405 nm) ---
B.pl.figure()
B.plot_exp(x=v_v, y=v_I, dy=v_dy, color='purple')
B.pl.xlabel('Voltage (V)')
B.pl.ylabel('Current (mA)')
B.pl.title('Voltage vs. Current Violet (405 nm)')
B.pl.ylim(-0.5, None)
B.pl.show()

# =========================================
# YELLOW (578 nm)
# =========================================
B.pl.figure()
B.plot_exp(x=y_v, y=y_I, color='orange', dy=y_dy)
B.pl.xlabel('Voltage (V)')
B.pl.ylabel('Current (mA)')
B.pl.title('Voltage vs. Current (578 nm)')

# Fits (baseline + two cutoff regions)
y_vb = B.in_between(1, 4, y_v)
y_basel = B.linefit(y_v[y_vb], y_I[y_vb], y_dy[y_vb])

y_Is1 = (0.05 <= y_I) & (y_I <= 0.30)
y_Il1 = B.linefit(y_v[y_Is1], y_I[y_Is1], y_dy[y_Is1])

y_Is2 = (0.00 <= y_I) & (y_I <= 0.10)
y_Il2 = B.linefit(y_v[y_Is2], y_I[y_Is2], y_dy[y_Is2])

# Draw extended lines
x_fit = np.linspace(np.min(y_v), np.max(y_v), 200)
m0, b0 = y_basel.slope, y_basel.offset
m1, b1 = y_Il1.slope, y_Il1.offset
m2, b2 = y_Il2.slope, y_Il2.offset

B.pl.plot(x_fit, m0 * x_fit + b0, color='red', label='Base fit (1-4 V)')
B.pl.plot(x_fit, m1 * x_fit + b1, color='pink', label='Secondary fit #1')
B.pl.plot(x_fit, m2 * x_fit + b2, color='purple', label='Secondary fit #2')

B.pl.ylim(-0.10, None)
B.pl.xlim(None, 1.5)

# Intersections + uncertainties
den1 = (m0 - m1)
x1 = (b1 - b0) / den1
y1 = m0 * x1 + b0
sx1 = np.sqrt((y_basel.sigma_o / den1)**2 + (y_Il1.sigma_o / den1)**2
              + (((b1 - b0) * y_basel.sigma_s) / den1**2)**2
              + (((b1 - b0) * y_Il1.sigma_s) / den1**2)**2)

den2 = (m0 - m2)
x2 = (b2 - b0) / den2
y2 = m0 * x2 + b0
sx2 = np.sqrt((y_basel.sigma_o / den2)**2 + (y_Il2.sigma_o / den2)**2
              + (((b2 - b0) * y_basel.sigma_s) / den2**2)**2
              + (((b2 - b0) * y_Il2.sigma_s) / den2**2)**2)

# Error bars (no markers) + weighted average
B.pl.errorbar([x1], [y1], xerr=[sx1], fmt='none', ecolor='black', capsize=3, label='V_s #1')
B.pl.errorbar([x2], [y2], xerr=[sx2], fmt='none', ecolor='gray', capsize=3, label='V_s #2')

w1, w2 = 1.0/sx1**2, 1.0/sx2**2
Vs_avg = (w1 * x1 + w2 * x2) / (w1 + w2)
s_avg = 1.0/np.sqrt(w1 + w2)

B.pl.axvline(Vs_avg, linestyle='--', linewidth=1, color='black', alpha=0.5, label='Avg V_s')
B.pl.legend(); B.pl.show()

print()
print(f"Yellow V_s #1 = {x1:.4f} ± {sx1:.4f} V")
print(f"Yellow V_s #2 = {x2:.4f} ± {sx2:.4f} V")
print(f"Yellow V_s (weighted avg) = {Vs_avg:.4f} ± {s_avg:.4f} V")

# =========================================
# GREEN (546 nm)
# =========================================
B.pl.figure()
B.plot_exp(x=g_v, y=g_I, color='green', dy=g_dy)
B.pl.xlabel('Voltage (V)')
B.pl.ylabel('Current (mA)')
B.pl.title('Voltage vs. Current (546 nm)')

g_vb = B.in_between(1, 4, g_v)
g_basel = B.linefit(g_v[g_vb], g_I[g_vb], g_dy[g_vb])

g_Is1 = (0.20 <= g_I) & (g_I <= 0.80)
g_Il1 = B.linefit(g_v[g_Is1], g_I[g_Is1], g_dy[g_Is1])

g_Is2 = (0.00 <= g_I) & (g_I <= 0.30)
g_Il2 = B.linefit(g_v[g_Is2], g_I[g_Is2], g_dy[g_Is2])

x_fit = np.linspace(np.min(g_v), np.max(g_v), 200)
m0, b0 = g_basel.slope, g_basel.offset
m1, b1 = g_Il1.slope, g_Il1.offset
m2, b2 = g_Il2.slope, g_Il2.offset

B.pl.plot(x_fit, m0 * x_fit + b0, color='red', label='Base fit (1-4 V)')
B.pl.plot(x_fit, m1 * x_fit + b1, color='pink', label='Secondary fit #1')
B.pl.plot(x_fit, m2 * x_fit + b2, color='purple', label='Secondary fit #2')

B.pl.ylim(-0.25, None)
B.pl.xlim(None, 1.5)

den1 = (m0 - m1)
x1 = (b1 - b0) / den1
y1 = m0 * x1 + b0
sx1 = np.sqrt((g_basel.sigma_o / den1)**2 + (g_Il1.sigma_o / den1)**2
              + (((b1 - b0) * g_basel.sigma_s) / den1**2)**2
              + (((b1 - b0) * g_Il1.sigma_s) / den1**2)**2)

den2 = (m0 - m2)
x2 = (b2 - b0) / den2
y2 = m0 * x2 + b0
sx2 = np.sqrt((g_basel.sigma_o / den2)**2 + (g_Il2.sigma_o / den2)**2
              + (((b2 - b0) * g_basel.sigma_s) / den2**2)**2
              + (((b2 - b0) * g_Il2.sigma_s) / den2**2)**2)

B.pl.errorbar([x1], [y1], xerr=[sx1], fmt='none', ecolor='black', capsize=3, label='V_s #1')
B.pl.errorbar([x2], [y2], xerr=[sx2], fmt='none', ecolor='gray', capsize=3, label='V_s #2')

w1, w2 = 1.0/sx1**2, 1.0/sx2**2
Vs_avg = (w1 * x1 + w2 * x2) / (w1 + w2)
s_avg = 1.0/np.sqrt(w1 + w2)

B.pl.axvline(Vs_avg, linestyle='--', linewidth=1, color='black', alpha=0.5, label='Avg V_s')

B.pl.legend(); B.pl.show()

print()
print(f"Green V_s #1 = {x1:.4f} ± {sx1:.4f} V")
print(f"Green V_s #2 = {x2:.4f} ± {sx2:.4f} V")
print(f"Green V_s (weighted avg) = {Vs_avg:.4f} ± {s_avg:.4f} V")

# =========================================
# BLUE (436 nm)
# =========================================
B.pl.figure()
B.plot_exp(x=b_v, y=b_I, color='blue', dy=b_dy)
B.pl.xlabel('Voltage (V)')
B.pl.ylabel('Current (mA)')
B.pl.title('Voltage vs. Current (436 nm)')

b_vb = B.in_between(2, 4, b_v)
b_basel = B.linefit(b_v[b_vb], b_I[b_vb], b_dy[b_vb])

# --- Blue (436 nm) ---
b_Is1 = (0.50 <= b_I) & (b_I <= 2.50)
b_Il1 = B.linefit(b_v[b_Is1], b_I[b_Is1], b_dy[b_Is1])

b_Is2 = (0.00 <= b_I) & (b_I <= 1.00)
b_Il2 = B.linefit(b_v[b_Is2], b_I[b_Is2], b_dy[b_Is2])


x_fit = np.linspace(np.min(b_v), np.max(b_v), 200)
m0, b0 = b_basel.slope, b_basel.offset
m1, b1 = b_Il1.slope, b_Il1.offset
m2, b2 = b_Il2.slope, b_Il2.offset

B.pl.plot(x_fit, m0 * x_fit + b0, color='red', label='Base fit (2-4 V)')
B.pl.plot(x_fit, m1 * x_fit + b1, color='pink', label='Secondary fit #1')
B.pl.plot(x_fit, m2 * x_fit + b2, color='purple', label='Secondary fit #2')

B.pl.ylim(-0.45, None)
B.pl.xlim(None, 2)

den1 = (m0 - m1)
x1 = (b1 - b0) / den1
y1 = m0 * x1 + b0
sx1 = np.sqrt((b_basel.sigma_o / den1)**2 + (b_Il1.sigma_o / den1)**2
              + (((b1 - b0) * b_basel.sigma_s) / den1**2)**2
              + (((b1 - b0) * b_Il1.sigma_s) / den1**2)**2)

den2 = (m0 - m2)
x2 = (b2 - b0) / den2
y2 = m0 * x2 + b0
sx2 = np.sqrt((b_basel.sigma_o / den2)**2 + (b_Il2.sigma_o / den2)**2
              + (((b2 - b0) * b_basel.sigma_s) / den2**2)**2
              + (((b2 - b0) * b_Il2.sigma_s) / den2**2)**2)

B.pl.errorbar([x1], [y1], xerr=[sx1], fmt='none', ecolor='black', capsize=3, label='V_s #1')
B.pl.errorbar([x2], [y2], xerr=[sx2], fmt='none', ecolor='gray', capsize=3, label='V_s #2')

w1, w2 = 1.0/sx1**2, 1.0/sx2**2
Vs_avg = (w1 * x1 + w2 * x2) / (w1 + w2)
s_avg = 1.0/np.sqrt(w1 + w2)

B.pl.axvline(Vs_avg, linestyle='--', linewidth=1, color='black', alpha=0.5, label='Avg V_s')
B.pl.legend(); B.pl.show()

print()
print(f"Blue V_s #1 = {x1:.4f} ± {sx1:.4f} V")
print(f"Blue V_s #2 = {x2:.4f} ± {sx2:.4f} V")
print(f"Blue V_s (weighted avg) = {Vs_avg:.4f} ± {s_avg:.4f} V")

# =========================================
# VIOLET (405 nm)
# =========================================
B.pl.figure()
B.plot_exp(x=v_v, y=v_I, color='purple', dy=v_dy)
B.pl.xlabel('Voltage (V)')
B.pl.ylabel('Current (mA)')
B.pl.title('Voltage vs. Current (405 nm)')

v_vb = B.in_between(2, 4, v_v)
v_basel = B.linefit(v_v[v_vb], v_I[v_vb], v_dy[v_vb])

# --- Violet (405 nm) ---
v_Is1 = (0.80 <= v_I) & (v_I <= 3.50)
v_Il1 = B.linefit(v_v[v_Is1], v_I[v_Is1], v_dy[v_Is1])

v_Is2 = (0.00 <= v_I) & (v_I <= 1.50)
v_Il2 = B.linefit(v_v[v_Is2], v_I[v_Is2], v_dy[v_Is2])


x_fit = np.linspace(np.min(v_v), np.max(v_v), 200)
m0, b0 = v_basel.slope, v_basel.offset
m1, b1 = v_Il1.slope, v_Il1.offset
m2, b2 = v_Il2.slope, v_Il2.offset

B.pl.plot(x_fit, m0 * x_fit + b0, color='red', label='Base fit (2-4 V)')
B.pl.plot(x_fit, m1 * x_fit + b1, color='pink', label='Secondary fit #1')
B.pl.plot(x_fit, m2 * x_fit + b2, color='purple', label='Secondary fit #2')

B.pl.ylim(-0.60, None)
B.pl.xlim(None, 2)

den1 = (m0 - m1)
x1 = (b1 - b0) / den1
y1 = m0 * x1 + b0
sx1 = np.sqrt((v_basel.sigma_o / den1)**2 + (v_Il1.sigma_o / den1)**2
              + (((b1 - b0) * v_basel.sigma_s) / den1**2)**2
              + (((b1 - b0) * v_Il1.sigma_s) / den1**2)**2)

den2 = (m0 - m2)
x2 = (b2 - b0) / den2
y2 = m0 * x2 + b0
sx2 = np.sqrt((v_basel.sigma_o / den2)**2 + (v_Il2.sigma_o / den2)**2
              + (((b2 - b0) * v_basel.sigma_s) / den2**2)**2
              + (((b2 - b0) * v_Il2.sigma_s) / den2**2)**2)

B.pl.errorbar([x1], [y1], xerr=[sx1], fmt='none', ecolor='black', capsize=3, label='V_s #1')
B.pl.errorbar([x2], [y2], xerr=[sx2], fmt='none', ecolor='gray', capsize=3, label='V_s #2')

w1, w2 = 1.0/sx1**2, 1.0/sx2**2
Vs_avg = (w1 * x1 + w2 * x2) / (w1 + w2)
s_avg = 1.0/np.sqrt(w1 + w2)

B.pl.axvline(Vs_avg, linestyle='--', linewidth=1, color='black', alpha=0.5, label='Avg V_s')
B.pl.legend(); B.pl.show()

print()
print(f"Violet V_s #1 = {x1:.4f} ± {sx1:.4f} V")
print(f"Violet V_s #2 = {x2:.4f} ± {sx2:.4f} V")
print(f"Violet V_s (weighted avg) = {Vs_avg:.4f} ± {s_avg:.4f} V")


# =========================================
# ULTRAVIOLET (365 nm)
# =========================================


# --- Overview plot for UV ---
B.pl.figure()
B.plot_exp(x=uv_v, y=uv_I, color='magenta', dy=uv_dy)
B.pl.xlabel('Voltage (V)')
B.pl.ylabel('Current (mA)')
B.pl.title('Voltage vs. Current (365 nm)')
B.pl.ylim(-0.6, None)
B.pl.xlim(None, 2.5)
B.pl.show()

# --- Baseline fit (flat region at high voltage) ---
uv_vb = B.in_between(2, 4, uv_v)
uv_basel = B.linefit(uv_v[uv_vb], uv_I[uv_vb], uv_dy[uv_vb])

# --- Two near-cutoff linear fits (adjust ranges as needed) ---
uv_Is1 = (1.0 <= uv_I) & (uv_I <= 4.0)
uv_Il1 = B.linefit(uv_v[uv_Is1], uv_I[uv_Is1], uv_dy[uv_Is1])

uv_Is2 = (0.0 <= uv_I) & (uv_I <= 1.5)
uv_Il2 = B.linefit(uv_v[uv_Is2], uv_I[uv_Is2], uv_dy[uv_Is2])

# --- Plot fits and extended lines ---
x_fit = np.linspace(np.min(uv_v), np.max(uv_v), 200)
m0, b0 = uv_basel.slope, uv_basel.offset
m1, b1 = uv_Il1.slope, uv_Il1.offset
m2, b2 = uv_Il2.slope, uv_Il2.offset

B.pl.figure()
B.plot_exp(x=uv_v, y=uv_I, dy=uv_dy, color='magenta')
B.pl.plot(x_fit, m0*x_fit + b0, color='red', label='Baseline fit (2–4 V)')
B.pl.plot(x_fit, m1*x_fit + b1, color='pink', label='Secondary fit #1')
B.pl.plot(x_fit, m2*x_fit + b2, color='purple', label='Secondary fit #2')
B.pl.xlabel('Voltage (V)')
B.pl.ylabel('Current (mA)')
B.pl.title('Voltage vs. Current (365 nm, UV)')
B.pl.legend()

# --- Intersections + uncertainties ---
den1 = (m0 - m1)
x1 = (b1 - b0) / den1
y1 = m0*x1 + b0
sx1 = np.sqrt((uv_basel.sigma_o / den1)**2 + (uv_Il1.sigma_o / den1)**2
              + (((b1 - b0) * uv_basel.sigma_s) / den1**2)**2
              + (((b1 - b0) * uv_Il1.sigma_s) / den1**2)**2)

den2 = (m0 - m2)
x2 = (b2 - b0) / den2
y2 = m0*x2 + b0
sx2 = np.sqrt((uv_basel.sigma_o / den2)**2 + (uv_Il2.sigma_o / den2)**2
              + (((b2 - b0) * uv_basel.sigma_s) / den2**2)**2
              + (((b2 - b0) * uv_Il2.sigma_s) / den2**2)**2)

B.pl.errorbar([x1], [y1], xerr=[sx1], fmt='none', ecolor='black', capsize=3, label='V_s #1')
B.pl.errorbar([x2], [y2], xerr=[sx2], fmt='none', ecolor='gray', capsize=3, label='V_s #2')

# --- Weighted average ---
w1, w2 = 1.0/sx1**2, 1.0/sx2**2
Vs_avg = (w1*x1 + w2*x2) / (w1 + w2)
s_avg = 1.0 / np.sqrt(w1 + w2)

B.pl.axvline(Vs_avg, linestyle='--', linewidth=1, color='black', alpha=0.5, label='Avg V_s')
B.pl.legend(); B.pl.show()

print()
print(f"UV (365 nm) V_s #1 = {x1:.4f} ± {sx1:.4f} V")
print(f"UV (365 nm) V_s #2 = {x2:.4f} ± {sx2:.4f} V")
print(f"UV (365 nm) V_s (weighted avg) = {Vs_avg:.4f} ± {s_avg:.4f} V")


# =========================================
# Global Fits (3 cases): V_s (eV) vs 1/λ (m)
# - Vs: mean/average case
# - Vs1: high-Vs case
# - Vs2: low-Vs case
# =========================================

# Constants (already defined above, kept here for clarity)
eC = 1.60217663e-19  # J/eV
c = 2.99792458e8     # m/s

# Wavelengths, inverse wavelengths
wave_nm = np.array([578, 546, 436, 405, 365])
wave_m = wave_nm * 1e-9
inv_wave = 1.0 / wave_m

# Data sets (already defined above, repeated here for the 3-fit block)
Vs   = np.array([0.7004, 0.8387, 0.9186, 1.0157, 1.4091])
s_Vs = np.array([0.0072, 0.0238, 0.0004, 0.0005, 0.0005])

Vs1   = np.array([0.6530, 0.6831, 0.8913, 0.9476, 1.3876])
s_Vs1 = np.array([0.0028, 0.0025, 0.0004, 0.0006, 0.0006])

Vs2   = np.array([2.2918, 2.2744, 1.1995, 1.3467, 1.5492])
s_Vs2 = np.array([0.0730, 0.0133, 0.0013, 0.0013, 0.0015])


def fit_and_report(label, x_inv_lambda, y_Vs, y_sig, color):
    """Weighted linear fit: V_s = (hc)*(1/λ) - ϕ. Returns (slope,
    intercept, slope_err, intercept_err).
    """
    w = 1.0 / y_sig
    p, cov = np.polyfit(x_inv_lambda, y_Vs, 1, w=w, cov=True)
    slope, intercept = p
    slope_err, intercept_err = np.sqrt(np.diag(cov))
    
    # Derived quantities
    hc_eV_m = slope
    dhc_eV_m = slope_err
    hc_eV_nm = slope * 1e9
    dhc_eV_nm = slope_err * 1e9
    
    phi_eV = -intercept
    dphi_eV = intercept_err
    
    h_eV_s = slope / c
    dh_eV_s = slope_err / c
    h_J_s = h_eV_s * eC
    dh_J_s = dh_eV_s * eC
    
    print(f"\n[{label}]")
    print(f"hc = {hc_eV_nm:.3f} ± {dhc_eV_nm:.3f} eV·nm (={hc_eV_m:.6e} ± {dhc_eV_m:.6e} eV·m)")
    print(f"ϕ = {phi_eV:.3f} ± {dphi_eV:.3f} eV")
    print(f"h = {h_J_s:.6e} ± {dh_J_s:.6e} J·s (={h_eV_s:.6e} ± {dh_eV_s:.6e} eV·s)")
    
    return slope, intercept, slope_err, intercept_err

# ---- Individual plots + reports (one figure per case) ----
# Mean case
plt.figure()
plt.errorbar(inv_wave, Vs, yerr=s_Vs, fmt='o', capsize=3, label='mean data', color='tab:blue')
s_m, b_m, ds_m, db_m = fit_and_report("Mean (Vs)", inv_wave, Vs, s_Vs, color='tab:blue')
xg = np.linspace(inv_wave.min(), inv_wave.max(), 200)
plt.plot(xg, s_m * xg + b_m, color='tab:blue', label='mean fit')
plt.xlabel('1/λ (m$^{-1}$)'); plt.ylabel('Stopping energy (eV)')
plt.title('Photoelectric: V_s (eV) vs 1/λ Mean')
plt.legend(); plt.show()

# High-Vs case
plt.figure()
plt.errorbar(inv_wave, Vs1, yerr=s_Vs1, fmt='o', capsize=3, label='high-Vs data', color='tab:orange')
s_h, b_h, ds_h, db_h = fit_and_report("High (Vs1)", inv_wave, Vs1, s_Vs1, color='tab:orange')
xg = np.linspace(inv_wave.min(), inv_wave.max(), 200)
plt.plot(xg, s_h * xg + b_h, color='tab:orange', label='high-Vs fit')
plt.xlabel('1/λ (m$^{-1}$)'); plt.ylabel('Stopping energy (eV)')
plt.title('Photoelectric: V_s (eV) vs 1/λ High')
plt.legend(); plt.show()

# Low-Vs case
plt.figure()
plt.errorbar(inv_wave, Vs2, yerr=s_Vs2, fmt='o', capsize=3, label='low-Vs data', color='tab:green')
s_l, b_l, ds_l, db_l = fit_and_report("Low (Vs2)", inv_wave, Vs2, s_Vs2, color='tab:green')
xg = np.linspace(inv_wave.min(), inv_wave.max(), 200)
plt.plot(xg, s_l * xg + b_l, color='tab:green', label='low-Vs fit')
plt.xlabel('1/λ (m$^{-1}$)'); plt.ylabel('Stopping energy (eV)')
plt.title('Photoelectric: V_s (eV) vs 1/λ Low')
plt.legend(); plt.show()

# ---- Optional: one overlay figure to compare all three at once ----
plt.figure()
xg = np.linspace(inv_wave.min(), inv_wave.max(), 200)

# Re-plot points (lighter) for context
plt.errorbar(inv_wave, Vs, yerr=s_Vs, fmt='o', capsize=3, label='mean data', color='tab:blue', alpha=0.6)
plt.errorbar(inv_wave, Vs1, yerr=s_Vs1, fmt='o', capsize=3, label='high-Vs data', color='tab:orange', alpha=0.6)
plt.errorbar(inv_wave, Vs2, yerr=s_Vs2, fmt='o', capsize=3, label='low-Vs data', color='tab:green', alpha=0.6)

# Draw the three best-fit lines
plt.plot(xg, s_m * xg + b_m, color='tab:blue', label='mean fit')
plt.plot(xg, s_h * xg + b_h, color='tab:orange', label='high-Vs fit')
plt.plot(xg, s_l * xg + b_l, color='tab:green', label='low-Vs fit')

plt.xlabel('1/λ (m$^{-1}$)')
plt.ylabel('Stopping energy (eV)')
plt.title('Photoelectric: V_s (eV) vs 1/λ (Mean, High, Low)')
plt.legend(); plt.show()




