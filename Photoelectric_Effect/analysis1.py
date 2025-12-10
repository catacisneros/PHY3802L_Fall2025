
# =========================================
# Imports & Setup
# =========================================
import numpy as np
import LT.box as B
import matplotlib.pyplot as plt

# =========================================
# Load Data
# =========================================
y_data = B.get_file('Data_yellow.data')
g_data = B.get_file('Data_green.data')
b_data = B.get_file('Data_blue.data')
v_data = B.get_file('Data_violet.data')
uv_data = B.get_file('Data_uv.data')  # ← NEW

# =========================================
# Arrays
# =========================================
y_v, y_I = y_data['V'], y_data['I']
g_v, g_I = g_data['V'], g_data['I']
b_v, b_I = b_data['V'], b_data['I']
v_v, v_I = v_data['V'], v_data['I']
uv_v, uv_I = uv_data['V'], uv_data['I']  # ← NEW

# =========================================
# Uncertainties
# =========================================
dy = 0.0005
y_dy  = np.full_like(y_I, dy)
g_dy  = np.full_like(g_I, dy)
b_dy  = np.full_like(b_I, dy)
v_dy  = np.full_like(v_I, dy)
uv_dy = np.full_like(uv_I, dy)  # ← NEW

# =========================================
# Collected Data: Raw Measurements
# =========================================
B.pl.figure()
B.plot_exp(x=y_v, y=y_I, color='orange', dy=y_dy, label='Yellow (578 nm)')
B.plot_exp(x=g_v, y=g_I, color='green', dy=g_dy, label='Green (546 nm)')
B.plot_exp(x=b_v, y=b_I, color='blue', dy=b_dy, label='Blue (436 nm)')
B.plot_exp(x=v_v, y=v_I, color='purple', dy=v_dy, label='Violet (405 nm)')
B.plot_exp(x=uv_v, y=uv_I, color='magenta', dy=uv_dy, label='Ultraviolet (365 nm)')  # ← NEW
B.pl.xlabel('Voltage (V)')
B.pl.ylabel('Current (mA)')
B.pl.title('Voltage vs. Current – All Wavelengths')
B.pl.legend()
B.pl.show()

# =========================================
# Helper Function for Line Intersections
# =========================================
def compute_vs(baseline, L1, L2):
    """Compute intersections (x1,x2) and weighted average (Vs_avg,s_avg)."""
    m0, b0 = baseline.slope, baseline.offset
    m1, b1 = L1.slope, L1.offset
    m2, b2 = L2.slope, L2.offset

    den1 = (m0 - m1)
    x1 = (b1 - b0) / den1
    y1 = m0 * x1 + b0
    sx1 = np.sqrt((baseline.sigma_o / den1)**2 + (L1.sigma_o / den1)**2
                  + (((b1 - b0) * baseline.sigma_s) / den1**2)**2
                  + (((b1 - b0) * L1.sigma_s) / den1**2)**2)

    den2 = (m0 - m2)
    x2 = (b2 - b0) / den2
    y2 = m0 * x2 + b0
    sx2 = np.sqrt((baseline.sigma_o / den2)**2 + (L2.sigma_o / den2)**2
                  + (((b2 - b0) * baseline.sigma_s) / den2**2)**2
                  + (((b2 - b0) * L2.sigma_s) / den2**2)**2)

    w1, w2 = 1.0/sx1**2, 1.0/sx2**2
    Vs_avg = (w1*x1 + w2*x2) / (w1 + w2)
    s_avg = 1.0 / np.sqrt(w1 + w2)

    return x1, sx1, x2, sx2, Vs_avg, s_avg

# =========================================
# Function to Analyze Each Color
# =========================================
def analyze_color(v, I, dy, color, label, base_range, fit1_range, fit2_range):
    """Perform baseline + 2 fits, plot, compute Vs."""
    B.pl.figure()
    B.plot_exp(x=v, y=I, dy=dy, color=color)
    B.pl.xlabel('Voltage (V)')
    B.pl.ylabel('Current (mA)')
    B.pl.title(f'Voltage vs. Current ({label})')

    vb = B.in_between(*base_range, v)
    base = B.linefit(v[vb], I[vb], dy[vb])

    s1 = (fit1_range[0] <= I) & (I <= fit1_range[1])
    L1 = B.linefit(v[s1], I[s1], dy[s1])

    s2 = (fit2_range[0] <= I) & (I <= fit2_range[1])
    L2 = B.linefit(v[s2], I[s2], dy[s2])

    x_fit = np.linspace(np.min(v), np.max(v), 200)
    B.pl.plot(x_fit, base.slope*x_fit + base.offset, color='red', label='Baseline')
    B.pl.plot(x_fit, L1.slope*x_fit + L1.offset, color='pink', label='Secondary #1')
    B.pl.plot(x_fit, L2.slope*x_fit + L2.offset, color='purple', label='Secondary #2')
    B.pl.legend()

    x1, sx1, x2, sx2, Vs_avg, s_avg = compute_vs(base, L1, L2)

    B.pl.errorbar([x1], [0], xerr=[sx1], fmt='none', ecolor='black', capsize=3, label='V_s #1')
    B.pl.errorbar([x2], [0], xerr=[sx2], fmt='none', ecolor='gray', capsize=3, label='V_s #2')
    B.pl.axvline(Vs_avg, linestyle='--', linewidth=1, color='black', alpha=0.5, label='Avg V_s')
    B.pl.legend(); B.pl.show()

    print(f"\n{label} V_s #1 = {x1:.4f} ± {sx1:.4f} V")
    print(f"{label} V_s #2 = {x2:.4f} ± {sx2:.4f} V")
    print(f"{label} V_s (weighted avg) = {Vs_avg:.4f} ± {s_avg:.4f} V")

    return Vs_avg, s_avg, x1, sx1, x2, sx2

# =========================================
# Individual Analyses
# =========================================
Y_Vs, Y_dV, Y_x1, Y_sx1, Y_x2, Y_sx2 = analyze_color(y_v, y_I, y_dy, 'orange', 'Yellow (578 nm)',
                                                     (1, 4), (0.05, 0.30), (0.00, 0.10))

G_Vs, G_dV, G_x1, G_sx1, G_x2, G_sx2 = analyze_color(g_v, g_I, g_dy, 'green', 'Green (546 nm)',
                                                     (1, 4), (0.20, 0.80), (0.00, 0.30))

B_Vs, B_dV, B_x1, B_sx1, B_x2, B_sx2 = analyze_color(b_v, b_I, b_dy, 'blue', 'Blue (436 nm)',
                                                     (2, 4), (0.50, 2.50), (0.00, 1.00))

V_Vs, V_dV, V_x1, V_sx1, V_x2, V_sx2 = analyze_color(v_v, v_I, v_dy, 'purple', 'Violet (405 nm)',
                                                     (2, 4), (0.80, 3.50), (0.00, 1.50))

UV_Vs, UV_dV, UV_x1, UV_sx1, UV_x2, UV_sx2 = analyze_color(uv_v, uv_I, uv_dy, 'magenta', 'Ultraviolet (365 nm)',
                                                           (2, 4), (1.00, 4.00), (0.00, 1.50))

# =========================================
# Global Fits: V_s (eV) vs 1/λ (m)
# =========================================
eC = 1.60217663e-19
c = 2.99792458e8

wave_nm = np.array([578, 546, 436, 405, 365])
wave_m = wave_nm * 1e-9
inv_wave = 1.0 / wave_m

Vs   = np.array([Y_Vs, G_Vs, B_Vs, V_Vs, UV_Vs])
s_Vs = np.array([Y_dV, G_dV, B_dV, V_dV, UV_dV])

Vs1   = np.array([Y_x1, G_x1, B_x1, V_x1, UV_x1])
s_Vs1 = np.array([Y_sx1, G_sx1, B_sx1, V_sx1, UV_sx1])

Vs2   = np.array([Y_x2, G_x2, B_x2, V_x2, UV_x2])
s_Vs2 = np.array([Y_sx2, G_sx2, B_sx2, V_sx2, UV_sx2])

# =========================================
# Linear Fit Function
# =========================================
def fit_and_report(label, x_inv_lambda, y_Vs, y_sig, color):
    w = 1.0 / y_sig
    p, cov = np.polyfit(x_inv_lambda, y_Vs, 1, w=w, cov=True)
    slope, intercept = p
    slope_err, intercept_err = np.sqrt(np.diag(cov))

    hc_eV_m = slope
    phi_eV = -intercept
    h_eV_s = slope / c
    h_J_s = h_eV_s * eC

    print(f"\n[{label}]")
    print(f"hc = {hc_eV_m*1e9:.3f} ± {slope_err*1e9:.3f} eV·nm")
    print(f"ϕ = {phi_eV:.3f} ± {intercept_err:.3f} eV")
    print(f"h  = {h_J_s:.6e} ± {(slope_err/c)*eC:.6e} J·s")

    return slope, intercept, slope_err, intercept_err

# =========================================
# Mean / High / Low Fits
# =========================================
plt.figure()
plt.errorbar(inv_wave, Vs, yerr=s_Vs, fmt='o', capsize=3, label='Mean data', color='tab:blue')
s_m, b_m, ds_m, db_m = fit_and_report("Mean (Vs)", inv_wave, Vs, s_Vs, 'tab:blue')
xg = np.linspace(inv_wave.min(), inv_wave.max(), 200)
plt.plot(xg, s_m*xg + b_m, color='tab:blue', label='Mean fit')
plt.xlabel('1/λ (m$^{-1}$)'); plt.ylabel('Stopping energy (eV)')
plt.title('Photoelectric: V_s vs 1/λ (Mean)')
plt.legend(); plt.show()

plt.figure()
plt.errorbar(inv_wave, Vs1, yerr=s_Vs1, fmt='o', capsize=3, label='High-Vs data', color='tab:orange')
s_h, b_h, ds_h, db_h = fit_and_report("High (Vs1)", inv_wave, Vs1, s_Vs1, 'tab:orange')
xg = np.linspace(inv_wave.min(), inv_wave.max(), 200)
plt.plot(xg, s_h*xg + b_h, color='tab:orange', label='High-Vs fit')
plt.xlabel('1/λ (m$^{-1}$)'); plt.ylabel('Stopping energy (eV)')
plt.title('Photoelectric: V_s vs 1/λ (High)')
plt.legend(); plt.show()

plt.figure()
plt.errorbar(inv_wave, Vs2, yerr=s_Vs2, fmt='o', capsize=3, label='Low-Vs data', color='tab:green')
s_l, b_l, ds_l, db_l = fit_and_report("Low (Vs2)", inv_wave, Vs2, s_Vs2, 'tab:green')
xg = np.linspace(inv_wave.min(), inv_wave.max(), 200)
plt.plot(xg, s_l*xg + b_l, color='tab:green', label='Low-Vs fit')
plt.xlabel('1/λ (m$^{-1}$)'); plt.ylabel('Stopping energy (eV)')
plt.title('Photoelectric: V_s vs 1/λ (Low)')
plt.legend(); plt.show()

# =========================================
# Overlay of All Fits
# =========================================
plt.figure()
xg = np.linspace(inv_wave.min(), inv_wave.max(), 200)
plt.errorbar(inv_wave, Vs,  yerr=s_Vs,  fmt='o', capsize=3, label='Mean', color='tab:blue',  alpha=0.6)
plt.errorbar(inv_wave, Vs1, yerr=s_Vs1, fmt='o', capsize=3, label='High', color='tab:orange', alpha=0.6)
plt.errorbar(inv_wave, Vs2, yerr=s_Vs2, fmt='o', capsize=3, label='Low', color='tab:green', alpha=0.6)
plt.plot(xg, s_m*xg + b_m, color='tab:blue')
plt.plot(xg, s_h*xg + b_h, color='tab:orange')
plt.plot(xg, s_l*xg + b_l, color='tab:green')
plt.xlabel('1/λ (m$^{-1}$)')
plt.ylabel('Stopping energy (eV)')
plt.title('Photoelectric: V_s vs 1/λ (Mean, High, Low)')
plt.legend(); plt.show()
