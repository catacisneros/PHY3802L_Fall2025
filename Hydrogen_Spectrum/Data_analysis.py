import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Helper functions
# ============================================================
def degmin_to_deg(d, m):
    """Convert degrees + minutes to decimal degrees."""
    return d + m/60

def degmin_to_rad(d, m):
    return np.deg2rad(d + m/60)

def arcmin_to_rad(a):
    return np.deg2rad(a/60)

# ============================================================
# Full dataset (deg, min, order, color, n_i)
# ============================================================
data = [
    (125, 50, 1, "dark violet", 6),
    (124, 11, 1, "violet", 5),
    (120, 21, 1, "blue green", 4),
    (108, 11, 1, "red", 3),

    (97, 0, 2, "dark violet", 6),
    (93, 5, 2, "violet", 5),
    (86, 28, 2, "blue green", 4),
    (60, 15, 2, "red", 3),

    (61, 1, 3, "violet", 5),
]

deg = np.array([row[0] for row in data], float)
minute = np.array([row[1] for row in data], float)
order = np.array([row[2] for row in data], float)
color = [row[3] for row in data]
n_i = np.array([row[4] for row in data], float)

# For plotting λ vs θ
theta_meas_deg = degmin_to_deg(deg, minute)

# ============================================================
# Instrument configuration
# ============================================================
theta_a = degmin_to_deg(208, 20)
theta_0 = degmin_to_deg(170, 1)

theta_a_rad = np.deg2rad(theta_a)
theta_0_rad = np.deg2rad(theta_0)
theta_meas_rad = degmin_to_rad(deg, minute)

# Geometry angles
theta_in = 0.5*(theta_0_rad - theta_a_rad)
theta_out = theta_meas_rad - 0.5*(theta_0_rad + theta_a_rad)

# ============================================================
# Compute wavelengths
# ============================================================
D = (1/1200)*1e-3   # grating spacing

lam = D*(np.cos(theta_in) - np.cos(theta_out)) / order
lam_nm = lam*1e9

# ============================================================
# Uncertainty propagation
# ============================================================
sigma_theta = arcmin_to_rad(30)     # 30 arcmin reading uncertainty
sigma_in = np.sqrt(2)*sigma_theta/2
sigma_out = np.sqrt(sigma_theta**2 + 0.5*sigma_theta**2)

sigma_lam = np.sqrt(
    (D*np.sin(theta_in)/order * sigma_in)**2 +
    (D*np.sin(theta_out)/order * sigma_out)**2
)
sigma_lam_nm = sigma_lam*1e9

# ============================================================
# Print all wavelengths
# ============================================================
print("\n=== All wavelengths ===")
for c, m, L, s in zip(color, order, lam_nm, sigma_lam_nm):
    print(f"{c:12s}  m={int(m)}  λ={L:.2f} ± {s:.2f} nm")

# ============================================================
# Weighted mean per color
# ============================================================
colors_unique = ["dark violet", "violet", "blue green", "red"]
mean_lam = []
mean_sig = []
mean_n = []

for c in colors_unique:
    mask = np.array(color) == c
    Lvals = lam[mask]
    Svals = sigma_lam[mask]

    w = 1/Svals**2
    Lmean = np.sum(w * Lvals) / np.sum(w)
    Smean = 1/np.sqrt(np.sum(w))

    mean_lam.append(Lmean)
    mean_sig.append(Smean)
    mean_n.append(n_i[mask][0])

mean_lam = np.array(mean_lam)
mean_sig = np.array(mean_sig)
mean_n = np.array(mean_n)

print("\n=== Weighted means per color ===")
for c, L, s in zip(colors_unique, mean_lam*1e9, mean_sig*1e9):
    print(f"{c:12s} {L:.2f} ± {s:.2f} nm")

# ============================================================
# Rydberg linear fit: y = a x + b
# ============================================================
x = 1/(mean_n**2)
y = 1/mean_lam
dy = mean_sig / mean_lam**2

# Weighted fit
w = 1/dy**2
S  = np.sum(w)
Sx = np.sum(w*x)
Sy = np.sum(w*y)
Sxx = np.sum(w*x*x)
Sxy = np.sum(w*x*y)

Delta = S*Sxx - Sx*Sx
a = (S*Sxy - Sx*Sy)/Delta
b = (Sxx*Sy - Sx*Sxy)/Delta
a_err = np.sqrt(S / Delta)
b_err = np.sqrt(Sxx / Delta)

R_slope = -a
R_slope_err = a_err
R_off = b*(2**2)
R_off_err = b_err*(2**2)

print("\n=== Rydberg constant ===")
print("slope m      =", a, "+/-", a_err)
print("offset b     =", b, "+/-", b_err)
print("R (slope)    =", R_slope, "+/-", R_slope_err)
print("R (offset)   =", R_off, "+/-", R_off_err)

# ============================================================
# PLOT 1: λ vs θ (report ready)
# ============================================================
plt.figure(figsize=(7,5), dpi=200)
plt.errorbar(theta_meas_deg, lam_nm, yerr=sigma_lam_nm,
             fmt='o', capsize=4, color='black', ecolor='gray')
plt.xlabel("Measured Angle (degrees)", fontsize=14)
plt.ylabel("Wavelength (nm)", fontsize=14)
plt.title("Hydrogen Spectrum Wavelengths", fontsize=16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(False)
plt.tight_layout()
plt.show()

# ============================================================
# PLOT 2: Rydberg fit (report ready)
# ============================================================
xx = np.linspace(min(x), max(x), 200)
yy = a*xx + b

plt.figure(figsize=(7,5), dpi=200)
plt.errorbar(x, y, yerr=dy, fmt='o', capsize=4,
             color='black', ecolor='gray', markersize=8, label="Data")
plt.plot(xx, yy, 'r-', linewidth=2, label=" Rydberg Linear Fit")
plt.xlabel("1 / n²", fontsize=14)
plt.ylabel("1 / λ (m⁻¹)", fontsize=14)
plt.title("Rydberg Linear Fit", fontsize=16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=12)
plt.grid(False)
plt.tight_layout()
plt.show()

