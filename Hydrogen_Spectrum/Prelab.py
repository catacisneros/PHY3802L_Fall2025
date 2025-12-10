import numpy as np
import LT.box as B

# --- helper functions ---
def degmin_to_deg(d, m):
    return d + m/60

def deg_to_rad(angle_deg):
    return np.deg2rad(angle_deg)

# --- experimental data ---
# first 4 = first order, next 5 = second order
d = np.array([125, 124, 120, 108, 97, 93, 86, 60, 61])
m = np.array([50, 11, 21, 11, 0, 5, 28, 15, 1])

# --- reference readings ---
theta_a_deg = degmin_to_deg(208, 20)   # optical axis
theta_0_deg = degmin_to_deg(183, 10)   # zero-order reflection

# --- compute total angles ---
theta_line_deg = degmin_to_deg(d, m)

# --- convert to geometry angles ---
theta_in_deg  = 0.5 * (theta_0_deg - theta_a_deg)
theta_out_deg = theta_line_deg - 0.5 * (theta_a_deg + theta_0_deg)

# --- convert to radians ---
theta_in_rad  = deg_to_rad(theta_in_deg)
theta_out_rad = deg_to_rad(theta_out_deg)

# --- grating spacing ---
d_spacing = 1/1200 * 1e-3   # meters

# --- diffraction orders ---
order = np.array([1, 1, 1, 1, 2, 2, 2, 2, 3])

# --- compute wavelengths ---
lam = (d_spacing / order) * (np.cos(theta_in_rad) - np.cos(theta_out_rad))  # meters

# --- uncertainty propagation ---
sigma_arcmin = 1.0
sigma_rad = np.deg2rad(sigma_arcmin / 60)
sigma_theta_in  = np.sqrt(sigma_rad**2 + sigma_rad**2) / 2
sigma_theta_out = np.sqrt(sigma_rad**2 + (sigma_rad**2 + sigma_rad**2)/4)
sigma_lam = (d_spacing / order) * np.sqrt(
    (np.sin(theta_in_rad) * sigma_theta_in)**2 +
    (np.sin(theta_out_rad) * sigma_theta_out)**2
)

# --- print wavelength results ---
print("\n=== Hydrogen Spectrum Wavelengths ===")
for i in range(len(lam)):
    print(f"Line {i+1}: θ_line = {theta_line_deg[i]:.3f}°,  m = {order[i]},  λ = {lam[i]*1e9:.2f} ± {sigma_lam[i]*1e9:.2f} nm")

# --- plot wavelengths ---
B.pl.figure()
B.pl.plot(np.arange(len(lam)), lam*1e9, 'bo', label='Measured wavelengths')
B.pl.errorbar(np.arange(len(lam)), lam*1e9, sigma_lam*1e9, fmt='none', ecolor='gray', capsize=3)
B.pl.xlabel('Spectral Line Number')
B.pl.ylabel('Wavelength (nm)')
B.pl.title('Hydrogen Spectrum – Measured Wavelengths')
B.pl.legend()
B.pl.grid(True)
B.pl.show()

# ======================================================
# --- Rydberg constant analysis ---
# ======================================================

# Balmer transitions: n2 = 3,4,5,6 → n1 = 2
lambda_first_order = lam[0:4]
n_values = np.array([3, 4, 5, 6])

inv_lambda = 1 / lambda_first_order
inv_n2 = 1 / (n_values**2)

# x = (1/4 - 1/n²), y = 1/λ
x = (1/4) - inv_n2
y = inv_lambda

rydberg_fit = B.linefit(x, y)
R_measured = abs(rydberg_fit.slope)
R_uncert = rydberg_fit.sigma_s
intercept = rydberg_fit.offset

# --- print final results ---
print("\n=== Rydberg Constant Calculation ===")
print(f"Measured Rydberg constant: R = {R_measured:.3e} ± {R_uncert:.3e} m⁻¹")
print(f"Theoretical Rydberg constant: R = 1.097e7 m⁻¹")
print(f"Percent error = {abs(R_measured - 1.097e7)/1.097e7*100:.2f}%")

# --- plot Rydberg line ---
x_fit = np.linspace(min(x), max(x), 100)
y_fit = rydberg_fit.slope * x_fit + rydberg_fit.offset

B.pl.figure()
B.pl.plot(x, y, 'bo', label='Measured data')
B.pl.plot(x_fit, y_fit, 'r-', label='Linear fit')
B.pl.xlabel('1/4 - 1/n²')
B.pl.ylabel('1/λ (1/m)')
B.pl.title('Hydrogen Spectrum: 1/λ vs 1/n²')
B.pl.legend()
B.pl.grid(True)
B.pl.show()


import matplotlib.pyplot as plt

# --- First Graph: λ vs θ (experimental values) ---
theta_plot = theta_line_deg
lambda_nm = lam * 1e9
sigma_lambda_nm = sigma_lam * 1e9

plt.figure(figsize=(7,5))
plt.errorbar(theta_plot, lambda_nm, yerr=sigma_lambda_nm, fmt='o', capsize=4, color='royalblue')
plt.title('Experimental Values of λ')
plt.xlabel('Angle in degrees')
plt.ylabel('λ (nm)')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# --- Second Graph: 1/λ vs 1/n² (Rydberg Analysis) ---
# For the Balmer series, n = 3,4,5,6 → n1 = 2
inv_lambda = 1 / lambda_first_order
inv_n2 = 1 / (n_values**2)

# Fit line using slope and intercept
x_fit = np.linspace(min(inv_n2), max(inv_n2), 100)
y_fit = rydberg_fit.slope * ((1/4) - x_fit) + rydberg_fit.offset

plt.figure(figsize=(7,5))
plt.errorbar(inv_n2, inv_lambda, fmt='o', capsize=4, color='mediumblue', label='Experimental points')
plt.plot(inv_n2, rydberg_fit.slope*((1/4)-inv_n2)+rydberg_fit.offset, 'r-', label='Best linear fit')
plt.title('Rydberg Analysis: 1/n² vs 1/λ')
plt.xlabel('1 / n²')
plt.ylabel('1 / λ (1/m)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

