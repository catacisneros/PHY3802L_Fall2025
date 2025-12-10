

import numpy as np
import LT.box as B

# --- Geometry ---
f2_m = 0.0252 #m
D1_m = 7.3533 #m
D2_m = 7.1374 #m

sigma_f2 = 0.0005 #m
sigma_D1 = 0.01 #m
sigma_D2 = 0.01 #m
sigma_pos = 0.00008  # m

# --- Data ---
freq = np.array([
    102, 204, 307, 407, 501, 602, 702, 800, 900, 1004,
   -102,-201,-303,-402,-502,-608,-702,-805,-903,-1007
])
pos = np.array([
    0.461, 0.441, 0.404, 0.360, 0.332, 0.299, 0.259, 0.221, 0.201, 0.188,
    0.536, 0.547, 0.600, 0.639, 0.669, 0.699, 0.730, 0.754, 0.792, 0.828
])

# --- Preprocessing ---
omega = 2 * np.pi * freq 
pos_m = pos * 1e-3 #to meters 
sigma_y = np.full_like(pos_m, sigma_pos * 1e-3)


# --- Fit: y = mω + b ---
lf = B.linefit(omega, pos_m, sigma_y)
m = lf.slope
b = lf.offset
sm = lf.sigma_s
sb = lf.sigma_o

# --- Compute c ---
geom = 4 * f2_m * (D1_m + D2_m)
c_meas = abs(geom / m)

dc_dm  = -c_meas / m
dc_df2 = 4 * (D1_m + D2_m) / m
dc_dD1 = 4 * f2_m / m
dc_dD2 = 4 * f2_m / m

sigma_c = np.sqrt((dc_dm*sm)**2 + (dc_df2*sigma_f2)**2 +
                  (dc_dD1*sigma_D1)**2 + (dc_dD2*sigma_D2)**2)



geom = 4 * f2_m * (D1_m + D2_m)
c_meas = abs(geom / m)

dc_dm  = -c_meas / m
dc_df2 = 4 * (D1_m + D2_m) / m
dc_dD1 = 4 * f2_m / m
dc_dD2 = 4 * f2_m / m

sigma_c = np.sqrt((dc_dm*sm)**2 + (dc_df2*sigma_f2)**2 +
                  (dc_dD1*sigma_D1)**2 + (dc_dD2*sigma_D2)**2)

# --- Results ---
c0 = 2.99792458e8
percent_err = abs((c_meas - c0)/c0)*100

print("=== Linear Fit: Δx vs ω ===")
print(f"slope (m)      = {m:.6e} ± {sm:.6e}")
print(f"intercept (b)  = {b:.6e} ± {sb:.6e}")
print()
print(f"Uncertainty of the total speed of light = {sigma_c}")
print("=== Result for c ===")
print(f"c (measured)   = {c_meas:.6e} ± {sigma_c:.6e} m/s")
print(f"c (accepted)   = {c0:.6e} m/s")
print(f"percent error  = {percent_err:.3f}%")

# --- Plot ---
B.plot_exp(omega, pos_m, sigma_y, label="Data")
B.plot_line(omega, m*omega + b, label="Fit")
# Add labels, title, and legend
B.pl.xlabel("Angular Velocity ω (rad/s)")
B.pl.ylabel("Image Deflection Δx (m)")
B.pl.title("Δx vs ω")
B.pl.legend(frameon=False, loc='best')
B.pl.grid(True)
B.pl.show()




