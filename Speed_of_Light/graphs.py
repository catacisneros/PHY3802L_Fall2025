#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  7 20:48:58 2025

@author: catacisneros
"""

import numpy as np
import LT.box as B

# --- Geometry ---
f2_m = 0.0252
D1_m = 7.3533
D2_m = 7.1374

sigma_f2 = 0.0005
sigma_D1 = 0.01
sigma_D2 = 0.01
sigma_pos = 0.027 / np.sqrt(12)  # mm

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


# ___ CLOCKWISE ---

# --- Data ---

#--CW--
freq_CW = np.array([
    102, 204, 307, 407, 501, 602, 702, 800, 900, 1004,
])
pos_CW = np.array([
    0.461, 0.441, 0.404, 0.360, 0.332, 0.299, 0.259, 0.221, 0.201, 0.188,
])


# --- Preprocessing ---
omega_CW = 2 * np.pi * freq_CW # anf vel
pos_m_CW = pos_CW * 1e-3 #to meters 
sigma_y_CW = np.full_like(pos_m_CW, sigma_pos * 1e-3)


# --- Fit: y = mω + b ---
lf_CW = B.linefit(omega_CW, pos_m_CW, sigma_y_CW)
m_CW = lf_CW.slope
b_CW = lf_CW.offset
sm_CW = lf_CW.sigma_s
sb_CW = lf_CW.sigma_o

# --- Compute c ---
geom = 4 * f2_m * (D1_m + D2_m)
c_meas_CW = abs(geom / m_CW)

dc_dm_CW  = -c_meas_CW / m_CW
dc_df2_CW = 4 * (D1_m + D2_m) / m_CW
dc_dD1_CW = 4 * f2_m / m_CW
dc_dD2_CW = 4 * f2_m / m_CW

sigma_c_CW = np.sqrt((dc_dm_CW*sm_CW)**2 + (dc_df2_CW*sigma_f2)**2 +
                  (dc_dD1_CW*sigma_D1)**2 + (dc_dD2_CW*sigma_D2)**2)

# --- Results ---
c0 = 2.99992458e8
percent_err = abs((c_meas_CW - c0)/c0)*100

print("=== CW Linear Fit: Δx vs ω ===")
print(f"slope (m)      = {m_CW:.6e} ± {sm_CW:.6e}")
print(f"intercept (b)  = {b_CW:.6e} ± {sb_CW:.6e}")
print(f"Uncertainty of the total speed of light = {sigma_c_CW} ")
print()
print("=== Result for c ===")
print(f"c (measured)   = {c_meas_CW:.6e} ± {sigma_c:.6e} m/s")
print(f"c (accepted)   = {c0:.6e} m/s")
print(f"percent error  = {percent_err:.3f}%")

# --- Plot ---
B.plot_exp(omega_CW, pos_m_CW, sigma_y_CW, label="Data")
B.plot_line(omega_CW, m_CW*omega_CW + b_CW, label="Fit")
# Add labels, title, and legend
B.pl.xlabel("Angular Velocity ω (rad/s)")
B.pl.ylabel("Image Deflection Δx (m)")
B.pl.title("CW Δx vs ω")
B.pl.legend(frameon=False, loc='best')
B.pl.grid(True)
B.pl.show()







#--CCW--
freq_CCW = np.array([
   102, 201, 303, 402, 502, 608, 702, 805, 903, 1007
])
pos_CCW = np.array([
    0.536, 0.547, 0.600, 0.639, 0.669, 0.699, 0.730, 0.754, 0.792, 0.828
])

# --- Preprocessing ---
omega_CCW = 2 * np.pi * freq_CCW # anf vel
pos_m_CCW = pos_CCW * 1e-3 #to meters 
sigma_y_CCW = np.full_like(pos_m_CCW, sigma_pos * 1e-3)


# --- Fit: y = mω + b ---
lf_CCW = B.linefit(omega_CCW, pos_m_CCW, sigma_y_CCW)
m_CCW = lf_CCW.slope
b_CCW = lf_CCW.offset
sm_CCW = lf_CCW.sigma_s
sb_CCW = lf_CCW.sigma_o

# --- Compute c ---
geom = 4 * f2_m * (D1_m + D2_m)
c_meas_CCW = abs(geom / m_CCW)

dc_dm_CW  = -c_meas_CW / m_CW
dc_df2_CW = 4 * (D1_m + D2_m) / m_CW
dc_dD1_CW = 4 * f2_m / m_CW
dc_dD2_CW = 4 * f2_m / m_CW

sigma_c_CW = np.sqrt((dc_dm_CW*sm_CW)**2 + (dc_df2_CW*sigma_f2)**2 +
                  (dc_dD1_CW*sigma_D1)**2 + (dc_dD2_CW*sigma_D2)**2)

# --- Results ---
c0 = 2.99992458e8
percent_err = abs((c_meas_CW - c0)/c0)*100

print("=== CW Linear Fit: Δx vs ω ===")
print(f"slope (m)      = {m_CW:.6e} ± {sm_CW:.6e}")
print(f"intercept (b)  = {b_CW:.6e} ± {sb_CW:.6e}")
print(f"Uncertainty of the total speed of light = {sigma_c_CW}")
print()
print("=== Result for c ===")
print(f"c (measured)   = {c_meas_CW:.6e} ± {sigma_c_CW:.6e} m/s")
print(f"c (accepted)   = {c0:.6e} m/s")
print(f"percent error  = {percent_err:.3f}%")

# --- Plot ---
B.plot_exp(omega_CW, pos_m_CW, sigma_y_CW, label="Data")
B.plot_line(omega_CW, m_CW*omega_CW + b_CW, label="Fit")
# Add labels, title, and legend
B.pl.xlabel("Angular Velocity ω (rad/s)")
B.pl.ylabel("Image Deflection Δx (m)")
B.pl.title("CW Δx vs ω")
B.pl.legend(frameon=False, loc='best')
B.pl.grid(True)
B.pl.show()

