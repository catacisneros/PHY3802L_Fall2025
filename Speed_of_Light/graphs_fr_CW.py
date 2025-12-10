
import numpy as np
import LT.box as B

f2 = 0.252 #m 
d1 = 7.353 #m
sigma_d1 = 0.1 #m
d2 = 7.137 #m
sigma_d2 = 0.1 #m

sigma_pos = 0.00008  # m (microscope reading uncertainty)


# --- Data ---
freq = np.array([
    102, 204, 307, 407, 501, 602, 702, 800, 900, 1004,
])
pos = np.array([
    0.461, 0.441, 0.404, 0.360, 0.332, 0.299, 0.259, 0.221, 0.201, 0.188,
])


omega = 2 * np.pi * freq
#print(omega)

pos = pos * 1e-3 # to meters
#print(pos)

#sigma_y = np.full_like(sigma_pos, pos)

#--- fit ---
fit_value = B.linefit(pos, omega)
slope = fit_value.slope
print()
offset = fit_value.offset
print()
error_slope = fit_value.sigma_s
error_offset = fit_value.sigma_o

#calculate c 

c = abs (4 * f2 * (d1 + d2) * slope)
print(c)
print()



#-error-

sigma_dsum = np.sqrt(sigma_d1**2 + sigma_d2**2) #swrt of the sum of uncertainties

sigma_c = c * np.sqrt(
    (error_slope / slope)**2 +
    (sigma_dsum / (d1 + d2))**2
)

print (sigma_c)

#---results---

print("\n----------------------------")
print("Speed of light result (Clockwise):")
print("c = {:.3e} ± {:.3e} m/s".format(c, sigma_c))
percent_error = abs((c - 2.998e8) / 2.998e8) * 100
print("Percent error vs accepted value: {:.2f}%".format(percent_error))
print("----------------------------")


# ---plot---

B.plot_exp(pos, omega, np.full_like(pos, sigma_pos), label="data")
# # Add labels, title, and legend
B.pl.xlabel("Image Deflection Δx (m)")
B.pl.ylabel("Angular Velocity ω (rad/s)")
B.pl.title("Δx vs ω - Clockwise")
B.pl.show()

