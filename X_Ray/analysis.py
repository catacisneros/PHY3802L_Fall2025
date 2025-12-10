
import LT.box as B
import numpy as np

# Apparatus Uncertainty

sigma_theta_deg = 0.1
sigma_theta_rad = np.radians(sigma_theta_deg) #convert to radians

def sin_uncertainty(theta_deg):
    theta_rad = np.radians(theta_deg)
    return np.abs(np.cos(theta_rad)) * sigma_theta_rad

# Define the X-ray wavelengths for copper (in meters)
lambda_alpha = 1.542e-10  # Cu Kα
lambda_beta  = 1.392e-10  # Cu Kβ




# ---- GENERAL ANALYSIS ---

#import data
all_data = 'all_data.data'
all_data = B.get_file(all_data)

#define variables
theta_all_data = B.get_data(all_data, 'theta')
n_all_data = B.get_data(all_data, 'N')
uncertainty_all_data = np.sqrt(n_all_data)
theta_all_data = B.get_data(all_data, 'theta')


# print(theta_all_data)
# print()
# print()
# print(n_all_data)

#plot data
B.plot_exp(theta_all_data, n_all_data)
uncertainty_all_data = np.sqrt(n_all_data, uncertainty_all_data)



# --- FOR PEAK 1 ---

#import data
peak_1_data = 'peak_1.data'
peak_1_data = B.get_file(peak_1_data)

#define variables
theta_peak_1 = B.get_data(peak_1_data,'theta')
n_peak_1 = B.get_data(peak_1_data,'N')
uncertainty_peak_1 = np.sqrt(n_peak_1)

#plot data
B.plot_exp(theta_peak_1, n_peak_1, uncertainty_peak_1)

# --- FOR PEAK 2 ---

#import data
peak_2_data = 'peak_2.data'
peak_2_data = B.get_file(peak_2_data)

#define variables
theta_peak_2 = B.get_data(peak_2_data,'theta')
n_peak_2 = B.get_data(peak_2_data,'N')
uncertainty_peak_2 = np.sqrt(n_peak_2)

#plot data
B.plot_exp(theta_peak_2, n_peak_2, uncertainty_peak_2)


# --- FOR PEAK 3 ---

#import data
peak_3_data = 'peak_3.data'
peak_3_data = B.get_file(peak_3_data)

#define variables
theta_peak_3 = B.get_data(peak_3_data,'theta')
n_peak_3 = B.get_data(peak_3_data,'N')
n_peak_3 = n_peak_3/3
uncertainty_peak_3 = np.sqrt(n_peak_3)

#plot data
B.plot_exp(theta_peak_3, n_peak_3, uncertainty_peak_3)


# --- plot axis ----
B.pl.xlabel('2θ (deg)')
B.pl.ylabel('counts (N)')
B.pl.title('X-ray Intensity vs 2θ (Combined Data)')





# --- FOR PEAKS ONLY --- 

#peak 1
order_peak_1 = 1

ka_mask_1 = (theta_peak_1 >= 28.0) & (theta_peak_1 <= 30.0)
kb_mask_1 = (theta_peak_1 >= 31.0) & (theta_peak_1 <= 33.0)

theta_Ka_1 = theta_peak_1[ka_mask_1]
N_Ka_1 = n_peak_1[ka_mask_1]
theta_Kb_1 = theta_peak_1[kb_mask_1]
N_Kb_1 = n_peak_1[kb_mask_1]

sin_Ka_1 = np.sin(np.radians(theta_Ka_1/ 2))
sin_Kb_1 = np.sin(np.radians(theta_Kb_1/ 2))

n_Ka_1 = np.arange(1, len(sin_Ka_1) + 1)
n_Kb_1 = np.arange(1, len(sin_Kb_1) + 1)


#plot_Ka_1 = B.plot_exp(n_Ka_1, sin_Ka_1)
#plot_Kb_1 = B.plot_exp(n_Ka_1, sin_Kb_1)



#peak 3
order_peak_2 = 2

ka_mask_2 = (theta_peak_2 >= 59.0) & (theta_peak_2 <= 61.0)
kb_mask_2 = (theta_peak_2 >= 66.0) & (theta_peak_2 <= 68.0)

theta_Ka_2 = theta_peak_2[ka_mask_2]
N_Ka_2 = n_peak_2[ka_mask_2]
theta_Kb_2 = theta_peak_2[kb_mask_2]
N_Kb_2 = n_peak_2[kb_mask_2]

sin_Ka_2 = np.sin(np.radians(theta_Ka_2/ 2))
sin_Kb_2 = np.sin(np.radians(theta_Kb_2/ 2))

n_Ka_2 = np.arange(1, len(sin_Ka_2) + 1)
n_Kb_2 = np.arange(1, len(sin_Kb_2) + 1)


#plot_Ka_2 = B.plot_exp(n_Ka_2, sin_Ka_2)
#plot_Kb_2 = B.plot_exp(n_Ka_2, sin_Kb_2)


#peak 3
order_peak_3 = 3

ka_mask_3 = (theta_peak_3 >= 95.0) & (theta_peak_3 <= 97.0)
kb_mask_3 = (theta_peak_3 >= 108.0) & (theta_peak_3 <= 113.0)

theta_Ka_3 = theta_peak_3[ka_mask_3]
N_Ka_3 = n_peak_3[ka_mask_3]
theta_Kb_3 = theta_peak_3[kb_mask_3]
N_Kb_3 = n_peak_3[kb_mask_3]

sin_Ka_3 = np.sin(np.radians(theta_Ka_3/ 2))
sin_Kb_3 = np.sin(np.radians(theta_Kb_3/ 2))

n_Ka_3 = np.arange(1, len(sin_Ka_3) + 1)
n_Kb_3 = np.arange(1, len(sin_Kb_3) + 1)


#plot_Ka_3 = B.plot_exp(n_Ka_3, sin_Ka_3)
#plot_Kb_3 = B.plot_exp(n_Ka_3, sin_Kb_3)






# --- Linear fits for Kα and Kβ ---

# True diffraction orders
n_true = np.array([1, 2, 3])

# Take representative sin(θ) values (averages from each peak)
sin_Ka_all = np.array([np.mean(sin_Ka_1), np.mean(sin_Ka_2), np.mean(sin_Ka_3)])
sin_Kb_all = np.array([np.mean(sin_Kb_1), np.mean(sin_Kb_2), np.mean(sin_Kb_3)])

# uncertainty
sigma_sin_Ka_all = np.array([np.mean(sin_uncertainty(theta_Ka_1 / 2)),
                             np.mean(sin_uncertainty(theta_Ka_2 / 2)),
                             np.mean(sin_uncertainty(theta_Ka_3 / 2))])


sigma_sin_Kb_all = np.array([np.mean(sin_uncertainty(theta_Kb_1) / 2),
                             np.mean(sin_uncertainty(theta_Kb_2 / 2)),
                             np.mean(sin_uncertainty(theta_Kb_3) / 2)])

# --- Fit line for Kα ---
fit_Ka = np.polyfit(n_true, sin_Ka_all, 1)  # [slope, intercept]
fit_line_Ka = np.poly1d(fit_Ka)

# --- Fit line for Kβ ---
fit_Kb = np.polyfit(n_true, sin_Kb_all, 1)
fit_line_Kb = np.poly1d(fit_Kb)



# --- Plot Kα ---
B.pl.figure()
B.pl.errorbar(n_true, sin_Ka_all, yerr=sigma_sin_Ka_all, fmt='o', label='Kα data')

B.pl.plot(n_true, fit_line_Ka(n_true), color='orange', label='Fit')
B.pl.title("Bragg’s Law: sin(θ) vs n for Kα (Linear Fit)")
B.pl.xlabel("Order n")
B.pl.ylabel("sin(θ)")
B.pl.grid(True)
B.pl.legend()



# --- Plot Kβ ---
# --- Plot Kβ cleanly ---
B.pl.figure()
B.pl.errorbar(n_true, sin_Kb_all, yerr=sigma_sin_Kb_all, fmt='o', color='green', label='Kβ data')

B.pl.plot(n_true, fit_line_Kb(n_true), color='red', label='Fit Kβ')
B.pl.title("Bragg’s Law: sin(θ) vs n for Kβ (Linear Fit)")
B.pl.xlabel("Order n")
B.pl.ylabel("sin(θ)")
B.pl.grid(True)
B.pl.legend()

# --- Combined sin(θ) vs n plot for Kα and Kβ ---

B.pl.figure()
B.pl.errorbar(n_true, sin_Ka_all, yerr=sigma_sin_Ka_all, fmt='o', color='orange', label='Kα data')
B.pl.plot(n_true, fit_line_Ka(n_true), color='darkorange', linestyle='--', label='Kα fit')

B.pl.errorbar(n_true, sin_Kb_all, yerr=sigma_sin_Kb_all, fmt='s', color='green', label='Kβ data')
B.pl.plot(n_true, fit_line_Kb(n_true), color='darkgreen', linestyle='--', label='Kβ fit')

B.pl.title("X-ray Diffraction: sin(θ) vs Order n for Kα and Kβ")
B.pl.xlabel("Order n")
B.pl.ylabel("sin(θ)")
B.pl.grid(True)
B.pl.legend()


# Print fit results
print("Kα fit: sinθ = {:.4f}n + {:.4f}".format(fit_Ka[0], fit_Ka[1]))
print("Kβ fit: sinθ = {:.4f}n + {:.4f}".format(fit_Kb[0], fit_Kb[1]))



# Optional: extract covariance from np.polyfit using full=True or np.polyfit with cov=True (Python 3.10+)
fit_Ka, cov_Ka = np.polyfit(n_true, sin_Ka_all, 1, cov=True)
fit_Kb, cov_Kb = np.polyfit(n_true, sin_Kb_all, 1, cov=True)

slope_Ka = fit_Ka[0]
slope_Kb = fit_Kb[0]
slope_err_Ka = np.sqrt(cov_Ka[0][0])
slope_err_Kb = np.sqrt(cov_Kb[0][0])

d_Ka = lambda_alpha / (2 * slope_Ka)


# Calculate d and its uncertainty
d_Ka = lambda_alpha / (2 * slope_Ka)
d_err_Ka = lambda_alpha * slope_err_Ka / (2 * slope_Ka**2)

d_Kb = lambda_beta / (2 * slope_Kb)
d_err_Kb = lambda_beta * slope_err_Kb / (2 * slope_Kb**2)

print(f"Kα: d = {d_Ka*1e12:.3f} ± {d_err_Ka*1e12:.3f} pm")
print(f"Kβ: d = {d_Kb*1e12:.3f} ± {d_err_Kb*1e12:.3f} pm")





# Close any open figures before starting new fits
# B.pl.close('all')

# ============================================
# CLEAN ZOOMED PEAK FITS (Gaussian curve + data points)
# ============================================


files_and_ranges = [
    ('peak_1.data', [(28.0, 30.0), (31.0, 33.0)]),    # 1st order Ka, Kb
    ('peak_2.data', [(59.0, 61.0), (66.0, 68.0)]),    # 2nd order Ka, Kb
    ('peak_3.data', [(95.0, 97.0), (108.0, 113.0)])   # 3rd order Ka, Kb
]

peak_num = 1

for file_name, ranges in files_and_ranges:
    data = B.get_file(file_name)
    theta = B.get_data(data, 'theta')
    N = B.get_data(data, 'N')
    unc = np.sqrt(N)

    # Adjust counts for peak_3
    if file_name == 'peak_3.data':
        N = N / 3
        unc = np.sqrt(N)

    for low, high in ranges:
        mask = (theta >= low) & (theta <= high)
        x = theta[mask]
        y = N[mask]
        dy = unc[mask]

        if len(x) == 0:
            continue

        # --- Initial guess for Gaussian parameters ---
        A0 = np.max(y)
        mu0 = x[np.argmax(y)]
        sigma0 = (x[-1] - x[0]) / 6  # rough width estimate

        # --- Gaussian fit using numpy polyfit style (curve_fit if needed) ---
        def gaussian(x, A, mu, sigma):
            return A * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))

        from scipy.optimize import curve_fit
        popt, pcov = curve_fit(gaussian, x, y, p0=[A0, mu0, sigma0])
        A, mu, sigma = popt
        perr = np.sqrt(np.diag(pcov))
        mu_err = perr[1]

        # --- Plot data + fitted Gaussian ---
        B.pl.figure()
        B.pl.errorbar(x, y, yerr=dy, fmt='o', color='orange', label='Data')
        x_fit = np.linspace(np.min(x), np.max(x), 300)
        y_fit = gaussian(x_fit, *popt)
        B.pl.plot(x_fit, y_fit, color='blue', label='Gaussian Fit')

        B.pl.title(f"Peak# {peak_num}: Gaussian Fit at 2θ ≈ {mu:.3f}°")
        B.pl.xlabel("2θ (deg)")
        B.pl.ylabel("Counts (N)")
        B.pl.legend([f"Fit: μ = {mu:.3f} ± {mu_err:.3f}°", "Data"])
        B.pl.grid(True)

        print(f"Peak {peak_num}: μ = {mu:.3f} ± {mu_err:.3f}°, σ = {sigma:.3f}")
        peak_num += 1



#new  plot with braggs law

# --- Build the combined Bragg plot: sin(theta) vs n*lambda ---

import numpy as np

# 2θ peak centers from your Gaussian fits (deg)
mu_deg = np.array([28.915, 31.962, 59.719, 66.930, 95.794, 110.518])

# Identify line type per peak (Kα for 1,3,5; Kβ for 2,4,6)
is_alpha = np.array([True, False, True, False, True, False])

# Diffraction order for each peak
n_all = np.array([1, 1, 2, 2, 3, 3], dtype=float)

# Wavelengths (meters)
lambda_alpha = 1.542e-10
lambda_beta  = 1.392e-10
lambda_all_m = np.where(is_alpha, lambda_alpha, lambda_beta)

# X values: n * λ (convert to pm for the axis label)
x_pm = n_all * lambda_all_m * 1e12  # pm

# Y values: sin(theta) with theta = (2θ)/2
theta_deg = mu_deg / 2.0
y = np.sin(np.radians(theta_deg))

# Fit: sin(theta) = m * (n*λ) + b, so m = 1/(2d)
fit, cov = np.polyfit(x_pm, y, 1, cov=True)
m, b = fit
dm = np.sqrt(cov[0, 0])

# Compute d from slope. Careful with units: x was in pm, so m has units 1/pm.
# d = 1 / (2m)
d_pm  = 1.0 / (2.0 * m)
dd_pm = dm / (2.0 * m**2)

print(f"Combined fit: sinθ = ({m:.6f})·(nλ[pm]) + {b:.6f}")
print(f"d = {d_pm:.3f} ± {dd_pm:.3f} pm")

# Plot
import matplotlib.pyplot as plt
plt.figure(figsize=(6,4))
plt.scatter(x_pm[is_alpha], y[is_alpha], marker='o', label='Kα data')
plt.scatter(x_pm[~is_alpha], y[~is_alpha], marker='s', label='Kβ data')

x_line = np.linspace(0, x_pm.max()*1.05, 200)
plt.plot(x_line, m*x_line + b, linestyle='--', label='Linear fit')

plt.title("Bragg plot: sin(θ) vs n·λ (single fit)")
plt.xlabel("n · λ (pm)")
plt.ylabel("sin(θ)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
