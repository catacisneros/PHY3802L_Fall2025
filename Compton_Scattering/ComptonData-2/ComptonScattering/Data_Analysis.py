# - PRACTICE TEST CATALINA CISNEROS - 

#0. import numpy and LT.box
import numpy as np
import LT.box as B

#1. Create array with given values 
Es = np.array([0.6617, 1.1732, 1.3325])

#2. Create an empty array size 3
peaks = np.zeros(3)

#3. Read calibration file and save it in hcal
hcal=B.get_spectrum('Callibration.Spe')

#4. show in a graph, with the automatic gaussian fit 
hcal.plot()
B.pl.xlim(0,600)

#4. Select appropriate ranges
ranges = [(240, 248), (419, 427), (474, 482)]
i = 0 
#6. Loop to repeat steps 4 and 5 
for (xmin, xmax) in ranges:
    hcal.fit(xmin,xmax)
    hcal.plot()
    
    #5. obtain the mean value and store it in the array peaks
    peaks[i] = hcal.mean.value
    i += 1

# 7. Graph E vs peaks and label x and y axis
B.pl.figure()
B.plot_exp(peaks, Es)
B.pl.xlabel('Peaks [Channel]')
B.pl.ylabel('Energy [MeV]')
B.pl.title('Energy vs Peaks')

# 8. Perform a linear fit on the graph
fitcal = B.linefit(peaks, Es)
fitcal.plot()

# 9. Verify calibration
B.pl.figure()
hcalnew = B.get_spectrum('Callibration.Spe', calibration=fitcal)
hcalnew.plot()
# 10. set xaxis label to energies
B.pl.xlabel('Energies (MeV)')
B.pl.xlim(0, 1.50)

# 11. Create six member array 
degrees = np.array([30, 40, 60, 80, 100, 120])
# 12. Convert to radians and store in thetas
thetas = np.deg2rad(degrees)

# 13. create an empty array for efs
efs = np.zeros(6)

# 14. create an empty array for efsErr
efsErr = np.zeros(6)

#22 - Main loop for all angles
deg = [30, 40, 60, 80, 100, 120]
# Fitting ranges for Compton peaks at different angles
ranges = [
    (0.45, 0.70),  # 30 degrees
    (0.35, 0.60),  # 40 degrees
    (0.26, 0.42),  # 60 degrees
    (0.20, 0.34),  # 80 degrees
    (0.16, 0.28),  # 100 degrees
    (0.13, 0.24)   # 120 degrees
]

for j, d in enumerate(deg):
    htarget=B.get_spectrum(f"{d}_degree_target.Spe", calibration = fitcal)
    hempty=B.get_spectrum(f"{d}_degree_no_target.Spe", calibration = fitcal)
    B.pl.figure()
    htarget.plot()
    hempty.plot()
    B.pl.title(f'{d} degrees - Target and Empty')
    B.pl.xlabel('Energy (MeV)')
    B.pl.ylabel('Counts')
    B.pl.xlim(0, 0.8)
    
    B.pl.figure()
    hcompton = htarget - hempty
    hcompton.plot()
    B.pl.xlabel('Energies (MeV)')
    B.pl.ylabel('counts')
    B.pl.title(f'{d} degrees - htarget-hempty')
    
    # Select appropriate ranges on htarget
    xmin, xmax = ranges[j]
    hcompton.fit(xmin, xmax)
    hcompton.plot()
    B.pl.title(f'{d} degrees - Fitted Compton Peak')
    B.pl.xlim(0, 0.8)
    
    # Store peak position mean in efs
    efs[j] = np.array(hcompton.mean.value)
    # Store error in efsErr
    efsErr[j] = hcompton.mean.err

# 23. compute e0/efs
e0 = Es[0]
eratio = e0/efs
print("Energy ratios (E0/E_scattered):")
print(eratio)

# 24. graph eratio vs cos thetas
B.pl.figure()
cos_t = 1.0 - np.cos(thetas)
B.plot_exp(cos_t, eratio, efsErr/efs * eratio)  # Include error bars
B.pl.ylabel('E0/E_scattered')
B.pl.xlabel('1 - cos(θ)')
B.pl.title('Compton Scattering: Energy Ratio vs 1-cos(θ)')

# 25. linefit to above
fit2 = B.linefit(cos_t, eratio)
fit2.plot()

# 26-29. Get fit parameters
eslope = fit2.slope
dslope = fit2.sigma_s
yoff   = fit2.offset
dyoff  = fit2.sigma_o

# 30. value of electron mass in MeV 
me = e0 / eslope
dme = e0 * dslope / (eslope**2)

# 31. Print electron mass and uncertainty
print(f"Electron mass m_e = {me:.3f} ± {dme:.3f} MeV")

# 32. Display electron mass result on the graph
B.pl.text(0.05, 0.90, f"m_e = {me:.3f} ± {dme:.3f} MeV", 
          transform=B.pl.gca().transAxes, fontsize=10, 
          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# 33. Print and show offset result on the graph
print(f"Offset = {yoff:.3f} ± {dyoff:.3f} (expected ≈ 1.000)")
B.pl.text(0.05, 0.84, f"Offset = {yoff:.3f} ± {dyoff:.3f} (expected ≈ 1.000)", 
          transform=B.pl.gca().transAxes, fontsize=10,
          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# %%
B.pl.show()
