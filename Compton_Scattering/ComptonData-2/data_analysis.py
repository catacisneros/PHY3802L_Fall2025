import numpy as np
import LT.box as B

# calibration numbers
energies_cal = np.array([1.1732, 1.3325, 0.6617])
channels_cal = np.array([423,    478,    244])

a, b = np.polyfit(energies_cal, channels_cal, 1)

def energy_to_channel(E):
    return a * E + b


# compton formula
E0 = 0.6617   # MeV
me = 0.511     # MeV

def Ef(theta_deg):
    theta = np.deg2rad(theta_deg)
    return E0 / (1 + (E0/me) * (1 - np.cos(theta)))


# Angles we want to predict
angles = np.array([30, 40, 60, 80, 100, 120])

# Arrays to store results
energies_pred = []
channels_pred = []

# Loop to compute energies and channels
for ang in angles:
    E_scattered = Ef(ang)
    energies_pred.append(E_scattered)

    ch_pred = energy_to_channel(E_scattered)
    channels_pred.append(ch_pred)


# Convert to numpy arrays (optional)
energies_pred = np.array(energies_pred)
channels_pred = np.array(channels_pred)

# print results
print("Predicted energies (MeV):")
print(energies_pred)

print("\nPredicted channel numbers:")
print(channels_pred)
print(" ")

#plot 
B.plot_exp(energies_pred, channels_pred)
B.pl.xlabel('Energy [MeV]')
B.pl.ylabel('peaks [Counts]')
B.pl.title('Energy vs Peaks (Predicted)')

# Perform a linear fit on the graph
B.pl.figure()
fitcal = B.linefit(energies_pred, channels_pred)
fitcal.plot()
B.pl.xlim(0.5, 700)

# calibration 
hcal=B.get_spectrum('Callibration.Spe')
hcal.plot()

#select ranges
peaks = np.zeros(3)
ranges = [(477, 479), (422, 424), (243, 245)]
i = 0 
for (xmin, xmax) in ranges:
    hcal.fit(xmin,xmax)
    hcal.plot()
    #5. obtain the mean value and store it in the array peaks
    peaks[i] = hcal.mean.value
    i += 1


B.pl.figure()
B.plot_exp(peaks, energies_cal)
B.pl.ylabel('Energy [MeV]')
B.pl.xlabel('peaks [Counts]')
B.pl.title('Energy vs Peaks')
# Perform a linear fit on the graph
fitcal = B.linefit(peaks, energies_cal)
fitcal.plot()

# Create 6 member array 
degrees = np.array([30, 40, 60, 80, 100, 120])
# Convert to radians and store in thetas
thetas = np.deg2rad(degrees)

# create an empty array for efs
efs = np.zeros(6)

# create an empty array for efsErr
efsErr = np.zeros(6)

# Apply calibration in 30_deg_target
htarget=B.get_spectrum('30_degree_target.Spe', calibration = fitcal)

# Apply calibration in no target 30 deg
hempty=B.get_spectrum('30_degree_no_target.Spe', calibration = fitcal)

B.pl.figure()
htarget.plot()
hempty.plot()
B.pl.title('htarget and hempty 30 degrees')
B.pl.xlim(0, 0.1e6)


B.pl.figure()
hcompton = htarget - hempty
hcompton.rebin(4)
hcompton.plot()
B.pl.xlim(0, 2)

B.pl.ylabel('Energies (MeV)')
B.pl.xlabel('counts')
B.pl.title('htarget-hempty 30 degrees')

