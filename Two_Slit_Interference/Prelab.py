import numpy as np
import LT.box as B

#5.2.B 

# Parameters
lam = 500e-9             # wavelength (m)
D = 0.01e-3              # slit width (m)
L = 0.5                  # distance to screen (m)

# Screen coordinates
x = np.linspace(-0.01, 0.01, 2000)
theta = np.arctan(x / L)

# Fraunhofer single slit
phi = (2 * np.pi / lam) * D * np.sin(theta)
I_single = (np.sin(phi/2) / (phi/2))**2

# Plot 
B.pl.figure()
B.plot_line(x, I_single)
B.pl.xlabel('x (m)')
B.pl.ylabel('Intensity')
B.pl.title('Single Slit Diffraction')

#5.2.C

# Parameters
S = 4.512            # slit separation (m)

x = np.linspace(-0.01, 0.01, 2000)
theta = np.arctan(x / L)

phi = (2 * np.pi / lam) * D * np.sin(theta)
psi = (2 * np.pi / lam) * S * np.sin(theta)

I_single = (np.sin(phi/2) / (phi/2))**2
I_double = I_single * (np.cos(psi/2))**2

B.pl.figure()
B.plot_line(x, I_double)
B.pl.xlabel('x (m)')
B.pl.ylabel('Intensity')
B.pl.title('Single Slit Diffraction')

#5.2.D

B.pl.figure()
B.plot_line(x, I_single, color='blue', label='Single Slit')
B.plot_line(x, I_double, color='red', label='Double Slit')

B.pl.xlabel('x (m)')
B.pl.ylabel('Intensity')
B.pl.title('Single vs Double Slit Pattern')
B.pl.legend()

