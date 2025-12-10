import LT.box as B
import numpy as np

# ---------- constants you must set ----------
lam = 670e-9          # laser wavelength in meters
L   = 1.23            # distance double slit to detector slit (m)  <- put your value

def sinc2(z):
    """Safe (sin z / z)^2 that works at z = 0."""
    out = np.ones_like(z)
    mask = np.abs(z) > 1e-8
    out[mask] = (np.sin(z[mask]) / z[mask])**2
    return out

# ---------- helper: load, normalize, and center one data file ----------
def load_data(fname):
    d = B.get_file(fname)

    # position of detector slit (mm) → meters
    x = d['S'] * 1e-3

    # voltage → intensity, subtract background and normalize to 1
    I = d['V'].astype(float)
    I = I - I.min()            # remove offset
    I = I / I.max()            # max = 1

    # center x so that the main maximum is near x = 0
    i_max = np.argmax(I)
    x0_center = x[i_max]
    x = x - x0_center

    # simple constant uncertainty
    dI = 0.05 * np.ones_like(I)

    return x, I, dI

# ---------- single slit fit (left slit) ----------
x_L, I_L, dI_L = load_data('left_slit.data.dat')

D_L  = B.Parameter(1.0e-4, 'D_L')    # slit width initial guess ~0.1 mm
x0_L = B.Parameter(0.0,    'x0_L')   # small shift from centering
I0_L = B.Parameter(1.0,    'I0_L')   # max intensity

def I_single_L(x):
    k   = 2.0 * np.pi / lam
    phi = 0.5 * k * D_L() * np.sin((x - x0_L()) / L)
    return I0_L() * sinc2(phi)

fit_L = B.genfit(I_single_L, [D_L, x0_L, I0_L],
                 x=x_L, y=I_L, y_err=dI_L)

B.pl.figure()
B.plot_exp(x_L, I_L, dI_L)
B.plot_line(fit_L.xpl, fit_L.ypl)
print("Left slit fit:")
print(D_L)      # value and error
print(x0_L)
print(I0_L)

# ---------- single slit fit (right slit) ----------
x_R, I_R, dI_R = load_data('right_slit.data.dat')

D_R  = B.Parameter(1.0e-4, 'D_R')    # similar width
x0_R = B.Parameter(0.0,    'x0_R')
I0_R = B.Parameter(1.0,    'I0_R')

def I_single_R(x):
    k   = 2.0 * np.pi / lam
    phi = 0.5 * k * D_R() * np.sin((x - x0_R()) / L)
    return I0_R() * sinc2(phi)

fit_R = B.genfit(I_single_R, [D_R, x0_R, I0_R],
                 x=x_R, y=I_R, y_err=dI_R)

B.pl.figure()
B.plot_exp(x_R, I_R, dI_R)
B.plot_line(fit_R.xpl, fit_R.ypl)
print("Right slit fit:")
print(D_R)
print(x0_R)
print(I0_R)

# ---------- double slit fit ----------
x_D, I_D, dI_D = load_data('two_slit.data.dat')

D   = B.Parameter(1.0e-4, 'D')       # common slit width
S   = B.Parameter(4.0e-4, 'S')       # slit separation initial guess
x0  = B.Parameter(0.0,    'x0')
I0  = B.Parameter(1.0,    'I0')

def I_double(x):
    k   = 2.0 * np.pi / lam
    phi = 0.5 * k * D() * np.sin((x - x0()) / L)
    psi = 0.5 * k * S() * np.sin((x - x0()) / L)
    envelope = sinc2(phi)
    fringes  = np.cos(psi)**2
    return I0() * envelope * fringes          # eq. (15.3) in safe form

fit_D = B.genfit(I_double, [D, S, x0, I0],
                 x=x_D, y=I_D, y_err=dI_D)

B.pl.figure()
B.plot_exp(x_D, I_D, dI_D)
B.plot_line(fit_D.xpl, fit_D.ypl)

print("Double slit fit:")
print(D)
print(S)
print(x0)
print(I0)
