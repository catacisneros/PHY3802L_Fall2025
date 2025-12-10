import LT.box as B
import numpy as np

data_files = ["emV200.data", "emV250.data", "emV300.data", "emV400.data"]

cm2m = 0.01

# coil/apparatus (same as before)
N   = 132.0
dN  = 0.0
Dc  = 30.5*cm2m
dDc = 1.0*cm2m
R   = Dc/2.0
dR  = dDc/2.0
mu0 = 4.0*np.pi*1.0e-7

fac = (4.0/5.0)**1.5

# simple weighted linear fit: y = m*x + b
def wlinfit(x, y, sig_y):
    w = 1.0 / (sig_y**2)
    Sw  = np.sum(w)
    Sx  = np.sum(w*x)
    Sy  = np.sum(w*y)
    Sxx = np.sum(w*x*x)
    Sxy = np.sum(w*x*y)
    Delta = Sw*Sxx - Sx*Sx
    m = (Sw*Sxy - Sx*Sy) / Delta
    b = (Sxx*Sy - Sx*Sxy) / Delta
    dm = np.sqrt(Sw/Delta)
    db = np.sqrt(Sxx/Delta)
    return m, b, dm, db

for file_name in data_files:
    f = B.get_file(file_name)

    I  = B.get_data(f, "I")
    D  = B.get_data(f, "D")
    dD = B.get_data(f, "dD")

    V  = f.par.get_value("V")
    dV = f.par.get_value("dV")
    dI = f.par.get_value("dI")

    # geometry -> radius
    r  = (D*cm2m)/2.0
    dr = (dD*cm2m)/2.0

    # B-field and its uncertainty (same formulas you had)
    Bf  = mu0*N*I*fac/R
    dBf = fac*mu0*np.sqrt( (I/R)**2*(dN**2)
                         + ((I*N)/(R**2))**2*(dR**2)
                         + ((N/R)**2)*(dI**2) )

    # point-by-point e/m and uncertainty (same equations)
    Rem  = 2.0*V/(Bf**2 * r**2)

    termV = (2.0/(Bf**2 * r**2) * dV)**2
    termB = (4.0*V/(Bf**3 * r**2) * dBf)**2
    termr = (4.0*V/(Bf**2 * r**3) * dr )**2
    dRem  = np.sqrt(termV + termB + termr)

    # ========= Plot: e/m vs I with a weighted LINEAR fit overlaid =========
    B.pl.figure(num=f"em_vs_I_V{int(V)}")

    # data with vertical error bars
    B.plot_exp(I, Rem, dRem)

    # weighted linear fit on the SAME axes
    m, b, dm, db = wlinfit(I, Rem, dRem)
    Igrid = np.linspace(np.min(I), np.max(I), 400)
    B.pl.plot(Igrid, m*Igrid + b)
              #label=f"linear fit: y = ({m:.3e}±{dm:.3e}) I + ({b:.3e}±{db:.3e})")
    

    # labels and title
    B.pl.xlabel("Current I [A]")
    B.pl.ylabel("EM Ratio [C/kg]")
    B.pl.title(f"EM Ratio vs Current at V = {V:.0f} V")
    B.pl.legend()
    B.pl.grid(True, alpha=0.3)
    B.pl.tight_layout()
    B.pl.savefig(f"em_vs_I_linefit_V{int(V)}.png", dpi=200)
    B.pl.show()
