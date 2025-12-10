#!/usr/bin/env python3
"""
Young's Double Slit Interference Analysis - COMPLETE VERSION
With all graphs and visualizations
Florida International University - Intermediate Physics Lab
"""

import LT.box as B
import numpy as np

# ============================================================================
# EXPERIMENTAL PARAMETERS
# ============================================================================
lam = 670e-9  # meters (670 nm)
L = 0.500  # meters (500 mm from Fig 15.5)
zero_offset = 8.5  # mV - adjust based on your dark current measurement

print("="*60)
print("YOUNG'S DOUBLE SLIT INTERFERENCE ANALYSIS")
print("="*60)
print(f"\nExperimental parameters:")
print(f"Wavelength: {lam*1e9:.0f} nm")
print(f"Distance L: {L*1000:.0f} mm")
print(f"Zero offset: {zero_offset:.1f} mV")

# ============================================================================
# LOAD DATA FILES
# ============================================================================
print("\nLoading experimental data files...")
print("Make sure the .dat files are in the same directory as this script!\n")

try:
    left_slit = B.get_file('left_slit.data.dat')
    S_left = left_slit['S']
    V_left = left_slit['V']
    print(f"✓ Left slit: {len(S_left)} data points loaded")
    
    right_slit = B.get_file('right_slit.data.dat')
    S_right = right_slit['S']
    V_right = right_slit['V']
    print(f"✓ Right slit: {len(S_right)} data points loaded")
    
    double_slit = B.get_file('two_slit.data.dat')
    S_double = double_slit['S']
    V_double = double_slit['V']
    print(f"✓ Double slit: {len(S_double)} data points loaded")
except Exception as e:
    print(f"ERROR: Could not load data files - {e}")
    print("Make sure all three .dat files are in the same directory as this script")
    import sys
    sys.exit(1)

# ============================================================================
# DATA PREPROCESSING
# ============================================================================
print("\nPreprocessing data...")

# Convert positions to angles (theta = x/L for small angles)
theta_left = (S_left * 1e-3) / L  # radians
theta_right = (S_right * 1e-3) / L  # radians
theta_double = (S_double * 1e-3) / L  # radians

# Normalize data
def normalize_data(V, zero_offset=8.5):
    """Normalize voltage data by subtracting zero offset and scaling to max=1"""
    V_corrected = V - zero_offset
    V_corrected[V_corrected < 0] = 0
    if V_corrected.max() > 0:
        V_normalized = V_corrected / V_corrected.max()
    else:
        V_normalized = V_corrected
    return V_normalized

I_left = normalize_data(V_left, zero_offset)
I_right = normalize_data(V_right, zero_offset)
I_double = normalize_data(V_double, zero_offset)

# Estimate errors (1% measurement uncertainty)
dI_left = 0.01 * np.ones_like(I_left)
dI_right = 0.01 * np.ones_like(I_right)
dI_double = 0.01 * np.ones_like(I_double)

print(f"Max normalized intensities:")
print(f"  Left: {I_left.max():.3f}, Right: {I_right.max():.3f}, Double: {I_double.max():.3f}")

# ============================================================================
# PLOT RAW DATA
# ============================================================================
print("\nCreating raw data plots...")

B.pl.figure(figsize=(15, 4))

B.pl.subplot(1, 3, 1)
B.plot_exp(S_left, V_left, color='blue', marker='o', markersize=4)
B.pl.xlabel('Position (mm)')
B.pl.ylabel('Voltage (mV)')
B.pl.title('Left Slit - Raw Data')
B.pl.grid(True, alpha=0.3)

B.pl.subplot(1, 3, 2)
B.plot_exp(S_right, V_right, color='green', marker='s', markersize=4)
B.pl.xlabel('Position (mm)')
B.pl.ylabel('Voltage (mV)')
B.pl.title('Right Slit - Raw Data')
B.pl.grid(True, alpha=0.3)

B.pl.subplot(1, 3, 3)
B.plot_exp(S_double, V_double, color='red', marker='^', markersize=3)
B.pl.xlabel('Position (mm)')
B.pl.ylabel('Voltage (mV)')
B.pl.title('Double Slit - Raw Data')
B.pl.grid(True, alpha=0.3)
B.pl.suptitle('Raw Voltage Measurements', fontsize=14)
B.pl.tight_layout()

# ============================================================================
# DEFINE FIT FUNCTIONS
# ============================================================================
print("\nDefining fit functions...")

# Single slit parameters
D_single = B.Parameter(0.085e-3, 'D')  # Slit width
x0_single = B.Parameter(0.0, 'x0')     # Position offset
I0_single = B.Parameter(1.0, 'I0')     # Maximum intensity

def single_slit_intensity(theta):
    """Single slit diffraction pattern"""
    k = 2.0 * np.pi / lam
    phi = k * D_single() * (theta - x0_single())
    with np.errstate(divide='ignore', invalid='ignore'):
        sinc = np.sin(phi/2.0) / (phi/2.0)
        sinc = np.where(np.abs(phi) < 1e-10, 1.0, sinc)
    return I0_single() * sinc**2

# Double slit parameters
D_double = B.Parameter(0.085e-3, 'D')   # Individual slit width
S_double = B.Parameter(0.406e-3, 'S')   # Slit separation
x0_double = B.Parameter(0.0, 'x0')      # Position offset
I0_double = B.Parameter(1.0, 'I0')      # Maximum intensity

def double_slit_intensity(theta):
    """Double slit interference pattern"""
    k = 2.0 * np.pi / lam
    phi = k * D_double() * (theta - x0_double())
    psi = k * S_double() * (theta - x0_double())
    with np.errstate(divide='ignore', invalid='ignore'):
        sinc = np.sin(phi/2.0) / (phi/2.0)
        sinc = np.where(np.abs(phi) < 1e-10, 1.0, sinc)
    return I0_double() * sinc**2 * np.cos(psi/2.0)**2

# ============================================================================
# FIT LEFT SLIT
# ============================================================================
print("\n" + "="*60)
print("FITTING LEFT SLIT DATA")
print("="*60)

max_idx = np.argmax(I_left)
x0_single.set(theta_left[max_idx] + 1e-6)  # Small offset to avoid division by zero
print(f"Initial center position: {x0_single():.6f} rad")

try:
    fit_left = B.genfit(single_slit_intensity, 
                        [D_single, x0_single, I0_single],
                        x=theta_left, 
                        y=I_left, 
                        y_err=dI_left)
    
    # Extract values and errors using .err attribute (confirmed to work)
    D_left_value = D_single()
    D_left_error = D_single.err
    x0_left_value = x0_single()
    x0_left_error = x0_single.err
    I0_left_value = I0_single()
    I0_left_error = I0_single.err
    
    print("\nLeft Slit Fit Results:")
    print(f"Slit width D = {D_left_value:.3e} ± {D_left_error:.3e} m")
    print(f"             = {D_left_value*1e6:.1f} ± {D_left_error*1e6:.1f} μm")
    print(f"Position x0  = {x0_left_value:.6f} ± {x0_left_error:.6f} rad")
    print(f"Intensity I0 = {I0_left_value:.3f} ± {I0_left_error:.3f}")
    print(f"Chi-squared  = {fit_left.chi2:.2f}")
    print(f"Reduced χ²   = {fit_left.chi2/(len(theta_left)-3):.2f}")
    
    # Plot left slit fit
    B.pl.figure(figsize=(10, 6))
    B.plot_line(fit_left.xpl*1000, fit_left.ypl, color='red', linewidth=2, label='Left Fit')
    B.pl.xlabel('Angle θ (mrad)', fontsize=12)
    B.pl.ylabel('Normalized Intensity', fontsize=12)
    B.pl.title(f'Left Slit - Data and Fit\nD = {D_left_value*1e6:.1f}±{D_left_error*1e6:.1f} μm', fontsize=14)
    B.pl.legend(fontsize=11)
    B.pl.grid(True, alpha=0.3)
    
except Exception as e:
    print(f"Error fitting left slit: {e}")
    D_left_value = 0.085e-3

# ============================================================================
# FIT RIGHT SLIT
# ============================================================================
print("\n" + "="*60)
print("FITTING RIGHT SLIT DATA")
print("="*60)

# Reset parameters
D_single.set(0.085e-3)
I0_single.set(1.0)
max_idx = np.argmax(I_right)
x0_single.set(theta_right[max_idx] + 1e-6)
print(f"Initial center position: {x0_single():.6f} rad")

try:
    fit_right = B.genfit(single_slit_intensity,
                         [D_single, x0_single, I0_single],
                         x=theta_right,
                         y=I_right,
                         y_err=dI_right)
    
    # Extract values and errors
    D_right_value = D_single()
    D_right_error = D_single.err
    x0_right_value = x0_single()
    x0_right_error = x0_single.err
    I0_right_value = I0_single()
    I0_right_error = I0_single.err
    
    print("\nRight Slit Fit Results:")
    print(f"Slit width D = {D_right_value:.3e} ± {D_right_error:.3e} m")
    print(f"             = {D_right_value*1e6:.1f} ± {D_right_error*1e6:.1f} μm")
    print(f"Position x0  = {x0_right_value:.6f} ± {x0_right_error:.6f} rad")
    print(f"Intensity I0 = {I0_right_value:.3f} ± {I0_right_error:.3f}")
    print(f"Chi-squared  = {fit_right.chi2:.2f}")
    print(f"Reduced χ²   = {fit_right.chi2/(len(theta_right)-3):.2f}")
    
    # Plot right slit fit
    B.plot_line(fit_right.xpl*1000, fit_right.ypl, color='orange', linewidth=2, label='Right Fit')
    B.pl.xlabel('Angle θ (mrad)', fontsize=12)
    B.pl.ylabel('Normalized Intensity', fontsize=12)
    B.pl.title(f'Right Slit - Data and Fit\nD = {D_right_value*1e6:.1f}±{D_right_error*1e6:.1f} μm', fontsize=14)
    B.pl.legend(fontsize=11)
    B.pl.grid(True, alpha=0.3)
    
except Exception as e:
    print(f"Error fitting right slit: {e}")
    D_right_value = 0.085e-3

# ============================================================================
# FIT DOUBLE SLIT
# ============================================================================
print("\n" + "="*60)
print("FITTING DOUBLE SLIT DATA")
print("="*60)

avg_slit_width = (D_left_value + D_right_value) / 2
print(f"Using average slit width from single slits: {avg_slit_width*1e6:.1f} μm")

max_idx = np.argmax(I_double)
x0_double.set(theta_double[max_idx] + 1e-6)

# Try different slit separations
slit_separations = [0.356e-3, 0.406e-3, 0.457e-3]  # 14, 16, 18 mils
best_chi2 = float('inf')
best_fit = None
best_params = {}

print("\nTrying different slit separations...")
for S_test in slit_separations:
    S_double.set(S_test)
    D_double.set(avg_slit_width)
    I0_double.set(1.0)
    
    try:
        fit_test = B.genfit(double_slit_intensity,
                           [D_double, S_double, x0_double, I0_double],
                           x=theta_double,
                           y=I_double,
                           y_err=dI_double)
        
        chi2_reduced = fit_test.chi2/(len(theta_double)-4)
        print(f"  S = {S_test*1e3:.3f} mm ({S_test/25.4e-6:.0f} mils): χ²_reduced = {chi2_reduced:.2f}")
        
        if fit_test.chi2 < best_chi2:
            best_chi2 = fit_test.chi2
            best_fit = fit_test
            # Store best parameter values
            best_params = {
                'D_value': D_double(),
                'D_error': D_double.err,
                'S_value': S_double(),
                'S_error': S_double.err,
                'x0_value': x0_double(),
                'x0_error': x0_double.err,
                'I0_value': I0_double(),
                'I0_error': I0_double.err
            }
    except Exception as e:
        print(f"  S = {S_test*1e3:.3f} mm: Fit failed")

if best_fit and best_params:
    print(f"\nBest fit achieved with S = {best_params['S_value']*1e3:.3f} mm")
    print("\nDouble Slit Fit Results:")
    print(f"Slit width D      = {best_params['D_value']:.3e} ± {best_params['D_error']:.3e} m")
    print(f"                  = {best_params['D_value']*1e6:.1f} ± {best_params['D_error']*1e6:.1f} μm")
    print(f"Slit separation S = {best_params['S_value']:.3e} ± {best_params['S_error']:.3e} m")
    print(f"                  = {best_params['S_value']*1e3:.3f} ± {best_params['S_error']*1e3:.3f} mm")
    print(f"                  = {best_params['S_value']/25.4e-6:.1f} ± {best_params['S_error']/25.4e-6:.1f} mils")
    print(f"Position x0       = {best_params['x0_value']:.6f} ± {best_params['x0_error']:.6f} rad")
    print(f"Intensity I0      = {best_params['I0_value']:.3f} ± {best_params['I0_error']:.3f}")
    print(f"Chi-squared       = {best_fit.chi2:.2f}")
    print(f"Reduced χ²        = {best_fit.chi2/(len(theta_double)-4):.2f}")
    
    # Plot double slit fit
    B.plot_line(best_fit.xpl*1000, best_fit.ypl, color='black', linewidth=2, label='Double Fit')
    B.pl.xlabel('Angle θ (mrad)', fontsize=12)
    B.pl.ylabel('Normalized Intensity', fontsize=12)
    B.pl.title(f'Double Slit - Data and Fit\nD = {best_params["D_value"]*1e6:.1f}±{best_params["D_error"]*1e6:.1f} μm, S = {best_params["S_value"]*1e3:.3f}±{best_params["S_error"]*1e3:.3f} mm', 
              fontsize=14)
    B.pl.legend(fontsize=11)
    B.pl.grid(True, alpha=0.3)
    
    # ========================================================================
    # ADDITIONAL CALCULATIONS
    # ========================================================================
    print("\n" + "="*60)
    print("ADDITIONAL CALCULATIONS")
    print("="*60)
    
    fringe_spacing_rad = lam / best_params['S_value']
    fringe_spacing_mm = fringe_spacing_rad * L * 1e3
    central_max_width = 2 * lam / best_params['D_value']
    num_fringes = central_max_width / fringe_spacing_rad
    
    print(f"\nFringe spacing:")
    print(f"  Angular: {fringe_spacing_rad*1000:.3f} mrad ({np.degrees(fringe_spacing_rad):.4f}°)")
    print(f"  Linear at detector: {fringe_spacing_mm:.3f} mm")
    
    print(f"\nCentral maximum width:")
    print(f"  Angular: {central_max_width*1000:.3f} mrad ({np.degrees(central_max_width):.4f}°)")
    print(f"  Number of fringes in central maximum: {num_fringes:.1f}")
    
    print(f"\nSlit width comparison:")
    print(f"  Left slit:   {D_left_value*1e6:.1f} ± {D_left_error*1e6:.1f} μm")
    print(f"  Right slit:  {D_right_value*1e6:.1f} ± {D_right_error*1e6:.1f} μm")
    print(f"  Double slit: {best_params['D_value']*1e6:.1f} ± {best_params['D_error']*1e6:.1f} μm")
    print(f"  Average of single slits: {avg_slit_width*1e6:.1f} μm")
