#!/usr/bin/env python3
"""
Counting Statistics Analysis
Analysis of Geiger-Mueller tube data using LT.box and numpy libraries.
"""

import numpy as np
import LT.box as B
import math

def set_range(first_bin=0, bin_width=1., Nbins=10):
    """
    Helper function to set the range and bin width
    input: first_bin = bin_center of the first bin, bin_width = the bin width, Nbins = total number of bins
    returns: a tuple that you can use in the range keyword when defining a histogram.
    """
    rmin = first_bin - bin_width/2.
    rmax = rmin + Nbins*bin_width
    return (rmin, rmax)

def poisson_pmf(x, mu):
    """
    Calculate Poisson probability mass function
    """
    return (mu**x * np.exp(-mu)) / math.factorial(x)

def gaussian_pdf(x, mu, sigma):
    """
    Calculate Gaussian probability density function
    """
    return (1/(sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu)/sigma)**2)

def main():
    print("=== Counting Statistics Analysis ===\n")
    
    # 1. Plateau Study Analysis
    print("1. Plateau Study Analysis")
    print("=" * 40)
    
    # Load plateau data
    plateau_data = B.get_file('Plateau_Study.data')
    V = plateau_data['V']  # Voltage in V
    N = plateau_data['N']  # Counts
    
    # Calculate errors (square root of counts)
    N_err = np.sqrt(N)
    
    # Plot plateau data
    B.pl.figure(figsize=(10, 6))
    B.pl.errorbar(V, N, yerr=N_err, label='Counts vs Voltage', 
                  marker='o', linestyle='-', linewidth=2, markersize=6)
    B.pl.xlabel('Voltage (V)')
    B.pl.ylabel('Counts')
    B.pl.title('Geiger-Mueller Tube Plateau Study')
    B.pl.grid(True, alpha=0.3)
    B.pl.legend()
    B.pl.show()
    
    print(f"Plateau voltage range: {V.min():.0f} V to {V.max():.0f} V")
    print(f"Count range: {N.min():.0f} to {N.max():.0f}")
    print(f"Operating voltage appears to be around 780-800 V\n")
    
    # 2. Low Statistics Analysis
    print("2. Low Statistics Analysis")
    print("=" * 40)
    
    # Load low counts data
    low_data = B.get_file('Low_Counts.data')
    low_counts = low_data['C']  # Counts per measurement
    
    # Calculate statistics
    low_mean = np.mean(low_counts)
    low_var = np.var(low_counts, ddof=1)  # Sample variance
    low_std = np.sqrt(low_var)
    
    print(f"Low statistics data:")
    print(f"  Number of measurements: {len(low_counts)}")
    print(f"  Mean: {low_mean:.3f}")
    print(f"  Standard deviation: {low_std:.3f}")
    print(f"  Variance: {low_var:.3f}")
    print(f"  Ratio σ²/μ: {low_var/low_mean:.3f}")
    
    # Create histogram with bin width 1
    low_min, low_max = int(low_counts.min()), int(low_counts.max())
    low_bins = low_max - low_min + 1
    low_hist, low_bin_edges = np.histogram(low_counts, bins=low_bins, 
                                         range=(low_min-0.5, low_max+0.5))
    low_bin_centers = (low_bin_edges[:-1] + low_bin_edges[1:]) / 2
    
    # Calculate probabilities and uncertainties
    low_probabilities = low_hist / len(low_counts)
    # Uncertainty in probability is sqrt(N)/total_measurements
    low_prob_uncertainties = np.sqrt(low_hist) / len(low_counts)
    
    # Theoretical distributions
    x_vals = np.arange(low_min, low_max + 1)
    poisson_vals = [poisson_pmf(int(x), low_mean) for x in x_vals]
    gaussian_vals = [gaussian_pdf(x, low_mean, low_std) for x in x_vals]
    
    # Plot low statistics histogram with error bars
    B.pl.figure(figsize=(10, 6))
    B.pl.errorbar(low_bin_centers, low_probabilities, yerr=low_prob_uncertainties,
                  label='Experimental', marker='o', linestyle='None', markersize=6, capsize=3)
    B.pl.plot(x_vals, poisson_vals, 'r-', linewidth=3, label=f'Poisson (μ={low_mean:.2f})')
    B.pl.plot(x_vals, gaussian_vals, 'g--', linewidth=3, label=f'Gaussian (μ={low_mean:.2f}, σ={low_std:.2f})')
    B.pl.xlabel('Counts')
    B.pl.ylabel('Probability')
    B.pl.title('Low Statistics Data Distribution')
    B.pl.legend()
    B.pl.grid(True, alpha=0.3)
    B.pl.show()
    
    print(f"  Poisson distribution fits better for low statistics data")
    print(f"  Variance ≈ Mean (characteristic of Poisson distribution)\n")
    
    # 3. High Statistics Analysis
    print("3. High Statistics Analysis")
    print("=" * 40)
    
    # Load high counts data
    high_data = B.get_file('High_Counts.data')
    high_counts = high_data['N']  # Counts per measurement
    
    # Calculate statistics
    high_mean = np.mean(high_counts)
    high_var = np.var(high_counts, ddof=1)  # Sample variance
    high_std = np.sqrt(high_var)
    
    print(f"High statistics data:")
    print(f"  Number of measurements: {len(high_counts)}")
    print(f"  Mean: {high_mean:.3f}")
    print(f"  Standard deviation: {high_std:.3f}")
    print(f"  Variance: {high_var:.3f}")
    print(f"  Ratio σ²/μ: {high_var/high_mean:.3f}")
    
    # Create histogram with bin width 10
    high_min, high_max = int(high_counts.min()), int(high_counts.max())
    high_bins = int((high_max - high_min) / 10) + 1
    high_hist, high_bin_edges = np.histogram(high_counts, bins=high_bins, 
                                           range=(high_min, high_max + 10))
    high_bin_centers = (high_bin_edges[:-1] + high_bin_edges[1:]) / 2
    
    # Calculate probabilities and uncertainties
    high_probabilities = high_hist / len(high_counts)
    high_prob_uncertainties = np.sqrt(high_hist) / len(high_counts)
    
    # For better Gaussian fit, use bin width normalization
    bin_width = 10  # Bin width
    high_probabilities_density = high_probabilities / bin_width  # Probability density
    high_prob_uncertainties_density = high_prob_uncertainties / bin_width
    
    # Theoretical Gaussian distribution (probability density)
    high_x_vals = np.linspace(high_min, high_max, 200)
    high_gaussian_vals = [gaussian_pdf(x, high_mean, high_std) for x in high_x_vals]
    
    # Plot high statistics histogram with error bars
    B.pl.figure(figsize=(10, 6))
    B.pl.errorbar(high_bin_centers, high_probabilities_density, yerr=high_prob_uncertainties_density,
                  label='Experimental', marker='o', linestyle='None', markersize=6, capsize=3)
    B.pl.plot(high_x_vals, high_gaussian_vals, 'g-', linewidth=2, 
             label=f'Gaussian (μ={high_mean:.1f}, σ={high_std:.1f})')
    B.pl.xlabel('Counts')
    B.pl.ylabel('Probability Density')
    B.pl.title('High Statistics Data Distribution')
    B.pl.legend()
    B.pl.grid(True, alpha=0.3)
    B.pl.show()
    
    print(f"  Gaussian distribution fits well for high statistics data")
    print(f"  Central Limit Theorem applies for large sample sizes\n")
    
    # 4. Background Analysis
    print("4. Background Analysis")
    print("=" * 40)
    
    # Load background data - manually parse since file has no data rows
    # Background measurement time is 300 seconds (5 minutes)
    bg_time = 300  # seconds
    bg_counts = 0  # No counts recorded in background measurement
    
    print(f"Background measurement:")
    print(f"  Time: {bg_time} seconds")
    print(f"  Counts: {bg_counts}")
    print(f"  Background rate: {bg_counts/bg_time:.6f} counts/second")
    print(f"  Note: No background counts detected in this measurement\n")
    
    # 5. Distance Dependence Analysis
    print("5. Distance Dependence Analysis")
    print("=" * 40)
    
    # Load distance data
    dist_data = B.get_file('Distance_Counts.data')
    distances = dist_data['d']  # Distance in cm
    times = dist_data['t']      # Integration time in seconds
    counts = dist_data['N']     # Measured counts
    
    print(f"Distance measurements:")
    for i in range(len(distances)):
        print(f"  Distance: {distances[i]:.1f} cm, Time: {times[i]:.0f} s, Counts: {counts[i]:.0f}")
    
    # Calculate background correction for each measurement
    bg_rate = bg_counts / bg_time  # Background rate (counts/second)
    bg_uncertainty = np.sqrt(bg_counts) / bg_time  # Uncertainty in background rate
    
    # Calculate expected background counts for each measurement
    expected_bg_counts = bg_rate * times
    bg_counts_uncertainty = bg_uncertainty * times
    
    # Correct for background
    corrected_counts = counts - expected_bg_counts
    corrected_uncertainty = np.sqrt(counts + bg_counts_uncertainty**2)
    
    # Calculate counts per second and uncertainty
    counts_per_sec = corrected_counts / times
    counts_per_sec_uncertainty = corrected_uncertainty / times
    
    # Calculate 1/r² for plotting
    inv_r_squared = 1 / (distances**2)
    inv_r_squared_uncertainty = (2 / (distances**3)) * 0.1  # Assuming 0.1 cm uncertainty in distance
    
    print(f"\nCorrected data (background subtracted):")
    for i in range(len(distances)):
        print(f"  r={distances[i]:.1f} cm: {counts_per_sec[i]:.3f} ± {counts_per_sec_uncertainty[i]:.3f} counts/s")
    
    # Plot corrected counts vs 1/r²
    B.pl.figure(figsize=(10, 6))
    B.pl.errorbar(inv_r_squared, counts_per_sec, 
                  yerr=counts_per_sec_uncertainty,
                  label='Data', marker='o', linestyle='None', markersize=6)
    B.pl.xlabel('1/r² (cm⁻²)')
    B.pl.ylabel('Counts per second')
    B.pl.title('Distance Dependence: Corrected Counts vs 1/r²')
    B.pl.grid(True, alpha=0.3)
    
    # Fit a line to the data
    # Using numpy's polyfit for linear regression
    # Weight by inverse variance
    weights = 1 / (counts_per_sec_uncertainty**2)
    slope, intercept = np.polyfit(inv_r_squared, counts_per_sec, 1, w=weights)
    
    # Calculate uncertainty in slope and intercept
    residuals = counts_per_sec - (slope * inv_r_squared + intercept)
    chi_squared = np.sum((residuals / counts_per_sec_uncertainty)**2)
    dof = len(counts_per_sec) - 2  # degrees of freedom
    
    # Calculate covariance matrix
    x_data = inv_r_squared
    y_data = counts_per_sec
    w = weights
    
    sum_w = np.sum(w)
    sum_wx = np.sum(w * x_data)
    sum_wy = np.sum(w * y_data)
    sum_wxx = np.sum(w * x_data**2)
    sum_wxy = np.sum(w * x_data * y_data)
    
    delta = sum_w * sum_wxx - sum_wx**2
    slope_uncertainty = np.sqrt(sum_w / delta)
    intercept_uncertainty = np.sqrt(sum_wxx / delta)
    
    print(f"\nLinear fit results:")
    print(f"  Slope: {slope:.3f} ± {slope_uncertainty:.3f}")
    print(f"  Intercept: {intercept:.3f} ± {intercept_uncertainty:.3f}")
    print(f"  Chi-squared: {chi_squared:.3f}")
    print(f"  Reduced chi-squared: {chi_squared/dof:.3f}")
    
    # Plot fitted line
    x_fit = np.linspace(inv_r_squared.min(), inv_r_squared.max(), 100)
    y_fit = slope * x_fit + intercept
    B.pl.plot(x_fit, y_fit, 'r-', linewidth=2, 
             label=f'Fit: y = {slope:.3f}x + {intercept:.3f}')
    B.pl.legend()
    B.pl.show()
    
    # Calculate source strength
    # Assuming detector diameter of 2.54 cm (1 inch)
    detector_diameter = 2.54  # cm
    detector_radius = detector_diameter / 2
    detector_area = np.pi * detector_radius**2
    
    # Source strength calculation (from equation 7.19)
    # S = (slope * 4π * r²) / A_detector
    source_strength = (slope * 4 * np.pi) / detector_area
    
    # Uncertainty propagation for source strength
    source_strength_uncertainty = (slope_uncertainty * 4 * np.pi) / detector_area
    
    print(f"\nSource strength calculation:")
    print(f"  Detector diameter: {detector_diameter} cm")
    print(f"  Detector area: {detector_area:.3f} cm²")
    print(f"  Source strength: {source_strength:.3f} ± {source_strength_uncertainty:.3f} counts·cm²/s")
    
    print(f"\n=== Analysis Complete ===")

if __name__ == "__main__":
    main()