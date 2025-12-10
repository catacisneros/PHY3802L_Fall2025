# Photoelectric-Effect
# Photoelectric Effect

## Objective
To verify Einstein’s photoelectric equation and determine Planck’s constant (h) by measuring stopping potential as a function of frequency.

## Method
- Light of known frequencies illuminated a photoemissive cathode.
- Stopping potential (Vₛ) was measured for each wavelength.
- Relation used:
  \[
  eV_s = hf - \phi
  \]
- A linear fit of \( V_s \) vs \( f \) yields slope \( h/e \) and intercept \( -\phi/e \).

## Analysis
Python + `LT.box.linefit` were used for fitting and uncertainty estimation.  
Chi-squared values confirmed fit quality.

## Results
- Planck’s constant: \( h = (6.62 ± 0.14) \times 10^{-34} \, J·s \)
- Work function: \( \phi = 2.2 ± 0.1 \, eV \)

## Key Takeaways
- Confirmed the quantized nature of light.
- Strengthened data analysis and linear fitting skills.
