# PSF Analysis

Extract PSFs and find aberrations like a tilt and a banana shape of the psf-core.

## 1. Principle

This script segments a blob around each PSF and processes the radial profile (with a rotating plane) of each one.
We expect the graph of a "good" PSF to be rather flat, while a "banana" or a tilted PSF should have dramatic fluctations.

## 2. Content

- "psf_analysis.py": Meant to be used in Fiji (Jython).
- "show-profiles.py": Meant to be used with Python3. Uses MatPlotLib to display a graph containing all the plots representing radial profiles.
