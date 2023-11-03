# PSF Analysis

Extract PSFs and find aberrations like a tilt and a banana shape of the psf-core.

## 1. Principle

This script segments a blob around each PSF and processes the radial profile (with a rotating plane) of each one.
We expect the graph of a "good" PSF to be rather flat, while a "banana" or a tilted PSF should have dramatic fluctations.

## 2. Content

- "psf_analysis.py": Meant to be used in Fiji (Jython).
- "show-profiles.py": Meant to be used with Python3. Uses MatPlotLib to display a graph containing all the plots representing radial profiles.

## 3. TO-DO

- [ ] Add some 'verbose' in the functions to keep the user aware of what is going on.
- [ ] Show a dialog to let the user choose settings instead of having them hard-coded.
- [ ] Add a dilation to the labels because they don't take enough space around each PSF and might contain holes.
- [ ] For the values to be more readable, we should normalize images before starting (in [0.0, 1.0]).
- [ ] Transform this script into a Jython module.
- [ ] Assessment about "good" vs. "bad" PSFs.
- [ ] Write unit-tests for the functions + cut big functions into unit tasks.
- [ ] Handle error correctly.