# Multiplane Light Converter (MPLC)

This repository contains the simulation and experimental framework for designing a **Multiplane Light Converter (MPLC)** system to sort Hermite–Gaussian (HG) modes using a phase-only Spatial Light Modulator (SLM). The project includes both simulation tools (in Python) and guidance for experimental realization in the optics lab.

## Overview

The goal of this project is to develop and test a compact MPLC system capable of cleanly sorting 4 HG modes with low insertion loss (IL) and mode-dependent loss (MDL), using as few SLM reflections as possible.

The platform consists of:

1. **Simulation Pipeline**: Python-based framework to train phase masks via wavefront matching and simulate how HG modes propagate through them.
2. **Experimental Tools**: Scripts for generating upsampled SLM patterns for lab use.
3. **Processing Framework**: Functions and scripts to analyze experimental measurements, build transfer matrices, and compute metrics (IL, MDL, crosstalk).


The design is inspired by:

* [Fontaine et al., *Laguerre–Gaussian Mode Sorter*, Nature Communications 10, 1865 (2019)](https://doi.org/10.1038/s41467-019-09840-4)
* Wavefront Matching optimization methods for inverse phase mask design.

---

## Simulation Setup

### 1. `init.py`

Before running any simulation, configure your parameters in `init.py`:

* `dx`, `dy`: Spacing used for mapping Gaussian beam centers.
* `d`: Distance between phase planes.
* `Nx`, `Ny`: Shape of the mask (excluding padding).
* `win`, `wout`: Waist of input and output Gaussian beams.
* `n_planes`: Number of phase masks.
* `width`, `pad`: Parameters for the absorber mask.
* `wl`: Wavelength used in lab experiments.
* `pp`: Pixel pitch of the SLM.
* `file_name`: Filename for saving and loading the trained system.

### 2. Training

Run the training script to generate optimized phase masks for the desired mode conversion:

```bash
python Train.py
```

This will use the wavefront-matching algorithm described in Fontaine et al. to iteratively update the phase masks. The final system is saved to a pickle file defined in `file_name`.

### 3. Testing

Once training is complete, you can simulate the propagation of HG modes and visualize the results:

```bash
python test.py
```

This script:

* Simulates the propagation of an input mode (e.g. `HG01`) through the trained MPLC system.
* Generates an animation of the intensity and phase across propagation.
* Computes and visualizes the transfer matrix and key metrics:

  * **Insertion Loss (IL)**
  * **Mode Dependent Loss (MDL)**

### 4. Generating SLM Patterns

To visualize and export the final phase masks for lab experiments:

```bash
python slm_pattern_generate.py
```

This script:

* Upsamples and normalizes the phase masks to the range \[0,255].
* Combines the masks horizontally.
* Saves the result as an 8-bit grayscale image ready for SLM upload.

Use `positions` to define the mask locations on the SLM (i.e., specify the centres you chose for each mask, either for your setup or to simplify calibration). 

## Processing Experimental Results

In addition to training and simulation, the repository provides tools for analyzing **experimental measurements**.

### 1. Processing Functions

The `processing_functions` module includes reusable utilities to:

* Load experimental images.  
* Detect Gaussian spots in the output plane.  
* Construct the transfer matrix **T** from measured input–output mode mappings.  
* Compute system metrics such as **IL**, **MDL**, and crosstalk.  

These functions act as the building blocks for all analysis.

### 2. Processing Results Script

To process experimental data, configure the input/output paths and run:

```bash
python processing_results.py
```

---

## File Structure

| File                      | Description                                                    |
| ------------------------- | -------------------------------------------------------------- |
| `init.py`                 | Configuration of physical and simulation parameters            |
| `Train.py`                | Trains MPLC phase masks using wavefront matching               |
| `test.py`                 | Visualizes propagation, computes IL/MDL, plots transfer matrix |
| `slm_pattern_generate.py` | Converts trained phase masks into grayscale images for the SLM |
| `mode.py`                 | HG mode generation and propagation via angular spectrum        |
| `phase_plane.py`          | Defines a phase mask and how it affects a mode                 |
| `MPLC.py`                 | Core class for MPLC propagation, training, and evaluation      |
| `animator.py`             | Utility for creating intensity and phase animations            |
| `processing_functions.py` | Utility functions for loading data, detecting spots, and computing metrics (IL, MDL, crosstalk) |
| `processing_results.py`   | Main script to analyze experimental results and export figures|

---

## References

* Fontaine, N. K. et al. (2019). *Laguerre-Gaussian mode sorter*, Nature Communications, 10:1865. [https://doi.org/10.1038/s41467-019-09840-4](https://doi.org/10.1038/s41467-019-09840-4)

---

## Citation

If you use this codebase in your academic work, please consider citing the references above and this repository.
