# MPLC-Uni-Final-Project

This repository contains two tightly coupled toolchains that together produce the phase masks required for our multi‑plane light converter (MPLC) experiment and, crucially, determine the **mode‑field diameter (MFD)** of the laser beam that is used as an external design parameter.

## Prerequisites

* **MATLAB R2022a or newer** with the Image Processing Toolbox.
* **Python 3.9+** with `numpy`, `scipy`, and `matplotlib` (a basic `pip install -r requirements.txt` suffices).
* A reflective phase‑only Spatial Light Modulator (SLM) with an 8‑bit greyscale input.

## Quick Start

1. **Measure the beam and calculate its MFD.** Run the Python script supplied in `/workspaces/MPLC-Uni-Final-Project/MFD_calculation.py`. The script asks for the excel file that includes the histogram data that was prodeuced using ImageJ with the images that were taken at the waist  from 0 cm and 60 cm distance and outputs a single number—your beam’s 1⁄e² diameter—in micrometres.
2. **Insert the MFD into the MATLAB optimiser.** Open `/workspaces/MPLC-Uni-Final-Project/MATLAB_code_SLM /MPLC_StartHere.m` and set the variable `MFD_in` to the value obtained in Step 1.
3. **Generate the phase masks.** Simply run `MPLC_StartHere.m`. The script performs the inverse‑design optimisation and writes a stack of phase arrays (`mask_01.mat`, `mask_02.mat`, …) into `output/mat/`.
4. **Convert the masks to an SLM‑ready image.** Execute `phase_mask_to_gray_image.m`. It quantises each phase array to 8‑bit greyscale and saves a PNG sequence (`mask_01.png`, `mask_02.png`, …) inside `output/png/`. These images can be loaded directly into the SLM controller.

## Repository Layout

## Citation

---
