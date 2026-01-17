#  MIST - Regression Analyzer

MIST - Regression Analyzer is an interactive Python-based tool designed for the analysis of Eclipse Timing Variations in binary star systems. It provides a graphical interface (GUI) to visualize O-C (Observed minus Calculated) diagrams and perform non-linear regression analysis using Parabolic, Sinusoidal, and Combined models.

Developed as part of an Astrophysics M.Sc. thesis, this tool assists in identifying secular period changes (Mass Transfer) and periodic modulations (Light-Time Effect).

<img width="1910" height="1116" alt="image" src="https://github.com/user-attachments/assets/0545e575-b4a8-4da6-8ff6-1a5cf50fffb9" />


# Key Features

* **Interactive GUI:** Built with **PyQt6** and **Matplotlib** for real-time interaction.
* **Multi-Model Fitting:**
    * **Linear Fit:** For updating linear ephemeris (T0, P).
    * **Parabolic Fit:** For secular period changes (Mass Transfer).
    * **Sinusoidal Fit:** For periodic effects (LITE).
    * **Combined Fit:** Simultaneous analysis of secular and periodic changes.
* **Robust Algorithms:** Uses **SciPy's** `curve_fit` (Levenberg-Marquardt algorithm) for non-linear least squares minimization.
* **Astrophysical Parameters:** Automatically calculates:
    * Mass transfer rate.
    * 3rd Body parameters.
* **Machine Learning Ready:** Exports extracted features (RMS, coefficients, Q-values) to CSV for ML classification tasks.
* **Publication Quality:** Exports regression plots in EPS, PDF, and PNG formats.

## Installation

### Prerequisites
* Python 3.8+
* pip

### Dependencies
Install the required libraries using the `requirements.txt` file:
# Usage
* Run the Tool : python3 regression.py
* Load Data: Click "Load CSV Data". (The tool expects a standard .csv file. Please refer to the regression.csv file included in this repository for a ready-to-use example.)
* Set Priors: Enter the initial Reference Epoch (T0) and Period (P) in the input boxes (top left).
* Use the toolbar buttons to toggle between Parabolic, Sinusoidal, or Combined fit.
* Click "Calculate Parameters" to derive physical values (Mass Transfer, 3rd Body Mass).
* Export: Save your plots or export results for Machine Learning.

```bash
git clone https://github.com/thematicmathematics/MIST---Regression-Analyzer
cd MIST---Regression-Analyzer
pip install -r requirements.txt
