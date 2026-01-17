import sys
import os
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QLineEdit, QPushButton, 
                             QTextEdit, QFileDialog, QMessageBox, QGroupBox, QDialog, QGridLayout, QInputDialog) 
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QIcon, QFont, QAction 
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt
class MIST_Toolbar(NavigationToolbar):
    def __init__(self, canvas, parent, coordinates=True):
        super().__init__(canvas, parent, coordinates)
        self.addSeparator()
        self.act_par = self.addAction("Parabola")
        self.act_par.setCheckable(True); self.act_par.setChecked(True)
        self.act_par.triggered.connect(parent.run_manual_plot)
        self.act_sin = self.addAction("Sinusoid")
        self.act_sin.setCheckable(True); self.act_sin.setChecked(True)
        self.act_sin.triggered.connect(parent.run_manual_plot)
        self.addSeparator()
        self.act_comb = self.addAction("Combined")
        self.act_comb.setCheckable(True)
        self.act_comb.setChecked(True)
        self.act_comb.triggered.connect(parent.run_manual_plot)
        self.addSeparator()
        self.act_exp_res = self.addAction("Exp. CSV")
        self.act_exp_res.triggered.connect(parent.export_results)
        self.act_exp_ml = self.addAction("Exp. ML")
        self.act_exp_ml.triggered.connect(parent.export_ml_data)         
class ParameterDialog(QDialog):
    def __init__(self, par_params, sine_params, p_orb, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Manual Parameter Adjustment")
        self.setGeometry(200, 200, 400, 300)
        self.p_orb = p_orb
        self.new_par = par_params
        self.new_sin = sine_params        
        layout = QVBoxLayout()               
        g1 = QGroupBox("Parabolic Parameters")
        l1 = QGridLayout()
        self.in_c2 = QLineEdit(f"{par_params[0]:.4e}"); l1.addWidget(QLabel("c2 (Q):"), 0, 0); l1.addWidget(self.in_c2, 0, 1)
        self.in_c1 = QLineEdit(f"{par_params[1]:.6f}"); l1.addWidget(QLabel("c1 (dP):"), 1, 0); l1.addWidget(self.in_c1, 1, 1)
        self.in_c0 = QLineEdit(f"{par_params[2]:.6f}"); l1.addWidget(QLabel("c0 (dT0):"), 2, 0); l1.addWidget(self.in_c0, 2, 1)
        g1.setLayout(l1); layout.addWidget(g1)               
        g2 = QGroupBox("Sinusoidal Parameters")
        l2 = QGridLayout()
        p_cyc = sine_params[1]
        p_yr = (p_cyc * p_orb) / 365.25 if p_orb > 0 else 0
        self.in_A = QLineEdit(f"{sine_params[0]:.5f}"); l2.addWidget(QLabel("Amp (A):"), 0, 0); l2.addWidget(self.in_A, 0, 1)
        self.in_Pyr = QLineEdit(f"{p_yr:.2f}"); l2.addWidget(QLabel("Per (Year):"), 1, 0); l2.addWidget(self.in_Pyr, 1, 1)
        self.in_phi = QLineEdit(f"{sine_params[2]:.4f}"); l2.addWidget(QLabel("Phase (rad):"), 2, 0); l2.addWidget(self.in_phi, 2, 1)
        self.in_off = QLineEdit(f"{sine_params[3]:.5f}"); l2.addWidget(QLabel("Offset (d):"), 3, 0); l2.addWidget(self.in_off, 3, 1)
        g2.setLayout(l2); layout.addWidget(g2)        
        btn_update = QPushButton("Update & Plot")
        btn_update.setStyleSheet("background-color: darkgreen; color: white; font-weight: bold; height: 30px;")
        btn_update.clicked.connect(self.save_and_close)
        layout.addWidget(btn_update)
        self.setLayout(layout)
    def save_and_close(self):
        try:
            c2, c1, c0 = float(self.in_c2.text()), float(self.in_c1.text()), float(self.in_c0.text())
            self.new_par = [c2, c1, c0]            
            A, P_yr, phi, off = float(self.in_A.text()), float(self.in_Pyr.text()), float(self.in_phi.text()), float(self.in_off.text())
            P_cyc = (P_yr * 365.25) / self.p_orb if self.p_orb > 0 else 1000
            self.new_sin = [A, P_cyc, phi, off]            
            self.accept()
        except ValueError:
            QMessageBox.warning(self, "Error", "Invalid number format!")
class MassInputDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Astrophysical Masses")
        self.setGeometry(300, 300, 300, 150)
        layout = QGridLayout()
        layout.addWidget(QLabel("Primary Mass (M1) [M_sun]:"), 0, 0)
        self.in_m1 = QLineEdit("1.0"); layout.addWidget(self.in_m1, 0, 1)
        layout.addWidget(QLabel("Secondary Mass (M2) [M_sun]:"), 1, 0)
        self.in_m2 = QLineEdit("0.5"); layout.addWidget(self.in_m2, 1, 1)
        btn_ok = QPushButton("Calculate Physics")
        btn_ok.setStyleSheet("background-color: #8A2BE2; color: white; font-weight: bold;")
        btn_ok.clicked.connect(self.accept)
        layout.addWidget(btn_ok, 2, 0, 1, 2)
        self.setLayout(layout)
    def get_masses(self):
        try:
            m1 = float(self.in_m1.text())
            m2 = float(self.in_m2.text())
            return m1, m2
        except: return None, None
class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=6, dpi=100):        
        self.fig = Figure(figsize=(width, height), dpi=dpi, layout=None)             
        gs = self.fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.05)        
        self.ax_top = self.fig.add_subplot(gs[0])
        self.ax_bottom = self.fig.add_subplot(gs[1], sharex=self.ax_top)             
        plt.setp(self.ax_top.get_xticklabels(), visible=False)              
        self.fig.subplots_adjust(bottom=0.10, top=0.95, left=0.12, right=0.95)        
        super(MplCanvas, self).__init__(self.fig)
class MIST_OC_App(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MIST Regression Analyzer")
        self.setGeometry(100, 100, 1000, 850)        
        self.df = None
        self.filename = None        
        self.last_par_params = [0, 0, 0]
        self.last_sin_params = [0, 1000, 0, 0]
        self.last_comb_params = None
        self.ml_data = {} 
        self.initUI()
    def initUI(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)             
        input_group = QGroupBox("Input Parameters")
        input_layout = QHBoxLayout()        
        self.input_t0 = QLineEdit()
        self.input_t0.setPlaceholderText("e.g. 52964.611")
        input_layout.addWidget(QLabel("T0 (BJD):"))
        input_layout.addWidget(self.input_t0)        
        self.input_p = QLineEdit()
        self.input_p.setPlaceholderText("e.g. 3.129427")
        input_layout.addWidget(QLabel("Period (d):"))
        input_layout.addWidget(self.input_p)        
        self.btn_load = QPushButton("Load CSV Data")
        self.btn_load.clicked.connect(self.load_csv)
        input_layout.addWidget(self.btn_load)        
        input_group.setLayout(input_layout)
        main_layout.addWidget(input_group) 
        act_group = QGroupBox("Actions")
        l3 = QVBoxLayout() 
        row1 = QHBoxLayout()
        self.btn_calc = QPushButton("Calculate & Plot")
        self.btn_calc.setStyleSheet("background-color: #2e8b57; color: white; font-weight: bold;")
        self.btn_calc.clicked.connect(self.run_analysis)
        row1.addWidget(self.btn_calc)                    
        self.btn_save_plot = QPushButton("Export Plot (Image)")
        self.btn_save_plot.setStyleSheet("background-color: #4682B4; color: white; font-weight: bold;")
        self.btn_save_plot.clicked.connect(self.save_plot)
        row1.addWidget(self.btn_save_plot)
        l3.addLayout(row1)
        row2 = QHBoxLayout()
        self.btn_compare = QPushButton("Compare (Sin vs Par)")
        self.btn_compare.setStyleSheet("background-color: #40E0D0; white; font-weight: bold;")
        self.btn_compare.clicked.connect(self.run_visual_comparison)
        row2.addWidget(self.btn_compare)              
        self.btn_params = QPushButton("Calculate Parameters")
        self.btn_params.setStyleSheet("background-color: #8A2BE2; color: white; font-weight: bold;") 
        self.btn_params.clicked.connect(lambda: self.calculate_parameters(True))
        row2.addWidget(self.btn_params)
        l3.addLayout(row2)
        row4 = QHBoxLayout()
        self.btn_manual = QPushButton("Update Plot from Boxes")
        self.btn_manual.setStyleSheet("background-color: #FF8C00; color: white; font-weight: bold;")
        self.btn_manual.clicked.connect(self.open_manual_dialog)
        row4.addWidget(self.btn_manual)
        self.btn_reset = QPushButton("Reset App")
        self.btn_reset.setStyleSheet("background-color: #B22222; color: white; font-weight: bold;")
        self.btn_reset.clicked.connect(self.reset_app)
        row4.addWidget(self.btn_reset)
        l3.addLayout(row4)
        act_group.setLayout(l3)
        main_layout.addWidget(act_group)
        self.canvas = MplCanvas(self, width=5, height=6, dpi=100)              
        self.toolbar = MIST_Toolbar(self.canvas, self)           
        plot_group = QGroupBox("Regression Diagram & Residuals")
        plot_layout = QVBoxLayout()
        plot_layout.addWidget(self.toolbar)
        plot_layout.addWidget(self.canvas)
        plot_group.setLayout(plot_layout)
        main_layout.addWidget(plot_group, stretch=2)                
        self.text_result = QTextEdit()
        self.text_result.setReadOnly(True)
        self.text_result.setStyleSheet("background-color: #1e1e1e; color: #00ff00; font-family: Consolas; font-size: 11px;")
        self.text_result.setMaximumHeight(180)                
        result_group = QGroupBox("Analysis Results")
        result_layout = QVBoxLayout()
        result_layout.addWidget(self.text_result)
        result_group.setLayout(result_layout)
        main_layout.addWidget(result_group)
    def load_csv(self):
        fname, _ = QFileDialog.getOpenFileName(self, "Open CSV File", "", "CSV Files (*.csv)")
        if fname:
            self.filename = fname
            try:
                self.df = pd.read_csv(fname)
                self.text_result.append(f"Loaded: {fname}")
                self.text_result.append(f"Data Points: {len(self.df)}")
                self.text_result.append("-" * 30)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Could not read file:\n{e}")    
    def save_plot(self):
        if self.df is None: return        
        fname, _ = QFileDialog.getSaveFileName(self, "Save Figure", "Regression_Diagram.eps", 
                                                "EPS Files (*.eps);;PDF Files (*.pdf);;PNG Image (*.png)")
        if fname:
            try:               
                self.canvas.fig.savefig(fname, dpi=300) 
                self.text_result.append(f"Plot saved successfully: {fname}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Could not save file:\n{e}")    
    def export_results(self):
        if not self.ml_data:
            QMessageBox.warning(self, "Warning", "No results found. Please Calculate first.")
            return
            
        fname, _ = QFileDialog.getSaveFileName(self, "Save Results", "MIST_Results.csv", "CSV Files (*.csv)")
        if fname:
            try:
                data_list = [
                    ("Star_Name", self.ml_data.get("Filename", "Unknown")),
                    ("T0_Input", self.ml_data.get("T0_input", 0)),
                    ("P_Input", self.ml_data.get("P_input", 0)),
                    ("Par_Q", self.ml_data.get("Q_val", 0)),
                    ("Par_Q_Err", self.ml_data.get("Q_err", 0)),
                    ("Sin_Amp", self.ml_data.get("Sin_A", 0)),
                    ("Sin_Per_yr", self.ml_data.get("Sin_P3_yr", 0)),
                    ("Comb_Q", self.ml_data.get("Comb_c2", 0)),
                    ("Comb_Amp", self.ml_data.get("Comb_A", 0)),
                    ("Comb_Per_yr", self.ml_data.get("Comb_P3_yr", 0)),
                    ("Comb_Per_Err", self.ml_data.get("Comb_P3_err", 0)),
                    
                    ("RMS_Error", self.ml_data.get("RMS", 0))
                ]
                data = {"Parameter": [k for k,v in data_list], "Value": [v for k,v in data_list]}
                pd.DataFrame(data).to_csv(fname, index=False)
                QMessageBox.information(self, "Success", "Results saved successfully with separated models!")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Could not save: {e}")
    def export_ml_data(self):
        if not self.ml_data:
            QMessageBox.warning(self, "Warning", "No analysis data. Calculate first.")
            return            
        items = ("0: Parabola", "1: Sinusoid", "2: Par+Sin (Combined)")
        item, ok = QInputDialog.getItem(self, "Select Label", "Choose the classification for this star:", items, 2, False)
        
        if ok and item:
            selected_label = int(item.split(":")[0])
            ml_file = "MIST_ML_Training_Data.csv"
            new_row = {
                "Star_ID": self.ml_data.get("Filename", "Unknown"),
                "T0": self.ml_data.get("T0_input", 0),
                "P0": self.ml_data.get("P_input", 0),
                "RMS_Error": self.ml_data.get("RMS", 0),
                "Label": selected_label
            }            
            if selected_label == 0: 
                new_row["Q_coeff"] = self.ml_data.get("Q_val", 0) 
                new_row["Q_err"] = self.ml_data.get("Q_err", 0)
                new_row["LITE_Amp"] = 0
                new_row["LITE_Per_yr"] = 0
                
            elif selected_label == 1: 
                new_row["Q_coeff"] = 0
                new_row["Q_err"] = 0
                new_row["LITE_Amp"] = self.ml_data.get("Sin_A", 0)
                new_row["LITE_Amp_err"] = self.ml_data.get("Sin_A_err", 0)
                new_row["LITE_Per_yr"] = self.ml_data.get("Sin_P3_yr", 0)
                
            elif selected_label == 2: 
                new_row["Q_coeff"] = self.ml_data.get("Comb_c2", 0)
                new_row["Q_err"] = self.ml_data.get("Comb_c2_err", 0)
                new_row["LITE_Amp"] = self.ml_data.get("Comb_A", 0)
                new_row["LITE_Amp_err"] = self.ml_data.get("Comb_A_err", 0)
                new_row["LITE_Per_yr"] = self.ml_data.get("Comb_P3_yr", 0)
            if new_row["Q_coeff"] != 0:
                new_row["P_dot_sec_yr"] = 2 * new_row["Q_coeff"] * (365.25 / new_row["P0"]) * 86400
            else:
                new_row["P_dot_sec_yr"] = 0
            try:
                if os.path.exists(ml_file):
                    df_ml = pd.read_csv(ml_file)
                    df_new = pd.DataFrame([new_row])
                    df_ml = pd.concat([df_ml, df_new], ignore_index=True)
                else:
                    df_ml = pd.DataFrame([new_row])   
                df_ml.to_csv(ml_file, index=False)            
                QMessageBox.information(self, "ML Export", f"Saved correctly for label: {item}\nP3 Used: {new_row['LITE_Per_yr']:.2f} yr")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Export failed: {e}")
    def parabola(self, x, a, b, c): return a*x**2 + b*x + c
    def sine_wave(self, x, A, P_cyc, phase, offset): return A * np.sin(2 * np.pi * x / P_cyc + phase) + offset    
    def combined_model(self, x, c2, c1, c0, A, P_cyc, phi):
        return (c2*x**2 + c1*x + c0) + (A * np.sin(2 * np.pi * x / P_cyc + phi))
    def get_epoch_oc(self, t0, p):
        times = self.df['Minimum'].values
        ids = self.df['ID'].values.astype(str)
        try: sigma = self.df['Sigma'].values
        except: sigma = np.ones(len(times))
        epochs, ocs, weights = [], [], []
        for t, min_id, sig in zip(times, ids, sigma):
            E_raw = (t - t0) / p
            if "SECONDARY" in min_id.upper() or "SEC" in min_id.upper() or "MIN II" in min_id.upper():
                E = round(E_raw - 0.5) + 0.5
            else:
                E = round(E_raw)
            epochs.append(E)
            ocs.append(t - (t0 + p * E))
            weights.append(1.0/sig if sig > 1e-9 else 1.0/1e-5)
        return np.array(epochs), np.array(ocs), np.array(weights)
    def reset_app(self):
        try:
            self.df = None
            self.filename = None
            self.last_par_params = [0, 0, 0]
            self.last_sin_params = [0, 1000, 0, 0]
            self.ml_data = {} 
            self.input_t0.clear()
            self.input_p.clear()
            if hasattr(self, 'toolbar'):
                self.toolbar.act_par.setChecked(True)
                self.toolbar.act_sin.setChecked(True)
            self.text_result.clear()
            self.canvas.ax_top.clear()
            self.canvas.ax_bottom.clear()
            self.canvas.ax_top.grid(True, linestyle=':', alpha=0.3)
            self.canvas.ax_bottom.grid(True, linestyle=':', alpha=0.3)
            self.canvas.draw()
            self.text_result.append(">> System Reset. Ready.")
        except Exception as e:
            QMessageBox.warning(self, "Reset Error", f"Hata: {e}")
    def open_manual_dialog(self):
        if self.df is None: return
        try: p_val = float(self.input_p.text())
        except: QMessageBox.warning(self, "Error", "Set Period first"); return
        dlg = ParameterDialog(self.last_par_params, self.last_sin_params, p_val, self)
        if dlg.exec():
            self.last_par_params = dlg.new_par
            self.last_sin_params = dlg.new_sin
            self.run_manual_plot()
    def run_manual_plot(self):
        try:
            t0 = float(self.input_t0.text())
            p = float(self.input_p.text())
        except: return      
        c_min1 = 'green'
        c_min2 = 'yellow'
        c_par  = 'red'
        c_sin  = 'black'
        c_comb = '#aa5500ff'
        ep, oc, _ = self.get_epoch_oc(t0, p)
        x_line = np.linspace(min(ep)-500, max(ep)+500, 1000)
        show_par, show_sin, show_comb = True, True, True
        if hasattr(self, 'toolbar'):
            if hasattr(self.toolbar, 'act_par'): show_par = self.toolbar.act_par.isChecked()
            if hasattr(self.toolbar, 'act_sin'): show_sin = self.toolbar.act_sin.isChecked()
            if hasattr(self.toolbar, 'act_comb'): show_comb = self.toolbar.act_comb.isChecked() 
        self.canvas.ax_top.clear(); self.canvas.ax_bottom.clear()
        mask_sec = (ep % 1 != 0)
        self.canvas.ax_top.scatter(ep[~mask_sec], oc[~mask_sec], c=c_min1, marker='o', label='Min I')
        self.canvas.ax_top.scatter(ep[mask_sec], oc[mask_sec], c=c_min2, marker='s', label='Min II')
        rms_par, rms_sin, rms_comb = None, None, None
        best_res = np.zeros_like(oc)
        label_res = "Residuals"
        min_rms = 9999.0
        if show_par:
            y_para = self.parabola(x_line, *self.last_par_params)
            self.canvas.ax_top.plot(x_line, y_para, color=c_par, linestyle='--', linewidth=1.5, label='Parabola')  
            res_p = oc - self.parabola(ep, *self.last_par_params)
            rms_par = np.sqrt(np.mean(res_p**2))
            if rms_par < min_rms:
                min_rms = rms_par
                best_res = res_p
                label_res = "Res(Par)"
        if show_sin:
            y_sine = self.sine_wave(x_line, *self.last_sin_params)
            self.canvas.ax_top.plot(x_line, y_sine, color=c_sin, linestyle='-', linewidth=1.5, label='Sinusoid')
            
            res_s = oc - self.sine_wave(ep, *self.last_sin_params)
            rms_sin = np.sqrt(np.mean(res_s**2))            
            if rms_sin < min_rms:
                min_rms = rms_sin
                best_res = res_s
                label_res = "Res(Sin)"
        if show_comb and self.last_comb_params is not None:
            y_comb = self.combined_model(x_line, *self.last_comb_params)
            self.canvas.ax_top.plot(x_line, y_comb, color=c_comb, linestyle='-', linewidth=2.5, label='Combined')            
            res_c = oc - self.combined_model(ep, *self.last_comb_params)
            rms_comb = np.sqrt(np.mean(res_c**2))            
            if rms_comb < min_rms:
                min_rms = rms_comb
                best_res = res_c
                label_res = "Res(Comb)"
        self.ml_data["RMS"] = min_rms
        self.canvas.ax_top.legend(loc='best', frameon=True, fancybox=True, framealpha=0.8)
        self.canvas.ax_top.grid(True, alpha=0.3, linestyle=':')
        self.canvas.ax_top.set_ylabel("O-C (d)")
        self.canvas.ax_bottom.scatter(ep, best_res, c='black', s=10, alpha=0.7)
        self.canvas.ax_bottom.axhline(0, c='gray', ls=':')
        self.canvas.ax_bottom.grid(True, alpha=0.3, linestyle=':')
        self.canvas.ax_bottom.set_xlabel("Epoch")
        self.canvas.ax_bottom.set_ylabel(f"{label_res}\nRMS:{min_rms:.5f}")
        self.canvas.draw()
        info_str = "\n[Fit Quality - RMS Comparison]"
        if rms_par is not None: info_str += f"\n   Parabola : {rms_par:.6f}"
        if rms_sin is not None: info_str += f"\n   Sinusoid : {rms_sin:.6f}"
        if rms_comb is not None: info_str += f"\n   Combined : {rms_comb:.6f}" 
        self.text_result.append(info_str)
    def run_visual_comparison(self):
        if self.df is None: return
        self.calculate_parameters(show_physics=False) 
    def calculate_parameters(self, show_physics=False):
        if self.df is None:
            QMessageBox.warning(self, "Warning", "Please load CSV first.")
            return        
        try:
            t0_val = float(self.input_t0.text())
            p_val = float(self.input_p.text())
        except ValueError:
            QMessageBox.warning(self, "Warning", "Check T0 and P values.")
            return        
        ep, oc, w = self.get_epoch_oc(t0_val, p_val) 
        idx = np.argsort(ep)
        epochs_sorted = ep[idx]
        ocs_sorted = oc[idx]
        w_s = w[idx]
        self.ml_data = {
            "T0_input": t0_val, "P_input": p_val,
            "Q_val": 0, "Q_err": 0, "P_dot": 0, "P_dot_err": 0,
            "A_val": 0, "A_err": 0, "P3_yr": 0, "P3_err": 0, "RMS": 0,
            "Filename": os.path.basename(self.filename) if self.filename else "Unknown"
        }
        self.text_result.append("\n" + "="*40)
        self.text_result.append("   ASTROPHYSICAL PARAMETERS CALCULATION")
        self.text_result.append("="*40)        
        if hasattr(self, 'toolbar'):
            show_par = self.toolbar.act_par.isChecked()
            show_sin = self.toolbar.act_sin.isChecked()
        else: show_par, show_sin = True, True
        if show_par:
            try:            
                p_par, pcov_par = curve_fit(self.parabola, epochs_sorted, ocs_sorted, sigma=w_s, absolute_sigma=False)
                self.last_par_params = p_par
                c2, c1, c0 = p_par
                err_par = np.sqrt(np.diag(pcov_par))
                p_dot = 2 * c2
                p_dot_sec_yr = p_dot * (365.25 / p_val) * 86400
                p_dot_err = 2 * err_par[0] * (365.25 / p_val) * 86400
                self.ml_data["Q_val"] = c2; self.ml_data["Q_err"] = err_par[0]
                self.ml_data["P_dot"] = p_dot_sec_yr; self.ml_data["P_dot_err"] = p_dot_err
                self.text_result.append(">> MODEL A: PARABOLIC")
                self.text_result.append(f"Q (c2): {c2:.3e} ± {err_par[0]:.3e}")
                self.text_result.append(f"P_dot : {p_dot_sec_yr:.4f} ± {p_dot_err:.4f} sec/year")
                self.text_result.append("-" * 30)
            except:
                self.text_result.append("Parabolic Fit Failed")        
        if show_sin:
            try:
                data_span = max(epochs_sorted) - min(epochs_sorted)
                amp_guess = (max(ocs_sorted) - min(ocs_sorted)) / 2
                guess_periods = [data_span * 0.8, data_span * 1.5, data_span * 2.5, 5000]            
                best_rss = float('inf'); best_params = [0, 1, 0, 0]; best_cov = None
                for p_guess in guess_periods:
                    try:
                        p0 = [amp_guess, p_guess, 0, 0]
                        t_params, t_cov = curve_fit(self.sine_wave, epochs_sorted, ocs_sorted, 
                                                    sigma=w_s, p0=p0, maxfev=5000,
                                                    bounds=([0, 100, -np.pi, -np.inf], [np.inf, data_span*5, np.pi, np.inf]))                    
                        res = ocs_sorted - self.sine_wave(epochs_sorted, *t_params)
                        rss = np.sum(res**2)
                        if rss < best_rss:
                            best_rss = rss; best_params = t_params; best_cov = t_cov
                    except: continue            
                A, P_mod_cyc, phi, off = best_params            
                self.last_sin_params = best_params
                if best_cov is not None:
                    err_sin = np.sqrt(np.diag(best_cov))
                    A_err = err_sin[0]; P_mod_err = err_sin[1]
                else: A_err = 0.0; P_mod_err = 0.0
                P_mod_year = (P_mod_cyc * p_val) / 365.25
                P_mod_year_err = (P_mod_err * p_val) / 365.25
                self.ml_data["Sin_A"] = A; self.ml_data["Sin_A_err"] = A_err
                self.ml_data["Sin_P3_yr"] = P_mod_year; self.ml_data["Sin_P3_err"] = P_mod_year_err 
                self.text_result.append(">> MODEL B: SINUSOIDAL")
                self.text_result.append(f"Amp (A) : {A:.5f} ± {A_err:.5f} d")
                self.text_result.append(f"Per (P) : {P_mod_year:.2f} ± {P_mod_year_err:.2f} yr")
                self.text_result.append(f"Phase   : {phi:.4f} rad")
                self.text_result.append(f"Offset  : {off:.5f} d")
                self.text_result.append("="*40)
            except Exception as e:
                self.text_result.append(f"Sinusoidal Fit Failed: {e}")
        try:
            if len(self.last_par_params) > 0 and len(self.last_sin_params) > 0:
                p0_comb = [
                    self.last_par_params[0], self.last_par_params[1], self.last_par_params[2], 
                    self.last_sin_params[0], self.last_sin_params[1], self.last_sin_params[2]
                ]
                p_comb, pcov_comb = curve_fit(self.combined_model, epochs_sorted, ocs_sorted, sigma=w_s, p0=p0_comb, maxfev=10000)
                self.last_comb_params = p_comb
                cc2, cc1, cc0, cA, cP_cyc, cPhi = p_comb
                perr = np.sqrt(np.diag(pcov_comb))
                cP_yr = (cP_cyc * p_val) / 365.25
                cP_yr_err = (perr[4] * p_val) / 365.25
                self.text_result.append("-" * 30)
                self.text_result.append(">> COMBINED MODEL :")
                self.text_result.append(f"   Q (c2) : {cc2:.3e} ± {perr[0]:.3e}")
                self.text_result.append(f"   Amp (A): {cA:.5f} ± {perr[3]:.5f} d")
                self.text_result.append(f"   Per (P): {cP_yr:.2f} ± {cP_yr_err:.2f} yr")
                self.ml_data.update({
                    "Comb_c2": cc2,      "Comb_c2_err": perr[0],
                    "Comb_A": cA,        "Comb_A_err": perr[3],
                    "Comb_P3_yr": cP_yr, "Comb_P3_err": cP_yr_err
                })
        except Exception as e:
            self.text_result.append(f"Combined Fit Failed: {e}")
        self.run_manual_plot() 
        if show_physics and (len(self.last_par_params) > 0 or len(self.last_sin_params) > 0 or self.last_comb_params is not None):    
            mass_dlg = MassInputDialog(self)
            if mass_dlg.exec():
                m1, m2 = mass_dlg.get_masses()   
                if m1 is not None and m2 is not None:
                    self.text_result.append("=" * 40)
                    self.text_result.append(f"PHYSICAL ANALYSIS (Errors Included) (M1={m1}, M2={m2})")
                    if len(self.last_par_params) > 0:
                        self.text_result.append("-" * 30)
                        self.text_result.append(">> [1] PARABOLIC FIT ONLY:")
                        try:
                            c2_par = self.ml_data.get("Q_val", 0)
                            c2_err = self.ml_data.get("Q_err", 0)   
                            p_dot = 2 * c2_par * (365.25 / p_val) * 86400
                            rate = p_dot / (p_val * 86400)   
                            if m1 != m2:
                                dM = (m1 * m2 / (3 * (m1 - m2))) * rate
                                dM_err = abs(dM * (c2_err / c2_par)) if c2_par != 0 else 0
                                self.text_result.append(f"   Mass Transfer: {dM:.3e} ± {dM_err:.3e} M_sun/yr")
                            else: 
                                self.text_result.append("   Mass Transfer: Invalid (M1=M2)")
                        except Exception as e:
                            self.text_result.append(f"   Calculation Error: {e}")
                    if len(self.last_sin_params) > 0:
                        self.text_result.append("-" * 30)
                        self.text_result.append(">> [2] SINUSOIDAL FIT ONLY:")
                        try:
                            A_sin = self.ml_data.get("Sin_A", 0)
                            A_err = self.ml_data.get("Sin_A_err", 0)
                            P3_sin = self.ml_data.get("Sin_P3_yr", 1.0) 
                            P3_err = self.ml_data.get("Sin_P3_err", 0.0)
                            a12 = A_sin * 173.14
                            a12_err = A_err * 173.14
                            self.text_result.append(f"   a12*sin(i): {a12:.4f} ± {a12_err:.4f} AU")
                            if P3_sin > 1e-5: 
                                fm3 = (a12**3) / (P3_sin**2)
                                if a12 > 0:
                                    frac_err = np.sqrt( (3*(a12_err/a12))**2 + (2*(P3_err/P3_sin))**2 )
                                    fm3_err = fm3 * frac_err
                                else: fm3_err = 0
                                self.text_result.append(f"   f(m3): {fm3:.5e} ± {fm3_err:.5e} M_sun")
                                self.text_result.append("   3rd Body Estimates (Sin Only):")
                                M_bin = m1 + m2
                                masses = np.linspace(0.001, 5.0, 50000)
                                for inc in [90, 60, 30]:
                                    sin_i = np.sin(np.deg2rad(inc))
                                    diffs = np.abs((masses*sin_i)**3 - fm3*(M_bin+masses)**2)
                                    best_m3 = masses[np.argmin(diffs)]
                                    self.text_result.append(f"     i={inc}°: {best_m3:.3f} M_sun")
                            else:
                                self.text_result.append("   Error: Period too small or 0.")

                        except Exception as e:
                            self.text_result.append(f"   Calculation Error: {e}")
                    if self.last_comb_params is not None:
                        self.text_result.append("-" * 30)
                        self.text_result.append(">> [3] COMBINED FIT (BEST):")
                        c2 = self.ml_data.get("Comb_c2", 0); c2_err = self.ml_data.get("Comb_c2_err", 0)
                        A  = self.ml_data.get("Comb_A", 0);  A_err  = self.ml_data.get("Comb_A_err", 0)
                        P3 = self.ml_data.get("Comb_P3_yr", 1); P3_err = self.ml_data.get("Comb_P3_err", 0)
                        try:
                            p_dot = 2 * c2 * (365.25 / p_val) * 86400
                            rate = p_dot / (p_val * 86400)
                            if m1 != m2:
                                dM = (m1 * m2 / (3 * (m1 - m2))) * rate
                                dM_err = abs(dM * (c2_err / c2)) if abs(c2) > 1e-20 else 0  
                                self.text_result.append(f"   Mass Transfer: {dM:.3e} ± {dM_err:.3e} M_sun/yr")
                                self.ml_data["dM_dt"] = dM
                            else:
                                self.text_result.append("   Mass Transfer: Invalid (M1=M2)")
                        except Exception as e:
                            self.text_result.append(f"   Mass Transfer Error: {e}")
                        try:
                            a12 = A * 173.14
                            a12_err = A_err * 173.14
                            self.text_result.append(f"   a12*sin(i): {a12:.4f} ± {a12_err:.4f} AU")
                            
                            if P3 > 1e-5: 
                                fm3 = (a12**3) / (P3**2)
                                if a12 > 1e-9:
                                    frac_err = np.sqrt((3*(a12_err/a12))**2 + (2*(P3_err/P3))**2)
                                    fm3_err = fm3 * frac_err
                                else: fm3_err = 0                                
                                self.text_result.append(f"   f(m3): {fm3:.5e} ± {fm3_err:.5e} M_sun")
                                self.ml_data["f_m3"] = fm3                                
                                self.text_result.append("   3rd Body Estimates (Detailed):")
                                M_bin = m1 + m2
                                masses = np.linspace(0.001, 5.0, 50000)
                                for inc in [90, 60, 30]:
                                    sin_i = np.sin(np.deg2rad(inc))
                                    diffs = np.abs((masses*sin_i)**3 - fm3*(M_bin+masses)**2)
                                    best_m3 = masses[np.argmin(diffs)]
                                    m3_jup = best_m3 * 1047.57
                                    o_type = "Planet" if m3_jup < 13 else "Brown Dwarf" if m3_jup < 80 else "Red Dwarf/Star"
                                    self.text_result.append(f"     i={inc}°: {best_m3:.3f} M_sun [{o_type}]")
                                    self.ml_data[f"M3_i{inc}"] = best_m3
                            else:
                                self.text_result.append("   f(m3): Period is too small/zero.")

                        except Exception as e:
                            self.text_result.append(f"   3rd Body Error: {e}")
                    self.text_result.append("=" * 40)
    def run_analysis(self):
        if self.df is None: return      
        try:
            t0 = float(self.input_t0.text())
            p = float(self.input_p.text())
        except: return          
        ep, oc, w = self.get_epoch_oc(t0, p)
        def lin(x, dp, off): return dp*x + off
        try:
            popt, pcov = curve_fit(lin, ep, oc, sigma=1.0/w, absolute_sigma=False)    
            dP_fit, offset_fit = popt    
            perr = np.sqrt(np.diag(pcov))
            dP_err, offset_err = perr       
            new_t0 = t0 + offset_fit
            new_p = p + dP_fit      
            predicted = lin(ep, *popt)
            residuals = oc - predicted
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((oc - np.mean(oc))**2)
            r_squared = 1 - (ss_res / ss_tot)
            res_str = f"""
=== LINEAR FIT RESULTS ===
R^2 Score       : {r_squared:.5f}
--------------------------
Offset (dT0)    : {offset_fit:.6f} ± {offset_err:.6f} d
dP (Period adj) : {dP_fit:.9f} ± {dP_err:.9f} d/E
--------------------------
NEW ELEMENTS:
Min I (BJD) = {new_t0:.6f} + {new_p:.9f} x E
--------------------------
"""
            self.text_result.append(res_str)
            #Draw Graph
            self.canvas.ax_top.clear(); self.canvas.ax_bottom.clear()
            self.canvas.ax_top.scatter(ep, oc, c='k', label='Data')
            self.canvas.ax_top.plot(ep, lin(ep, *popt), 'b--', label=f'Linear Fit (R²={r_squared:.3f})')
            self.canvas.ax_top.legend()
            self.canvas.ax_top.grid(True, alpha=0.3)
            self.canvas.ax_bottom.scatter(ep, residuals, c='k', s=10)
            self.canvas.ax_bottom.axhline(0, c='gray', ls=':')
            self.canvas.ax_bottom.set_xlabel("Epoch")
            self.canvas.ax_bottom.set_ylabel("Residuals")
            self.canvas.ax_bottom.grid(True, alpha=0.3)   
            self.canvas.draw()   
        except Exception as e:
            self.text_result.append(f"Linear Fit Error: {e}")
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MIST_OC_App()
    window.show()
    sys.exit(app.exec())