# newzdriftprocessor.py
#
# Contains the NewZdriftProcessor class for automated, continuous Z-drift analysis
# using paired TIFF and RElog files detected in watch folders.

import os
import numpy as np
from scipy.ndimage import gaussian_filter1d, gaussian_filter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import tkinter as tk
from tkinter import filedialog, ttk
import time
import threading
from collections import deque

from PIL import Image, ImageTk

import suite2p
from scanimagetiffio import SITiffIO
from utils_image import UnrotateCropFrame, RegFrame, compute_zpos_sp, findFOV

class NewZdriftProcessor(tk.Frame):
    def __init__(self, master=None, folder=None, app=None):
        super().__init__(master)
        self.master = master
        self.folder_to_watch = folder
        self.relog_folder_to_watch = folder
        self.app = app
        self.grid()
        
        self.numFrames = tk.StringVar(value="500")

        self.monitoring_active = False
        self.monitoring_thread = None
        self.processed_files = set()
        self.processed_relog_files = set()
        self.pending_tiffs = deque()
        self.pending_relogs = deque()
        self.pending_lock = threading.Lock()
        self.analysis_running = False
        self._monitoring_poll_job = None

        self.create_widgets()
        self.create_canvas_reg()      
        self.create_canvas_corr()
        self.create_progress_bar()

    # --- GUI setup functions (create_widgets, canvases, progress_bar) are unchanged ---
    def create_widgets(self):
        # --- Manual Setup ---
        tk.Label(self, text="--- Manual Setup & Analysis ---").grid(row=0, column=0, columnspan=2, pady=(0, 5))
        tk.Button(self, text="Select Tiff (sets folder)", command=self.import_tiff).grid(row=1, column=0)
        tk.Button(self, text="Select RElog (for manual)", command=self.import_RElog).grid(row=2, column=0)
        
        tk.Label(self, text="numFrames").grid(row=3, column=0)
        tk.Entry(self, textvariable=self.numFrames).grid(row=4, column=0)
        
        tk.Button(self, text="Manual Correlation Analysis", command=self.manual_correlation_analysis).grid(row=1, column=1, rowspan=4)

        # --- Automatic Monitoring ---
        tk.Label(self, text="--- Automatic Monitoring ---").grid(row=5, column=0, columnspan=2, pady=(10, 5))
        
        tk.Button(self, text="Select TIFF Directory", command=self.select_tiff_directory).grid(row=6, column=0, columnspan=2)
        self.tiff_dir_label = tk.Label(self, text="No TIFF directory selected", fg="red")
        self.tiff_dir_label.grid(row=7, column=0, columnspan=2)
        
        tk.Button(self, text="Select RElog Directory", command=self.select_relog_directory).grid(row=8, column=0, columnspan=2)
        self.relog_dir_label = tk.Label(self, text="No RElog directory selected", fg="red")
        self.relog_dir_label.grid(row=9, column=0, columnspan=2)
        
        self.start_button = tk.Button(self, text="Start Monitoring", command=self.start_monitoring)
        self.start_button.grid(row=10, column=0)
        
        self.stop_button = tk.Button(self, text="Stop Monitoring", command=self.stop_monitoring, state=tk.DISABLED)
        self.stop_button.grid(row=10, column=1)

    def create_canvas_reg(self):
        self.canvas_reg = tk.Canvas(self, width=512, height=512, bg="#4D4D4D")
        self.canvas_reg.grid(row=11, column=0, columnspan=2)
        
    def create_canvas_corr(self):
        self.canvas_corr = tk.Canvas(self, width=512, height=256, bg="white")
        self.canvas_corr.grid(row=12, column=0, columnspan=2)

    def create_progress_bar(self):
        self.progress_label = tk.Label(self, text="Status: Idle")
        self.progress_label.grid(row=13, column=0, columnspan=2, sticky="w", padx=5)
        self.progress_bar = ttk.Progressbar(self, orient="horizontal", length=500, mode="determinate")
        self.progress_bar.grid(row=14, column=0, columnspan=2, pady=5, padx=5)

    def import_tiff(self):
        self.tifffilename = filedialog.askopenfilename(
            initialdir=self.folder_to_watch, title="Select a tiff file to define folder", filetypes=[("tiff files", "*.tif")]
        )
        if not self.tifffilename: return
        self.folder_to_watch = os.path.dirname(self.tifffilename)
        self.DPFolder = os.path.join(self.folder_to_watch, "DP")
        if self.app: self.app.log_message(f"Set folder to watch: {self.folder_to_watch}")
        self.processed_files.clear()

    def import_RElog(self):
        self.relogfilename = filedialog.askopenfilename(
            initialdir=self.folder_to_watch, title="Select a RElog file", filetypes=[("txt files", "*.txt")]
        )
        if self.app and self.relogfilename:
            self.app.log_message(f"Imported manual RElog file: {self.relogfilename}")

    def select_tiff_directory(self):
        path = filedialog.askdirectory(initialdir=self.folder_to_watch, title="Select TIFF directory")
        if path:
            self.folder_to_watch = path
            self.DPFolder = os.path.join(self.folder_to_watch, "DP")
            self.tiff_dir_label.config(text=path, fg="green")
            if self.app:
                self.app.log_message(f"TIFF directory selected: {path}")

    def select_relog_directory(self):
        path = filedialog.askdirectory(initialdir=self.relog_folder_to_watch, title="Select RElog directory")
        if path:
            self.relog_folder_to_watch = path
            self.relog_dir_label.config(text=path, fg="green")
            if self.app:
                self.app.log_message(f"RElog directory selected: {path}")

    def manual_correlation_analysis(self):
        if not hasattr(self, 'tifffilename') or not hasattr(self, 'relogfilename'):
            self.app.log_message("Error: Please select a Tiff and RElog file for manual analysis.")
            return
        threading.Thread(target=self.run_full_analysis, args=(self.tifffilename, self.relogfilename), daemon=True).start()

    def start_monitoring(self):
        if not hasattr(self, "DPFolder"):
            self.app.log_message("Error: Please select the TIFF directory before starting.")
            return
        if not self.relog_folder_to_watch:
            self.app.log_message("Error: Please select the RElog directory before starting.")
            return
        if not os.path.exists(os.path.join(self.DPFolder, 'meanstacks.npy')):
            self.app.log_message("Error: Calibration data (DP/meanstacks.npy) not found.")
            return

        # Reset state for a new monitoring session
        self.monitoring_active = True
        self.processed_files = set(f for f in os.listdir(self.folder_to_watch) if f.endswith('.tif'))
        self.processed_relog_files = set(f for f in os.listdir(self.relog_folder_to_watch) if f.endswith('.txt'))
        self.pending_tiffs = deque()
        self.pending_relogs = deque()
        
        self.monitoring_thread = threading.Thread(target=self._monitor_folder_loop, daemon=True)
        self.monitoring_thread.start()
        self._start_polling_pending_pairs()
        
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.app.log_message(f"Monitoring started for folder: {self.folder_to_watch}")
        self._update_progress(0, "Status: Monitoring for new files...")

    def stop_monitoring(self):
        self.monitoring_active = False
        if self._monitoring_poll_job is not None:
            self.after_cancel(self._monitoring_poll_job)
            self._monitoring_poll_job = None
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.app.log_message("Monitoring stopped.")
        self._update_progress(0, "Status: Idle")

    def _monitor_folder_loop(self):
        while self.monitoring_active:
            try:
                all_tiffs = {f for f in os.listdir(self.folder_to_watch) if f.endswith('.tif')}
                all_relogs = {f for f in os.listdir(self.relog_folder_to_watch) if f.endswith('.txt')}
                new_tiffs = sorted(list(all_tiffs - self.processed_files))
                new_relogs = sorted(list(all_relogs - self.processed_relog_files))

                for tiff_name in new_tiffs:
                    tiff_path = os.path.join(self.folder_to_watch, tiff_name)
                    if self._is_file_complete(tiff_path):
                        with self.pending_lock:
                            self.pending_tiffs.append(tiff_path)
                        self.processed_files.add(tiff_name)
                        self.app.log_message(f"Detected new TIFF: {tiff_name}")

                for relog_name in new_relogs:
                    relog_path = os.path.join(self.relog_folder_to_watch, relog_name)
                    if self._is_file_complete(relog_path):
                        with self.pending_lock:
                            self.pending_relogs.append(relog_path)
                        self.processed_relog_files.add(relog_name)
                        self.app.log_message(f"Detected new RElog: {relog_name}")

                time.sleep(5)
            except Exception as e:
                self.app.log_message(f"An error occurred in monitoring thread: {e}")
                time.sleep(10)

    def _is_file_complete(self, filepath, delay=2):
        try:
            size1 = os.path.getsize(filepath)
            if size1 < 1024: return False
            time.sleep(delay)
            size2 = os.path.getsize(filepath)
            return size1 == size2
        except FileNotFoundError:
            return False

    def _update_progress(self, value, text):
        self.progress_bar['value'] = value
        self.progress_label['text'] = f"Status: {text}"
        self.master.update_idletasks()

    def _start_polling_pending_pairs(self):
        if self._monitoring_poll_job is None:
            self._monitoring_poll_job = self.after(500, self._poll_pending_pairs)

    def _poll_pending_pairs(self):
        if not self.monitoring_active:
            self._monitoring_poll_job = None
            return

        tiff_path = None
        relog_path = None
        if not self.analysis_running:
            with self.pending_lock:
                if self.pending_tiffs and self.pending_relogs:
                    tiff_path = self.pending_tiffs.popleft()
                    relog_path = self.pending_relogs.popleft()

            if tiff_path and relog_path:
                self.analysis_running = True
                try:
                    self.app.log_message(
                        f"Pairing TIFF {os.path.basename(tiff_path)} with RElog {os.path.basename(relog_path)}."
                    )
                    self.run_full_analysis(tiff_path, relog_path)
                finally:
                    self.analysis_running = False

        self._monitoring_poll_job = self.after(500, self._poll_pending_pairs)

    def run_full_analysis(self, tiff_path, relog_path=None):
        self._update_progress(5, f"Processing {os.path.basename(tiff_path)}...")
        if self.app: self.app.log_message("Loading and synchronizing data...")
        circlecenterfilename = os.path.join(self.DPFolder, "circlecenter.txt")
        with open(circlecenterfilename, "r") as f:
            self.rotx, self.roty = map(float, f.readlines()[-1].split())
        
        S = SITiffIO()
        S.open_tiff_file(tiff_path, "r")
        
        if relog_path:
            S.open_rotary_file(relog_path)
        else:
            self.app.log_message("Error: No RElog data available for analysis.")
            return

        self._update_progress(20, "Interpolating timestamps...")
        S.interp_times()
        
        tailArray, tailAng = S.tail(int(self.numFrames.get()))      
        self.unrotFrames = UnrotateCropFrame(tailArray, tailAng, rotCenter=[self.rotx, self.roty])
        
        self._update_progress(40, "Registering frames...")
        self.meanRegImg, self.regFrames = RegFrame(self.unrotFrames)
        self.display_regFrame()
        
        if self.app: self.app.log_message("Perform Correlation Analysis...")
        self._update_progress(60, "Correlating with Z-stack...")
        meanstacks = np.load(os.path.join(self.DPFolder, "meanstacks.npy"))
        
        ops = suite2p.default_ops()
        _, _, self.corrMatrix = compute_zpos_sp(meanstacks, self.regFrames, ops)
        self.corrMatrix = gaussian_filter1d(self.corrMatrix.copy(), 2, axis=0)
        
        self._update_progress(90, "Generating results plot...")
        self.display_corrMatrix(tiff_path)
        self._update_progress(100, "Analysis complete.")
        time.sleep(2)
        if self.monitoring_active:
             self._update_progress(0, "Status: Monitoring for new files...")
        else:
            self._update_progress(0, "Status: Idle")

    def display_regFrame(self):
        # (Unchanged)
        fig = plt.figure(figsize=(5.12, 5.12), dpi=100)  
        plt.imshow(self.meanRegImg, cmap='gray')
        plt.axis('off')
        reg_img_path = os.path.join(self.DPFolder, "meanReg.png")
        fig.savefig(reg_img_path)
        plt.close(fig)
        self.meanRegImg_tk = ImageTk.PhotoImage(Image.open(reg_img_path)) 
        self.canvas_reg.create_image(self.rotx, self.roty, anchor="center", image=self.meanRegImg_tk)       
        
    def display_corrMatrix(self, tiff_path):
        # (Unchanged)
        fig = plt.figure(figsize=(5.12, 2.56), dpi=100)
        nplanes, nframes = self.corrMatrix.shape
        gs = GridSpec(1, 2, width_ratios=[3, 1])

        ax1 = fig.add_subplot(gs[0, 0])  
        ax1.imshow(self.corrMatrix, aspect='auto', cmap='gray')
        ax1.set_xlabel('Frame Number')
        ax1.set_ylabel('Stack index')
        ax1.set_yticks(np.arange(0, nplanes, 5))    
        ax1.set_yticklabels(np.arange(0, nplanes, 5) - int(nplanes/2))
        ax1.axhline(y=nplanes/2, color='r', linestyle='-')

        ax2 = fig.add_subplot(gs[0, 1])
        sumCorrMatrix = np.sum(self.corrMatrix, axis=1)
        ax2.plot(sumCorrMatrix, np.arange(0, nplanes), color='grey')
        ax2.set_xlabel('Sum of cc')
        ax2.set_yticks(np.arange(0, nplanes, 5))
        ax2.set_yticklabels(np.arange(0, nplanes, 5) - int(nplanes/2))
        ax2.set_ylim(ax2.get_ylim()[::-1])
        ax2.axhline(y=nplanes/2, color='r', linestyle='-')
        maxIndex = np.argmax(sumCorrMatrix)
        ax2.plot(sumCorrMatrix[maxIndex], maxIndex, 'ro')
        shiftamount = maxIndex - int(nplanes/2)   
        ax2.text(0.5, 0.1, f'Dft= {shiftamount}', horizontalalignment='center', verticalalignment='center', transform=ax2.transAxes, color='r')

        plt.tight_layout()
        plt.subplots_adjust(wspace=0.2)
        
        base_name = os.path.splitext(os.path.basename(tiff_path))[0]
        corr_matrix_path = os.path.join(self.DPFolder, f"corrMatrix_{base_name}.png")
        fig.savefig(corr_matrix_path)
        plt.close(fig)
        
        self.corrMatrix_tk = ImageTk.PhotoImage(Image.open(corr_matrix_path)) 
        self.canvas_corr.create_image(0, 0, anchor="nw", image=self.corrMatrix_tk)

        if self.app:
            self.app.log_message(f"--- Analysis for: {os.path.basename(tiff_path)} ---")
            if shiftamount < 0:
                self.app.log_message(f"Recommendation: Move z-focus {-shiftamount} micrometers down")
            else:
                self.app.log_message(f"Recommendation: Move z-focus {shiftamount} micrometers up")
