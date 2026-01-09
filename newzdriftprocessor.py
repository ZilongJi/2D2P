# newzdriftprocessor.py
#
# Contains the NewZdriftProcessor class for automated, continuous Z-drift analysis
# using a single, trigger-synchronized master RElog file.
# UPDATED: Now includes robust parsing for the specific RElog format and improved
# sequential trigger-to-TIFF matching.

import os
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d, gaussian_filter
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import tkinter as tk
from tkinter import filedialog, ttk
import time
import threading

from PIL import Image, ImageTk

import suite2p
from scanimagetiffio import SITiffIO
from utils_image import UnrotateCropFrame, RegFrame, compute_zpos_sp, findFOV

class NewZdriftProcessor(tk.Frame):
    def __init__(self, master=None, folder=None, app=None):
        super().__init__(master)
        self.master = master
        self.folder_to_watch = folder
        self.app = app
        self.grid()
        
        self.numFrames = tk.StringVar(value="500")

        self.monitoring_active = False
        self.monitoring_thread = None
        self.processed_files = set()
        self.master_relog_path = None
        self.master_relog_data = None # Will hold the parsed pandas DataFrame
        self.master_relog_trigger_times = []
        
        ### UPDATED/CORRECTED ### Keeps track of the last used trigger to ensure sequential matching
        self.last_used_trigger_index = -1

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
        
        tk.Button(self, text="Select Master RElog File", command=self.select_master_relog).grid(row=6, column=0, columnspan=2)
        self.master_relog_label = tk.Label(self, text="No RElog file selected", fg="red")
        self.master_relog_label.grid(row=7, column=0, columnspan=2)
        
        self.start_button = tk.Button(self, text="Start Monitoring", command=self.start_monitoring)
        self.start_button.grid(row=8, column=0)
        
        self.stop_button = tk.Button(self, text="Stop Monitoring", command=self.stop_monitoring, state=tk.DISABLED)
        self.stop_button.grid(row=8, column=1)

    def create_canvas_reg(self):
        self.canvas_reg = tk.Canvas(self, width=512, height=512, bg="#4D4D4D")
        self.canvas_reg.grid(row=9, column=0, columnspan=2)
        
    def create_canvas_corr(self):
        self.canvas_corr = tk.Canvas(self, width=512, height=256, bg="white")
        self.canvas_corr.grid(row=10, column=0, columnspan=2)

    def create_progress_bar(self):
        self.progress_label = tk.Label(self, text="Status: Idle")
        self.progress_label.grid(row=11, column=0, columnspan=2, sticky="w", padx=5)
        self.progress_bar = ttk.Progressbar(self, orient="horizontal", length=500, mode="determinate")
        self.progress_bar.grid(row=12, column=0, columnspan=2, pady=5, padx=5)

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

    def select_master_relog(self):
        path = filedialog.askopenfilename(
            initialdir=self.folder_to_watch, title="Select the MASTER RElog file for the session", filetypes=[("txt files", "*.txt")]
        )
        if path:
            self.master_relog_path = path
            self.master_relog_label.config(text=os.path.basename(path), fg="green")
            self.app.log_message(f"Master RElog file selected: {path}")

    def manual_correlation_analysis(self):
        if not hasattr(self, 'tifffilename') or not hasattr(self, 'relogfilename'):
            self.app.log_message("Error: Please select a Tiff and RElog file for manual analysis.")
            return
        threading.Thread(target=self.run_full_analysis, args=(self.tifffilename, self.relogfilename), daemon=True).start()

    def start_monitoring(self):
        if not os.path.exists(os.path.join(self.DPFolder, 'meanstacks.npy')):
            self.app.log_message("Error: Calibration data (DP/meanstacks.npy) not found.")
            return
        if not self.master_relog_path:
            self.app.log_message("Error: Please select the Master RElog file before starting.")
            return

        ### UPDATED/CORRECTED ### New, robust parsing logic for the specific RElog format.
        try:
            self.app.log_message("Parsing Master RElog for trigger events...")
            
            # Use a regular expression to split the complex line format
            # This splits on one or more spaces, OR the '=' sign surrounded by optional spaces.
            df = pd.read_csv(self.master_relog_path, sep=r'\s*=\s*|\s+', header=None, engine='python')
            
            # Combine the first two columns (date and time) into a single datetime object
            timestamps = pd.to_datetime(df[0] + ' ' + df[1])
            
            # Extract the angle and trigger values, which are in the 4th and 6th columns
            angles = df[3].astype(float)
            triggers = df[5].astype(float)
            
            # Create a clean, final DataFrame
            self.master_relog_data = pd.DataFrame({
                'timestamp': timestamps,
                'angle': angles,
                'trigger': triggers
            })
            
            # Find the start of each trigger block (where trigger goes from 0 to 1)
            trigger_series = self.master_relog_data['trigger']
            start_indices = np.where((trigger_series.iloc[:-1].values == 0) & (trigger_series.iloc[1:].values == 1.0))[0] + 1
            
            if len(start_indices) == 0:
                raise ValueError("No 'Trigger=1.0' start events (0->1 transition) found.")

            # Store the datetime objects of these trigger starts
            self.master_relog_trigger_times = self.master_relog_data['timestamp'].iloc[start_indices].tolist()
            
            self.app.log_message(f"Successfully parsed RElog. Found {len(self.master_relog_trigger_times)} trigger events.")
            
        except Exception as e:
            self.app.log_message(f"FATAL: Error parsing master RElog file: {e}")
            self.app.log_message("Please check the RElog file format. Stopping.")
            return

        # Reset state for a new monitoring session
        self.monitoring_active = True
        self.processed_files = set(f for f in os.listdir(self.folder_to_watch) if f.endswith('.tif'))
        self.last_used_trigger_index = -1
        
        self.monitoring_thread = threading.Thread(target=self._monitor_folder_loop, daemon=True)
        self.monitoring_thread.start()
        
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.app.log_message(f"Monitoring started for folder: {self.folder_to_watch}")
        self._update_progress(0, "Status: Monitoring for new files...")

    def stop_monitoring(self):
        self.monitoring_active = False
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.app.log_message("Monitoring stopped.")
        self._update_progress(0, "Status: Idle")

    def _monitor_folder_loop(self):
        while self.monitoring_active:
            try:
                all_tiffs = {f for f in os.listdir(self.folder_to_watch) if f.endswith('.tif')}
                new_tiffs = sorted(list(all_tiffs - self.processed_files)) # Sort to process in order

                for tiff_name in new_tiffs:
                    tiff_path = os.path.join(self.folder_to_watch, tiff_name)
                    if self._is_file_complete(tiff_path):
                        self.app.log_message(f"File {tiff_name} is complete. Starting analysis.")
                        self.run_full_analysis(tiff_path)
                        self.processed_files.add(tiff_name)
                
                
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

    ### UPDATED/CORRECTED ### Improved synchronization using a sequential trigger index.
    def _synchronize_with_master_relog(self, tiff_sio_obj):
        """Finds the next available trigger and sets the corresponding data segment."""
        next_trigger_index = self.last_used_trigger_index + 1
        
        if next_trigger_index >= len(self.master_relog_trigger_times):
            self.app.log_message(f"Warning: No more trigger events available in RElog file. Cannot process new TIFFs.")
            return False
        
        # Get the start time for the current data segment
        segment_start_time = self.master_relog_trigger_times[next_trigger_index]
        self.app.log_message(f"Matching TIFF to trigger event #{next_trigger_index + 1} at: {segment_start_time.strftime('%Y-%m-%d %H:%M:%S.%f')}")

        # Find the end of the segment (the start of the *next* trigger, or the end of the file)
        if next_trigger_index + 1 < len(self.master_relog_trigger_times):
            segment_end_time = self.master_relog_trigger_times[next_trigger_index + 1]
        else:
            segment_end_time = self.master_relog_data['timestamp'].iloc[-1]
        
        # Extract the data segment using pandas boolean indexing
        segment_df = self.master_relog_data[
            (self.master_relog_data['timestamp'] >= segment_start_time) & 
            (self.master_relog_data['timestamp'] < segment_end_time)
        ]
        
        # Inject the synchronized data into the SITiffIO object
        # NOTE: SITiffIO expects columns named 'timestamp' and 'angle'
        tiff_sio_obj.rotary_data = segment_df[['timestamp', 'angle']].copy()
        
        # Crucially, update the index so the next file uses the next trigger
        self.last_used_trigger_index = next_trigger_index
        
        return True

    def run_full_analysis(self, tiff_path, relog_path=None):
        self._update_progress(5, f"Processing {os.path.basename(tiff_path)}...")
        if self.app: self.app.log_message("Loading and synchronizing data...")
        circlecenterfilename = os.path.join(self.DPFolder, "circlecenter.txt")
        with open(circlecenterfilename, "r") as f:
            self.rotx, self.roty = map(float, f.readlines()[-1].split())
        
        S = SITiffIO()
        S.open_tiff_file(tiff_path, "r")
        
        if self.monitoring_active and not relog_path:
            if not self._synchronize_with_master_relog(S):
                self._update_progress(0, "Status: Synchronization failed.")
                return
        elif relog_path:
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
