import os
import tkinter as tk
from centerdetector import CenterDetector
from stackprocessor import StackProcessor
### RENAMED ### Import the new class from the new file
from newzdriftprocessor import NewZdriftProcessor

class TwoDTWoP(tk.Tk):
    def __init__(self, folder):
        super().__init__()
        self.folder = folder

        self.grid_columnconfigure(0, weight=3)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.create_widgets()
        self.create_log_window()
        
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

    def create_widgets(self):
        processor_panel = tk.Frame(self, borderwidth=0)
        processor_panel.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        for col in range(3):
            processor_panel.columnconfigure(col, weight=1)
        processor_panel.rowconfigure(0, weight=1)

        center_frame = tk.LabelFrame(processor_panel, text="Center Detector", padx=5, pady=5)
        center_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        stack_frame = tk.LabelFrame(processor_panel, text="Stack Processor", padx=5, pady=5)
        stack_frame.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")
        zdrift_frame = tk.LabelFrame(processor_panel, text="Z-Drift Monitor", padx=5, pady=5)
        zdrift_frame.grid(row=0, column=2, padx=5, pady=5, sticky="nsew")

        self.center_processor = CenterDetector(center_frame, self.folder, app=self)
        self.center_processor.pack(fill="both", expand=True)
        self.stack_processor = StackProcessor(stack_frame, self.folder, app=self)
        self.stack_processor.pack(fill="both", expand=True)
        
        ### RENAMED ### Instantiate the new class
        self.zdrift_processor = NewZdriftProcessor(zdrift_frame, self.folder, app=self)
        self.zdrift_processor.pack(fill="both", expand=True)

    def create_log_window(self):
        log_frame = tk.LabelFrame(self, text="Log Output", padx=5, pady=5)
        log_frame.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        
        scrollbar = tk.Scrollbar(log_frame)
        scrollbar.grid(row=0, column=1, sticky="ns")
        
        self.log_text = tk.Text(log_frame, height=50, width=48, yscrollcommand=scrollbar.set)
        self.log_text.grid(row=0, column=0, sticky="nsew")
        self.log_text.configure(state="disabled")
        
        scrollbar.config(command=self.log_text.yview)
        
    def log_message(self, message):
        self.log_text.configure(state="normal")
        self.log_text.insert("end", message + "\n")
        self.log_text.insert("end", "-" * 20 + "\n")
        self.log_text.configure(state="disabled")
        self.log_text.see("end")

    def on_closing(self):
        if self.zdrift_processor.monitoring_active:
            self.zdrift_processor.stop_monitoring()
        self.destroy()

# Create an instance of the merged class and run the application
folder_path = (
    "/home/zilong/Desktop/2D2P/Data"
)
app = TwoDTWoP(folder_path)
app.title("2D2P Automated Z-Drift Corrector")
app.mainloop()
