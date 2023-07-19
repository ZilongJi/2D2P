import os
import tkinter as tk
from centerdetector import CenterDetector
from stackprocessor import StackProcessor
from zdriftprocessor import ZdriftProcessor

class TwoDTWoP(tk.Tk):
    def __init__(self, folder):
        super().__init__()
        self.folder = folder

        # create a folder called APP under self.folder to save results 
        # if the folder already exists, delete it and create a new one
        self.appfolder = self.folder + "/APP"
        if os.path.exists(self.appfolder):
            os.system("rm -rf " + self.appfolder)
        os.mkdir(self.appfolder)

        self.create_widgets()
        
        # create a log window
        self.create_log_window()

    def create_widgets(self):
        self.center_processor = CenterDetector(self, self.folder, self.appfolder, app=self)
        self.stack_processor = StackProcessor(self, self.folder, self.appfolder, app=self)
        self.zdrift_processor = ZdriftProcessor(self, self.folder, self.appfolder, app=self)

        # Place the three classes into the main window using the grid manager
        self.center_processor.grid(row=0, column=0)
        self.stack_processor.grid(row=0, column=1)
        self.zdrift_processor.grid(row=0, column=2, rowspan=2)

    def create_log_window(self):
        # create a text widget to display all the logs
        self.log_text = tk.Text(self, height=30, width=150)
        self.log_text.grid(row=1, column=0, columnspan=2)
        self.log_text.configure(state="disabled")
        
    def log_message(self, message):
        self.log_text.configure(state="normal")
        self.log_text.insert("end", message + "\n")
        # Add a line to separate different logs
        self.log_text.insert("end", "-" * 20 + "\n")
        # Disable the text widget so that the user cannot change the logs
        self.log_text.configure(state="disabled")
        # Scroll the text widget to the end
        self.log_text.see("end")


# Create an instance of the merged class and run the application
folder_path = (
    "/home/zilong/Desktop/2D2P/Data"
)
app = TwoDTWoP(folder_path)
app.title("2D2P")
app.mainloop()
