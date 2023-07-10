import os
import tkinter as tk
from centerdetector import CenterDetector
from stackprocessor import StackProcessor
from zdriftprocessor import ZdriftProcessor


class App(tk.Tk):
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

    def create_widgets(self):
        self.center_processor = CenterDetector(self, self.folder, self.appfolder)
        self.stack_processor = StackProcessor(self, self.folder, self.appfolder)
        self.zdrift_processor = ZdriftProcessor(self, self.folder, self.appfolder)

        # #pack the three classes into the main window from left to right
        # self.tiff_processor.pack(side="left")
        # self.stack_processor.pack(side="left")
        # self.zdrift_processor.pack(side="left")

        # Place the three classes into the main window using the grid manager
        self.center_processor.grid(row=0, column=0)
        self.stack_processor.grid(row=0, column=1)
        self.zdrift_processor.grid(row=0, column=2)

# Create an instance of the merged class and run the application
folder_path = (
    "/home/zilong/Desktop/2D2P/Data/162_10072023"
)
app = App(folder_path)
app.title("2D2P")
app.mainloop()
