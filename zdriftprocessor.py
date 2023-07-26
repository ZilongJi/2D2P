# I want to create a tkinter GUI class that will allow me to import a tiff file, a RElog file, and a ROIs file. and a VRlog file
# After import these files, I want to create a widget that allows me to unrotate each frame according to information in the log files
# and then crop the unrotated frames as in stackprocessor.py

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

import suite2p
from scanimagetiffio import SITiffIO
from utils_image import UnrotateCropFrame, RegFrame, compute_zpos_sp

class ZdriftProcessor(tk.Frame):
    def __init__(self, master=None, folder=None, app=None):
        super().__init__(master)
        self.master = master
        self.folder = folder
        self.app = app
        self.grid()
        
        # initialize the numer of frames for later correlation analysis
        self.numFrames = tk.StringVar()

        self.create_widgets()
  
        self.create_canvas_reg()      
        #self.create_canvas_shift()
        self.create_canvas_corr()
        
    def create_widgets(self):
        #create a button to import tiff file
        tk.Button(self, text="Import tiff", command=self.import_tiff).grid(row=0, column=0)
        
        #create a button to import RElog file
        tk.Button(self, text="Import RElog", command=self.import_RElog).grid(row=1, column=0)
        
        # create a label and entry to set the number of frames for correlation analysis
        tk.Label(self, text="numFrames").grid(row=2, column=0)
        tk.Entry(self, textvariable=self.numFrames).grid(row=3, column=0)
        
        #create a button to do correlation analysis
        tk.Button(self, text="Correlation Analysis", command=self.correlationanalysis).grid(row=0, column=1)
        
    def create_canvas_reg(self):
        # create a canvas to display registration results
        self.canvas_reg = tk.Canvas(self, width=512, height=512, bg="#4D4D4D")
        self.canvas_reg.grid(row=4, column=0, columnspan=2)
        
    def create_canvas_shift(self):
        # create a canvas to display rigid and non-rigid shift results
        self.canvas_shift = tk.Canvas(self, width=512, height=128, bg="white")  
        self.canvas_shift.grid(row=5, column=0, columnspan=2)
        
    def create_canvas_corr(self):
        # create a canvas to display results
        self.canvas_corr = tk.Canvas(self, width=512, height=256, bg="white")
        self.canvas_corr.grid(row=6, column=0, columnspan=2)
        
    def import_tiff(self):
        # import tiff file
        self.tifffilename = filedialog.askopenfilename(
            initialdir=self.folder, title="Select a tiff file", filetypes=[("tiff files", "*.tif")]
        )
        if self.app is not None:
            self.app.log_message("Imported tiff file: {}".format(self.tifffilename))
        
        self.DPFolder = os.path.dirname(self.tifffilename) + "/DP"
        
    def import_RElog(self):
        # import RElog file
        self.relogfilename = filedialog.askopenfilename(
            initialdir=self.folder, title="Select a RElog file", filetypes=[("txt files", "*.txt")]
        )
        if self.app is not None:
            self.app.log_message("Imported RElog file: {}".format(self.relogfilename))
        
    def correlationanalysis(self):
        
        #1, unrotate the regFrames with the detected rotation center
        if self.app is not None:
            self.app.log_message("Unrotate tiff file and perform image registration...")
        
        # read the rotation center from the circlecenter txt file
        circlecenterfilename = self.DPFolder + "/circlecenter.txt"
        with open(circlecenterfilename, "r") as f:
            # read the last row
            last_line = f.readlines()[-1]
            # assign the x and y coordinates to self.rotx and self.roty
            self.rotx = float(last_line.split()[0])
            self.roty = float(last_line.split()[1])
        
        S = SITiffIO()
        S.open_tiff_file(self.tifffilename, "r")
        S.open_rotary_file(self.relogfilename)
        #extract the last self.numFrames frames from the tiff file
        tailArray, tailAng = S.tail(int(self.numFrames.get()))
               
        # unrotate each frame in the tiff file with the detected rotation center
        self.unrotFrames  = UnrotateCropFrame(tailArray, tailAng, rotCenter=[self.rotx, self.roty])
        
        #perform image registraion
        self.meanRegImg, self.regFrames = RegFrame(self.unrotFrames)
        
        #display the meanRegImg frames in the canvas
        self.display_regFrame()
        
        #2, perform correlation analysis
        if self.app is not None:
            self.app.log_message("Perform Correlation Analysis...")
        #load the mean stacks 'named meanstacks.npy' in the addfolder which is a npy file
        meanstacks = np.load(self.DPFolder + "/meanstacks.npy")
        
        ops = suite2p.default_ops()
        self.corrMatrix = compute_zpos_sp(meanstacks, self.regFrames, ops)
        
        #display the corrMatrix in the canvas
        self.display_corrMatrix()
 
    def display_regFrame(self):
        #visual the corrMatrix in the 512*256 canvas
        fig = plt.figure(figsize=(512/100,512/100),dpi=100)  
        
        #show the meanRegImg in the canvas
        plt.imshow(self.meanRegImg, cmap='gray')
        plt.axis('off')
        
        # read the rotation center from the circlecenter txt file
        circlecenterfilename = self.DPFolder + "/circlecenter.txt"
        with open(circlecenterfilename, "r") as f:
            # read the last row
            last_line = f.readlines()[-1]
            # assign the x and y coordinates to self.rotx and self.roty
            rotx = float(last_line.split()[0])
            roty = float(last_line.split()[1])
        
        fig.savefig(self.DPFolder + "/meanReg.png")
        
        #convert the png file to a tk image and display it in the canvas
        self.meanRegImg_tk = ImageTk.PhotoImage(Image.open(self.DPFolder + "/meanReg.png")) 
        self.canvas_reg.create_image(rotx, roty, anchor="center", image=self.meanRegImg_tk)       
        
    def display_corrMatrix(self):
        
        #visual the corrMatrix in the 512*256 canvas
        fig = plt.figure(figsize=(512/100,256/100),dpi=100)
        
        nplanes, nframes = self.corrMatrix.shape

        #plot 1 occupy 3/4 of the canvas and plot 2 occupy 1/4 of the canvas
        gs = GridSpec(1, 2, width_ratios=[3, 1])

        ax1 = fig.add_subplot(gs[0, 0])  
        ax1.imshow(self.corrMatrix, aspect='auto',cmap='gray')
        ax1.set_xlabel('Frame Number')
        ax1.set_ylabel('Stack index')
        #recenter the y label with zero representing the middle stack
        ax1.set_yticks(np.arange(0, nplanes, 5))    
        ax1.set_yticklabels(np.arange(0, nplanes, 5)-int(nplanes/2))
        #add a red line to separate the two blocks
        ax1.axhline(y=nplanes/2, color='r', linestyle='-')

        #do the second plot
        ax2 = fig.add_subplot(gs[0, 1])
        #sum the correlation matrix along the frame axis
        sumCorrMatrix = np.sum(self.corrMatrix, axis=1)
        #plot with a grey line
        ax2.plot(sumCorrMatrix, np.arange(0, nplanes), color='grey')
        ax2.set_xlabel('Sum of cc')
        ax2.set_yticks(np.arange(0, nplanes, 5))
        #reset y label every 5 stacks and subtratc the middle stack index
        ax2.set_yticklabels(np.arange(0, nplanes, 5)-int(nplanes/2))
        #flip y axis
        ax2.set_ylim(ax2.get_ylim()[::-1])
        #add a red line to separate the two blocks
        ax2.axhline(y=nplanes/2, color='r', linestyle='-')
        #find the peak of the sumCorrMatrix and plot it with a red dot
        maxIndex = np.argmax(sumCorrMatrix)
        ax2.plot(sumCorrMatrix[maxIndex], maxIndex, 'ro')
        #get the shift amount
        shiftamount = maxIndex - int(nplanes/2)   
        #add a text and the right bottom corner to show the shift amount on plot 2
        ax2.text(0.5, 0.1, 'Dft= ' + str(shiftamount), horizontalalignment='center', verticalalignment='center', transform=ax2.transAxes, color='r')

        plt.tight_layout()
        # Remove the some empty space between the subplots
        plt.subplots_adjust(wspace=0.2)
        fig.savefig(self.DPFolder + "/corrMatrix.png")
        
        #convert the png file to a tk image and display it in the canvas
        self.corrMatrix_tk = ImageTk.PhotoImage(Image.open(self.DPFolder + "/corrMatrix.png")) 
        self.canvas_corr.create_image(0, 0, anchor="nw", image=self.corrMatrix_tk)

        if self.app is not None:
            #log the shift amount
            if shiftamount < 0:
                self.app.log_message("Move zforcus {} micrometers down".format(-shiftamount))
            else:
                self.app.log_message("Move zforcus {} micrometers up".format(shiftamount))   

    
if __name__ == "__main__":
    root = tk.Tk()
    app = ZdriftProcessor(
        master=root,
        folder="/home/zilong/Desktop/2D2P/Data",
    )
    app.mainloop()

