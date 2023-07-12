# I want to create a tkinter GUI class that will allow me to import a tiff file, a RElog file, and a ROIs file. and a VRlog file
# After import these files, I want to create a widget that allows me to unrotate each frame according to information in the log files
# and then crop the unrotated frames as in stackprocessor.py

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from utils_image import UnrotateFrame_SITiffIO

class ZdriftProcessor(tk.Frame):
    def __init__(self, master=None, folder=None, appfolder=None, app=None):
        super().__init__(master)
        self.master = master
        self.folder = folder
        self.appfolder = appfolder
        self.app = app
        self.grid()
        
        # initialize the numer of frames for later correlation analysis
        self.numFrames = tk.StringVar()

        self.create_widgets()
        self.create_canvas()
        
    def create_widgets(self):
        #create a button to import tiff file
        tk.Button(self, text="Import tiff", command=self.import_tiff).grid(row=0, column=0)
        
        #create a button to import RElog file
        tk.Button(self, text="Import RElog", command=self.import_RElog).grid(row=1, column=0)
        
        # create a label and entry to set the number of frames for correlation analysis
        tk.Label(self, text="numFrames").grid(row=2, column=0)
        tk.Entry(self, textvariable=self.numFrames).grid(row=3, column=0)
        
        #create a button to unrottate the tiff
        tk.Button(self, text="Unrotate Tiff", command=self.unrotatetiff).grid(row=0, column=1)

        #create a button to do correlation analysis
        tk.Button(self, text="Correlation Analysis", command=self.correlationanalysis).grid(row=1, column=1)
        
        
    def create_canvas(self):
        # create a canvas to display results
        self.canvas = tk.Canvas(self, width=512, height=256, bg="white")
        self.canvas.grid(row=4, column=0, columnspan=2)
        
    def import_tiff(self):
        # import tiff file
        self.tifffilename = filedialog.askopenfilename(
            initialdir=self.folder, title="Select a tiff file", filetypes=[("tiff files", "*.tif")]
        )
        self.app.log_message("Imported tiff file: {}".format(self.tifffilename))
        
    def import_RElog(self):
        # import RElog file
        self.relogfilename = filedialog.askopenfilename(
            initialdir=self.folder, title="Select a RElog file", filetypes=[("txt files", "*.txt")]
        )
        self.app.log_message("Imported RElog file: {}".format(self.relogfilename))
        
    def unrotatetiff(self):
        self.app.log_message("Unrotate tiff file...")
        
        # read the rotation center from the circlecenter txt file
        circlecenterfilename = self.appfolder + "/circlecenter.txt"
        with open(circlecenterfilename, "r") as f:
            # read the last row
            last_line = f.readlines()[-1]
            # assign the x and y coordinates to self.rotx and self.roty
            self.rotx = float(last_line.split()[0])
            self.roty = float(last_line.split()[1])
            
        # unrotate each frame in the tiff file with the detected rotation center
        unrotFrames = UnrotateFrame_SITiffIO(
            self.tifffilename, self.relogfilename, rotCenter=[self.rotx, self.roty], numFrames=int(self.numFrames.get())
        )

        self.unrotFrames = np.array(unrotFrames)
        
        self.app.log_message("Unrotation finished...")
        
        #display the unrotated frames in the canvas

    def correlationanalysis(self):
        self.app.log_message("Perform Correlation Analysis...")
        #load the mean stacks 'named meanstacks.npy' in the addfolder which is a npy file
        meanstacks = np.load(self.appfolder + "/meanstacks.npy")
        
        #corrleting each frame in self.unrotFrames with meanstacks
        #and add the correlation value to a matrix called corrMatrix
        corrMatrix = np.zeros((self.unrotFrames.shape[0], meanstacks.shape[0]))
        for i in range(self.unrotFrames.shape[0]):
            for j in range(meanstacks.shape[0]):
                corrMatrix[i,j] = np.corrcoef(self.unrotFrames[i,:,:].flatten(), meanstacks[j,:,:].flatten())[0,1]
        
        self.corrMatrix = corrMatrix
        
        #display the corrMatrix in the canvas
        self.display_processed_images()
        
    def display_processed_images(self):
        
        #visual the corrMatrix in the 512*256 canvas
        fig = plt.figure(figsize=(512/100,256/100),dpi=100)

        #plot 1 occupy 3/4 of the canvas and plot 2 occupy 1/4 of the canvas
        gs = GridSpec(1, 2, width_ratios=[3, 1])

        ax1 = fig.add_subplot(gs[0, 0])  
        ax1.imshow(self.corrMatrix.T, aspect='auto',cmap='viridis')
        ax1.set_xlabel('Frame Number')
        ax1.set_ylabel('Stack index')
        #recenter the y label with zero representing the middle stack
        ax1.set_yticks(np.arange(0, self.corrMatrix.shape[1], 5))    
        ax1.set_yticklabels(np.arange(0, self.corrMatrix.shape[1], 5)-int(self.corrMatrix.shape[1]/2))
        #add a red line to separate the two blocks
        ax1.axhline(y=self.corrMatrix.shape[1]/2, color='r', linestyle='-')

        #do the second plot
        ax2 = fig.add_subplot(gs[0, 1])
        #sum the correlation matrix along the frame axis
        sumCorrMatrix = np.sum(self.corrMatrix, axis=0)
        #plot with a grey line
        ax2.plot(sumCorrMatrix, np.arange(0, self.corrMatrix.shape[1]), color='grey')
        ax2.set_xlabel('Sum of cc')
        ax2.set_yticks(np.arange(0, self.corrMatrix.shape[1], 5))
        #reset y label every 5 stacks and subtratc the middle stack index
        ax2.set_yticklabels(np.arange(0, self.corrMatrix.shape[1], 5)-int(self.corrMatrix.shape[1]/2))
        #flip y axis
        ax2.set_ylim(ax2.get_ylim()[::-1])
        #add a red line to separate the two blocks
        ax2.axhline(y=self.corrMatrix.shape[1]/2, color='r', linestyle='-')
        #find the peak of the sumCorrMatrix and plot it with a red dot
        maxIndex = np.argmax(sumCorrMatrix)
        ax2.plot(sumCorrMatrix[maxIndex], maxIndex, 'ro')
        #get the shift amount
        shiftamount = maxIndex - int(self.corrMatrix.shape[1]/2)   
        #add a text and the right bottom corner to show the shift amount on plot 2
        ax2.text(0.5, 0.1, 'Dft= ' + str(shiftamount), horizontalalignment='center', verticalalignment='center', transform=ax2.transAxes, color='r')

        plt.tight_layout()
        # Remove the some empty space between the subplots
        plt.subplots_adjust(wspace=0.2)
        fig.savefig(self.appfolder + "/corrMatrix.png")
        
        #convert the png file to a tk image and display it in the canvas
        self.corrMatrix_tk = ImageTk.PhotoImage(Image.open(self.appfolder + "/corrMatrix.png")) 
        self.canvas.create_image(0, 0, anchor="nw", image=self.corrMatrix_tk)

        #log the shift amount
        self.app.log_message("Z-drift shift amount: {}".format(shiftamount))
        if shiftamount < 0:
            self.app.log_message("Move zforcus {} micrometers down".format(-shiftamount))
        else:
            self.app.log_message("Move zforcus {} micrometers up".format(shiftamount))   

    
if __name__ == "__main__":
    root = tk.Tk()
    app = ZdriftProcessor(
        master=root,
        folder="/home/zilong/Desktop/2D2P/Data/162_10072023",
        appfolder="/home/zilong/Desktop/2D2P/Data/162_10072023/APP",
    )
    app.mainloop()

