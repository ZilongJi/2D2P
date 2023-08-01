import os
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from utils_image import UnrotateCropFrame, RegFrame, findFOV
from scanimagetiffio import SITiffIO
from scipy.ndimage import gaussian_filter1d

class FOVFinder(tk.Frame):
    def __init__(self, master=None, folder=None, app=None):
        super().__init__(master)
        self.master = master
        self.folder = folder
        self.app = app
        self.grid()

        self.create_widgets()
        #self.create_canvas_zstack()    
        #self.create_canvas_tiff()
        self.create_canvas_FOVanalysis()   

    def create_widgets(self):
        # create a button to import zstack file
        tk.Button(self, text="Import ZStack npy", command=self.import_zstack).grid(row=0, column=0)

        # create a button to import tiff file
        tk.Button(self, text="Import Tiff", command=self.import_tiff).grid(row=1, column=0)
        
        # create a button to import RElog file
        tk.Button(self, text="Import RElog file", command=self.import_RElog).grid(row=2, column=0)

        # create a button to find the field of view
        tk.Button(self, text="Find FOV", command=self.findFOV).grid(row=3, column=0)
        
        # create a canvas to display the field of view

    def create_canvas_zstack(self):
        # create a canvas to display the processed stack frames
        # there are several frames in the stack, so I want to create a canvas
        # that allows me to display each of the frames in the stack by clicking
        # the up and down arrows (need to be created)
        self.canvas_zstack = tk.Canvas(self, height=512, width=512, bg="#4D4D4D")
        self.canvas_zstack.grid(row=4, column=0)
    
    def create_canvas_tiff(self):
        self.canvas_tiff = tk.Canvas(self, height=512, width=512, bg="#4D4D4D")
        self.canvas_tiff.grid(row=4, column=1)
    
    def create_canvas_FOVanalysis(self):
        self.canvas_FOVanalysis = tk.Canvas(self, height=512, width=512, bg="#4D4D4D")
        self.canvas_FOVanalysis.grid(row=4, column=0)

    def import_zstack(self):
        # filedialog: and set initialdir set as Desktop, filetypes set as tiff and all files
        self.zstackfilename = filedialog.askopenfilename(
            initialdir=self.folder,
            title="Select file",
            filetypes=(("tiff files", "*.npy"), ("all files", "*.*")),
        )
        if self.app is not None:
            # log the message in the text widget
            self.app.log_message("Imported numpy zstack file: " + self.zstackfilename)
            
        self.zstack = np.load(self.zstackfilename)
    
    def import_tiff(self):
        # filedialog: and set initialdir set as Desktop, filetypes set as tiff and all files
        self.tifffilename = filedialog.askopenfilename(
            initialdir=self.folder,
            title="Select file",
            filetypes=(("tiff files", "*.tif"), ("all files", "*.*")),
        )
        if self.app is not None:
            # log the message in the text widget
            self.app.log_message("Imported tiff file: " + self.tifffilename)
            
        self.FOVFolder = os.path.dirname(self.tifffilename) + "/FOV"
        if os.path.exists(self.FOVFolder):
            os.system("rm -rf " + self.FOVFolder)
        os.mkdir(self.FOVFolder)

        self.DPFolder = os.path.dirname(self.tifffilename) + "/DP"
        
    def import_RElog(self):
        # filedialog: and set initialdir set as Desktop, filetypes set as txt and all files
        self.relogfilename = filedialog.askopenfilename(
            initialdir=self.folder,
            title="Select file",
            filetypes=(("txt files", "*.txt"), ("all files", "*.*")),
        )
        
        if self.app is not None:        
            # log the message in the text widget
            self.app.log_message("Imported RElog file: " + self.relogfilename)

    def findFOV(self):
        if self.app is not None:
            self.app.log_message("Find the field of view...")

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
        tailArray, tailAng = S.tail(500)   
        # unrotate each frame in the tiff file with the detected rotation center
        unrotFrames  = UnrotateCropFrame(tailArray, tailAng, rotCenter=[self.rotx, self.roty])
        #perform image registraion
        meanRegImg, _ = RegFrame(unrotFrames)
        
        self.meanRegImg = meanRegImg
        
        self.ymax, self.xmax, self.zcorr = findFOV(self.zstack, meanRegImg, maxrotangle=30)
        self.zcorr_gs = gaussian_filter1d(self.zcorr.copy(), 2, axis=0)
        
        #display the max zstack in the canvas
        #self.display_zstack()
        #display the mean tiff in the canvas
        #self.display_tiff()
        #display the FOV analysis in the canvas
        self.display_FOVanalysis()
        
    def display_zstack(self):  
        self.canvas_zstack.delete("all")  
        #find the max value in zcorr and mark it with a red dot
        maxvalue = np.max(self.zcorr_gs)
        maxindex = np.where(self.zcorr_gs == maxvalue)
        
        stack_ind = maxindex[0][0]
        
        stack_img = self.zstack[stack_ind]
        
        #chnage to tkinter image for display
        fig = plt.figure(figsize=(512/100,512/100),dpi=100) 
        #show the meanRegImg in the canvas
        plt.imshow(stack_img, cmap='gray')
        plt.axis('off')
        fig.savefig(self.FOVFolder + "/maxstack.png")

        stack_img = (stack_img - stack_img.min()) / (stack_img.max() - stack_img.min()) * 255
        stack_img = stack_img.astype("uint8")

        # convert the image to ImageTk format
        stack_img_tk = ImageTk.PhotoImage(Image.fromarray(stack_img).convert("L"))
        
        # display the max zstack in the canvas
        self.canvas_zstack.create_image(self.rotx, self.roty, anchor="center", image=stack_img_tk)
            
    def display_tiff(self):
        self.canvas_tiff.delete("all")  
        fig = plt.figure(figsize=(512/100,512/100),dpi=100) 
        plt.imshow(self.meanRegImg, cmap='gray')
        plt.axis('off')
        fig.savefig(self.FOVFolder + "/meanReg.png")     

        meanImg = self.meanRegImg
        meanImg = (meanImg - meanImg.min()) / (meanImg.max() - meanImg.min()) * 255
        meanImg = meanImg.astype("uint8")

        # convert the image to ImageTk format
        meanImg_tk = ImageTk.PhotoImage(Image.fromarray(meanImg).convert("L"))
        # display the mean tiff in the canvas
        self.canvas_tiff.create_image(self.rotx, self.roty, anchor="center", image=meanImg_tk)

    def display_FOVanalysis(self):
        fig = plt.figure(figsize=(512/100,512/100),dpi=100)
        plt.imshow(self.zcorr_gs, cmap='gray')
        
        maxvalue = np.max(self.zcorr_gs)
        maxindex = np.where(self.zcorr_gs == maxvalue)       
        plt.plot(maxindex[1][0], maxindex[0][0], 'ro')
        
        plt.xticks(np.arange(0, 61, 10), np.arange(-30, 31, 10))
        
        plt.xlabel('rotation degree')
        plt.ylabel('stack index')
        
        #according to the maxindex, find the value in self.ymax and self.xmax
        ymax_value = self.ymax[maxindex[0][0], maxindex[1][0]]
        xmax_value = self.xmax[maxindex[0][0], maxindex[1][0]]
        
        #add text to the plot with red color
        plt.text(maxindex[1][0], maxindex[0][0], 'maxZplane='+ str(maxindex[0][0])+'\nshift (y,x) = (' + str(ymax_value) +',' + str(xmax_value)+')', fontsize=12, color='red')
        
        fig.savefig(self.FOVFolder + "/FOV.png")
        
        #convert the png file to a tk image and display it in the canvas
        self.fov_tk = ImageTk.PhotoImage(Image.open(self.FOVFolder + "/FOV.png")) 
        self.canvas_FOVanalysis.create_image(0, 0, anchor="nw", image=self.fov_tk)


if __name__ == "__main__":
    root = tk.Tk()
    app = FOVFinder(
        master=root,
        folder="/home/zilong/Desktop/2D2P/Data/183_28072023"
    )
    app.mainloop()