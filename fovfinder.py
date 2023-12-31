import os
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from utils_image import UnrotateCropFrame, RegFrame, findFOV, hist_match
from scanimagetiffio import SITiffIO
from scipy.ndimage import gaussian_filter

class FOVFinder(tk.Frame):
    def __init__(self, master=None, folder=None, app=None):
        super().__init__(master)
        self.master = master
        self.folder = folder
        self.app = app
        self.grid()
        
        # add tk.Entry widgets to allow users to enter the number of volumes, stacks and frames
        self.maxrotangle = tk.StringVar()

        self.create_widgets()
        self.create_canvas_zstack()    
        self.create_canvas_tiff()
        self.create_canvas_FOVanalysis()   

    def create_widgets(self):
        # create a button to import zstack file
        tk.Button(self, text="Import ZStack npy", command=self.import_zstack).grid(row=0, column=0)

        # create a button to import tiff file
        tk.Button(self, text="Import Tiff", command=self.import_tiff).grid(row=1, column=0)
        
        # create a button to import RElog file
        tk.Button(self, text="Import RElog file", command=self.import_RElog).grid(row=2, column=0)
        
        # Create labels and entry widgets for the three numbers entered by the user
        tk.Label(self, text="maxrotangle:").grid(row=3, column=0)
        tk.Entry(self, textvariable=self.maxrotangle).grid(row=4, column=0)

        # create a button to find the field of view
        tk.Button(self, text="Find FOV", command=self.findFOV).grid(row=5,column=0)
        
        # create a canvas to display the field of view

    def create_canvas_zstack(self):
        # create a canvas to display the processed stack frames
        # there are several frames in the stack, so I want to create a canvas
        # that allows me to display each of the frames in the stack by clicking
        # the up and down arrows (need to be created)
        self.canvas_zstack = tk.Canvas(self, height=512, width=512, bg="#4D4D4D")
        self.canvas_zstack.grid(row=6, column=0)
    
    def create_canvas_tiff(self):
        self.canvas_tiff = tk.Canvas(self, height=512, width=512, bg="#4D4D4D")
        self.canvas_tiff.grid(row=6, column=1)
    
    def create_canvas_FOVanalysis(self):
        self.canvas_FOVanalysis = tk.Canvas(self, height=256, width=1024, bg="#4D4D4D")
        self.canvas_FOVanalysis.grid(row=7, column=0, columnspan=2)

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
        tailArray, tailAng = S.tail(1000)   
        # unrotate each frame in the tiff file with the detected rotation center
        unrotFrames  = UnrotateCropFrame(tailArray, tailAng, rotCenter=[self.rotx, self.roty])
        #perform image registraion
        meanRegImg, _ = RegFrame(unrotFrames)
        
        #histogram equalization
        #self.meanRegImg = hist_match(meanRegImg, self.zstack[35]).astype(np.int16)
        
        self.meanRegImg = meanRegImg
        
        maxrotangle = int(self.maxrotangle.get())
        
        self.ymax, self.xmax, self.zcorr = findFOV(self.zstack, self.meanRegImg, maxrotangle=maxrotangle)
        
        #gaussian filter the ymax, xmax, and zcorr
        self.ymax_gs = gaussian_filter(self.ymax.copy(), 2)
        self.xmax_gs = gaussian_filter(self.xmax.copy(), 2)
        self.zcorr_gs = gaussian_filter(self.zcorr.copy(), 2)
        
        #display the max zstack in the canvas
        self.display_zstack()
        #display the mean tiff in the canvas
        self.display_tiff()
        #display the FOV analysis in the canvas
        self.display_FOVanalysis()
        
    def display_zstack(self):  
        self.canvas_zstack.delete("all")  
        maxvalue = np.max(self.zcorr_gs)
        self.maxindex = np.where(self.zcorr_gs == maxvalue)
        
        stack_ind = self.maxindex[0][0]
        
        stack_img = self.zstack[stack_ind]

        stack_img = (stack_img - stack_img.min()) / (stack_img.max() - stack_img.min()) * 255
        stack_img = stack_img.astype("uint8")
        #save
        im = Image.fromarray(stack_img).convert("L")
        im.save(self.FOVFolder + "/maxzstack_index_"+str(stack_ind)+".png")

        #convert the image to ImageTk format
        self.stack_img_tk = ImageTk.PhotoImage(im)
        
        # display the max zstack in the canvas
        self.canvas_zstack.create_image(self.rotx, self.roty, anchor="center", image=self.stack_img_tk)
            
    def display_tiff(self):
        self.canvas_tiff.delete("all")  
        #save the meanRegImg as a png file using Image
        meanImg = self.meanRegImg.copy()
        meanImg = (meanImg - meanImg.min()) / (meanImg.max() - meanImg.min()) * 255
        meanImg = meanImg.astype("uint8")
        im = Image.fromarray(meanImg).convert("L")
        im.save(self.FOVFolder + "/meanReg.png")
        
        # convert the image to ImageTk format
        self.im_tk = ImageTk.PhotoImage(im)
        # display the mean tiff in the canvas
        self.canvas_tiff.create_image(self.rotx, self.roty, anchor="center", image=self.im_tk)

    def display_FOVanalysis(self):
        #according to the maxindex, find the value in self.ymax and self.xmax
        ymax_value = self.ymax[self.maxindex[0][0], self.maxindex[1][0]]
        xmax_value = self.xmax[self.maxindex[0][0], self.maxindex[1][0]]

        fig = plt.figure(figsize=(1024/100,256/100),dpi=100)
        
        cmap = 'cividis'
        
        maxrotangle = int(self.maxrotangle.get())
        
        interval = maxrotangle//3
        plt.subplot(1,3,1)
        plt.imshow(self.zcorr_gs, cmap=cmap, aspect='auto')
        plt.xticks(np.arange(0, 2*maxrotangle+1, interval), np.arange(-maxrotangle, maxrotangle+1, interval))
        plt.colorbar()
        plt.title('zcorr map, maxzplane='+ str(self.maxindex[0][0]))
        plt.xlabel('rotation degree')
        plt.ylabel('stack index')
        #mark the max value with a red dot
        plt.plot(self.maxindex[1][0], self.maxindex[0][0], 'ro')
        
        cmap='coolwarm'
        plt.subplot(1,3,2)
        plt.imshow(self.ymax_gs, cmap=cmap, aspect='auto')
        plt.xticks(np.arange(0, 2*maxrotangle+1, interval), np.arange(-maxrotangle, maxrotangle+1, interval))
        plt.colorbar()
        plt.title('ymax map, shift='+str(ymax_value))
        plt.xlabel('rotation degree')
        plt.ylabel('stack index')
        #mark the max value with a red dot
        plt.plot(self.maxindex[1][0], self.maxindex[0][0], 'ro')


        plt.subplot(1,3,3)
        plt.imshow(self.xmax_gs, cmap=cmap, aspect='auto')
        plt.xticks(np.arange(0, 2*maxrotangle+1, interval), np.arange(-maxrotangle, maxrotangle+1, interval))
        plt.colorbar()
        plt.title('xmax map, shift='+str(xmax_value))
        plt.xlabel('rotation degree')
        plt.ylabel('stack index')
        #mark the max value with a red dot
        plt.plot(self.maxindex[1][0], self.maxindex[0][0], 'ro')

        plt.tight_layout()

        fig.savefig(self.FOVFolder + "/FOV.png")
        
        #convert the png file to a tk image and display it in the canvas
        self.fov_tk = ImageTk.PhotoImage(Image.open(self.FOVFolder + "/FOV.png")) 
        self.canvas_FOVanalysis.create_image(0, 0, anchor="nw", image=self.fov_tk)

if __name__ == "__main__":
    root = tk.Tk()
    app = FOVFinder(
        master=root,
        folder="/home/zilong/Desktop/2D2P/Data"
    )
    app.mainloop()