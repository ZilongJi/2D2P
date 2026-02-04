# Create a tkinter GUI class that to 
# 1: import a zstack tiff file as well as a RElog file.
# 2: create widgets to import the number of volumes, stacks and frames
# 3: create a button to get the mean frames for each stack 
#     -- first unrotate each frame
#     -- second crop each frame
#     -- third get the mean frame for each stack
# 4: create a canvas to display the processed stack frames
#    -- by click the up and down arrows, display different frames in the stack file

import os
import numpy as np
import shutil
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk, ImageDraw
from utils_image import get_meanZstack
# from scanimagetiffio import SITiffIO
import tifffile

class StackProcessor(tk.Frame):
    def __init__(self, master=None, folder=None, app=None):
        super().__init__(master)
        self.master = master
        self.folder = folder
        self.app = app
        self.grid()

        # add tk.Entry widgets to allow users to enter the number of volumes, stacks and frames
        self.volumes = tk.StringVar()
        self.stacks = tk.StringVar()
        self.frames = tk.StringVar()

        # initialize parameters for later use
        self.display_index = 10

        self.create_widgets()
        self.create_canvas()       

    def create_widgets(self):
        # create a button to import tiff file
        tk.Button(self, text="Import ZStack", command=self.import_tiff).grid(row=0, column=0)

        # create a button to import RElog file
        tk.Button(self, text="Import RElog file", command=self.import_RElog).grid(row=1, column=0)

        # Create labels and entry widgets for the three numbers entered by the user
        tk.Label(self, text="Volumes:").grid(row=2, column=0)
        tk.Entry(self, textvariable=self.volumes).grid(row=3, column=0)

        tk.Label(self, text="Stacks:").grid(row=4, column=0)
        tk.Entry(self, textvariable=self.stacks).grid(row=5, column=0)

        tk.Label(self, text="Frames:").grid(row=6, column=0)
        tk.Entry(self, textvariable=self.frames).grid(row=7, column=0)

        # create a button to unrotate the zstacks
        tk.Button(self, text="Get mean Zstacks", command=self.getmeanzstack).grid(row=0, column=1)

    def create_canvas(self):
        # create a canvas to display the processed stack frames
        # there are several frames in the stack, so I want to create a canvas
        # that allows me to display each of the frames in the stack by clicking
        # the up and down arrows (need to be created)
        self.canvas = tk.Canvas(self, height=512, width=512, bg="#4D4D4D")
        self.canvas.grid(row=8, column=0, columnspan=2)

    def import_tiff(self):
        # filedialog: and set initialdir set as self.folder, filetypes set as tiff and all files
        self.tifffilename = filedialog.askopenfilename(
            initialdir=self.folder,
            title="Select file",
            filetypes=(("tiff files", "*.tif"), ("all files", "*.*")),
        )
        if self.app is not None:
            # log the message in the text widget
            self.app.log_message("Imported tiff file: " + self.tifffilename)

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

    def getmeanzstack(self):
        if self.app is not None:
            self.app.log_message("Get mean Zstacks by unrotating and cropping each frame, and then averaging over each stack")

        # read the rotation center from the circlecenter txt file
        circlecenterfilename = self.DPFolder + "/circlecenter.txt"
        with open(circlecenterfilename, "r") as f:
            # read the last row
            last_line = f.readlines()[-1]
            # assign the x and y coordinates to self.rotx and self.roty
            self.rotx = float(last_line.split()[0])
            self.roty = float(last_line.split()[1])
            
        S = SITiffIO()
        #load the tiff file together with the rotary data
        S.open_tiff_file(self.tifffilename, "r") 
        S.open_rotary_file(self.relogfilename)
        S.interp_times()  # might take a while...
        
        # get the number of volumes, stacks and frames from the user input
        try:
            num_v = int(self.volumes.get())
            num_s = int(self.stacks.get())
            num_f = int(self.frames.get())
        except ValueError:
            if self.app is not None:
                self.app.log_message(
                    "Error! Enter the number of volumes, stacks and frames"
                )
            return
        
        #get the mean frame for each stack (get_meanZstack processes each stack one-byone, memory-save)
        self.meanStacks = get_meanZstack(S, num_v, num_s, num_f, Rotcenter=[self.rotx, self.roty], ImgReg=True)
        
        
        if self.app is not None:
            self.app.log_message("Save the mean Zstacks as npy files and png images to the DP folder")

        #save the unrotated stacks as a npy file in the DP folder
        np.save(self.DPFolder + "/meanstacks.npy", self.meanStacks)
        
        #save each stack as a png file in the DP folder
        
        #create a folder to save the stack png files
        #if the folder already exists, delete it and create a new one
        if os.path.exists(self.DPFolder + "/zstacks"):
            shutil.rmtree(self.DPFolder + "/zstacks")
        os.mkdir(self.DPFolder + "/zstacks")
        
        #save each stack as a png file using plt
        for i in range(self.meanStacks.shape[0]):
            fig = plt.figure()
            plt.imshow(self.meanStacks[i,:,:], cmap="gray")
            plt.axis("off")
            plt.savefig(self.DPFolder + "/zstacks/stack" + str(i+1) + ".png", bbox_inches="tight", pad_inches=0)
            plt.close(fig)
            
        if self.app is not None:
            self.app.log_message("Unrotation and crop finished...")
            
        # display
        self.display_processed_images()

    def display_processed_images(self):
        """
        display the processed images in the canvas by clicking up and down keys on the keyboard
        """
        self.canvas.delete("all")
        # normalize the image to 0-255 and convert to uint8 before displaying
        image = self.meanStacks[self.display_index]
        image = (image - image.min()) / (image.max() - image.min()) * 255
        image = image.astype("uint8")

        # convert the image to ImageTk format
        image = Image.fromarray(image).convert("L")
        #add a number to the image to indicate the stack index with red color
        draw = ImageDraw.Draw(image)
        draw.text((10, 10), str(self.display_index + 1), fill='red')
        
        self.img_tk = ImageTk.PhotoImage(image)

        # display the image
        # the image has a size different from 512*512 after cropping, which is for example, 362*362
        # so display it with a center same as the rotation center
        self.canvas.create_image(
            self.rotx, self.roty, anchor="center", image=self.img_tk
        )
        # bind the up and down keys to the canvas
        self.canvas.bind("<Up>", self.previous_image)
        self.canvas.bind("<Down>", self.next_image)
        self.canvas.focus_set()  # set the focus on canvas

    def next_image(self, event):
        if self.display_index < len(self.meanStacks) - 1:
            self.display_index += 1
        self.display_processed_images()

    def previous_image(self, event):
        if self.display_index > 0:
            self.display_index -= 1
        self.display_processed_images()


if __name__ == "__main__":
    root = tk.Tk()
    app = StackProcessor(
        master=root,
        folder="/home/zilong/Desktop/2D2P/Data/162"
    )
    app.mainloop()
