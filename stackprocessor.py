# I want to create a tkinter GUI class that will allow me to import a tiff file,
# as well as a RElog file.
# After import these files, I want to create a widget that allows me to
# unrotate each frame according to information in the log files
# I also want to create a canvas that will allow me to display each of the frames
# in the processed stack file, by click the up and down arrows,
# I can display different frames in the stack file
import os
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from utils_image import UnrotateCropFrame
from scanimagetiffio import SITiffIO

class StackProcessor(tk.Frame):
    def __init__(self, master=None, folder=None, app=None):
        super().__init__(master)
        self.master = master
        self.folder = folder
        self.app = app
        self.grid()

        # add tk.Entry widgets to allow users to enter
        # the number of volumes, stacks and frames
        self.volumes = tk.StringVar()
        self.stacks = tk.StringVar()
        self.frames = tk.StringVar()

        # initialize parameters for later use
        self.display_index = 21

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
        tk.Button(self, text="Unrotate ZStacks", command=self.unrotatezstack).grid(row=0, column=1)

    def create_canvas(self):
        # create a canvas to display the processed stack frames
        # there are several frames in the stack, so I want to create a canvas
        # that allows me to display each of the frames in the stack by clicking
        # the up and down arrows (need to be created)
        self.canvas = tk.Canvas(self, height=512, width=512, bg="#4D4D4D")
        self.canvas.grid(row=8, column=0, columnspan=2)

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

    def unrotatezstack(self):
        if self.app is not None:
            self.app.log_message("Unrotate tiff file...")

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
        #load the tiff file together with the rotary data
        S.open_tiff_file(self.tifffilename, "r") 
        S.open_rotary_file(self.relogfilename)
        S.interp_times()  # might take a while...
        #get all frames and angles in S
        Array = []; 
        for i in range(S.get_n_frames()):
            Array.append(S.get_frame(i+1))
        #change Array to a numpy array
        Array = np.array(Array)
        Angle = S.get_all_theta()

        # unrotate each frame in the tiff file with the detected rotation center
        unrotcropFrames = UnrotateCropFrame(Array, Angle, rotCenter=[self.rotx, self.roty])      
        
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

        # reshape the unrotFrames to a 5D array
        unrotcropFrames = unrotcropFrames.reshape(
            num_v, num_s, num_f, unrotcropFrames.shape[1], unrotcropFrames.shape[2]
        )
        # average across the first and third dimension
        self.meanStacks = np.mean(unrotcropFrames, axis=(0, 2))
        
        #save the unrotated stacks as a npy file in the app folder
        np.save(self.DPFolder + "/meanstacks.npy", self.meanStacks)
        
        if self.app is not None:
            self.app.log_message("Unrotation and crop finished...")


        # display
        self.display_processed_images()

    def display_processed_images(self):
        """
        display the processed images in the canvas by clicking the created up and down widgets
        """
        self.canvas.delete("all")
        # display an image in the stack
        # normalize the image to 0-255 and convert to uint8 before displaying
        image = self.meanStacks[self.display_index]
        image = (image - image.min()) / (image.max() - image.min()) * 255
        image = image.astype("uint8")

        # convert the image to ImageTk format
        self.img_tk = ImageTk.PhotoImage(Image.fromarray(image).convert("L"))

        # display the image
        # the image has its own size, which is 362 by 362
        # I want to display it with the center of [256,256] on the canvas

        self.canvas.create_image(
            self.rotx, self.roty, anchor="center", image=self.img_tk
        )
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
