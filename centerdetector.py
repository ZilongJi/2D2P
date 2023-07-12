# I want to create a class for detecting the center of the circle in the tiff file
# Detailed process will be as follows:
#   1, a widget to import a tiff file, and the RElog file
#   2, a widget to average all the frames in the tiff file, and then present the averaged image
#   3, a widget to draw a circle on the averaged image, and then present the center of the circle
# The name of the class is called centerdetector
#   4, a canvas to display the angle information

# create by Zilong Ji, 4/07/2023
# contact: zilong.ji@ucl.ac.uk

import os
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from scanimagetiffio.scanimagetiffio import SITiffIO
from utils_image import getMeanTiff_randomsampling, getMeanTiff_equalsampling

class CenterDetector(tk.Frame):
    def __init__(self, master=None, folder=None, appfolder=None, app=None):
        super().__init__(master)
        self.master = master
        self.folder = folder
        self.appfolder = appfolder
        self.app = app
        self.grid()

        # initialize the value of fraction or Bins for later use
        self.fractionorBins = tk.StringVar()

        self.create_widgets()
        self.create_canvas()
        self.create_canvas4angle()

        self.circlecenter = np.asarray(
            [0, 0]
        )  # initialze the center position of the drawed circle
        
        self.spaceclicks = 0  # initialize the number of space clicks

    def create_widgets(self):
        # create a button to import tiff file
        tk.Button(self, text="Import tiff", command=self.import_tiff).grid(row=0, column=0)

        # create a button to import RElog file
        tk.Button(self, text="Import RElog", command=self.import_RElog).grid(row=1, column=0)

        # create a label and entry to set the fraction or Bins of the tiff file to be averaged
        tk.Label(self, text="Ave. Frac/Bins").grid(row=2, column=0)
        tk.Entry(self, textvariable=self.fractionorBins).grid(row=3, column=0)

        # create a button to average the tiff file
        tk.Button(self, text="Calculate mean frame", command=self.averagetif).grid(row=0, column=1, rowspan=4)

    def create_canvas(self):
        # arrange the canvas to the right side of widgets
        self.canvas = tk.Canvas(self, width=512, height=512, bg="white")
        self.canvas.grid(row=4, column=0, columnspan=2)

    def create_canvas4angle(self):
        # arrange the canvas to the right side of widgets
        self.canvas4angle = tk.Canvas(self, width=512, height=256, bg="white")
        self.canvas4angle.grid(row=5, column=0, columnspan=2)

    def import_tiff(self):
        # filedialog: and set initialdir set as Desktop, filetypes set as tiff and all files
        self.tifffilename = filedialog.askopenfilename(
            initialdir=self.folder,
            title="Select file",
            filetypes=(("tiff files", "*.tif"), ("all files", "*.*")),
        )
        # log the message in the text widget
        self.app.log_message("Imported tiff file: " + self.tifffilename)

    def import_RElog(self):
        # filedialog: and set initialdir set as Desktop, filetypes set as txt and all files
        self.relogfilename = filedialog.askopenfilename(
            initialdir=self.folder,
            title="Select file",
            filetypes=(("txt files", "*.txt"), ("all files", "*.*")),
        )
        # log the message in the text widget
        self.app.log_message("Imported RElog file: " + self.relogfilename)

    def averagetif(self):
        # read the tiff file via SITiffIO
        self.app.log_message("Reading tiff file...")
        S = SITiffIO()
        S.open_tiff_file(self.tifffilename, "r")
        S.open_rotary_file(self.relogfilename)
        S.interp_times()  # might take a while...
        self.S = S
        self.app.log_message(
            "Counted " + str(S.get_n_frames()) + " frames in the tif file."
        )
        self.app.log_message("Done reading tiff file.")

        try:
            # change string to float
            fracorBins = float(self.fractionorBins.get())
            
            #if fracorBins is smaller than 1, it is a fraction then use getMeanTiff_randomsampling
            #other wise it is a number of bins then use getMeanTiff_equalsampling
            if fracorBins < 1:
                self.app.log_message("Get the averaged image by random sampling...")
                self.meantif = getMeanTiff_randomsampling(self.S, frac=fracorBins)
            else:   
                self.app.log_message("Get the averaged image by equal sampling...")
                self.meantif = getMeanTiff_equalsampling(self.S, numBins=int(fracorBins))
            
            # display the averaged image on the canvas
            self.display_image(self.meantif)
            
        except ValueError:
            self.app.log_message("Error! Please enter a valid fraction number.")

        # display the polar plot of all the angle in the rotary encoder file
        self.angles = S.get_all_theta()
        self.display_angles(self.angles)
        
    def display_angles(self, angles):
        #perform a polar plot of all the angles and display it on the canvas
        #save the polar plot as a png file under self.appfolder
        angles = np.array(angles)
        #do a polar plot of theb histogram of the angles
        #set figure size as 256 *256 pixels
        fig = plt.figure(figsize=(256/100, 256/100))
        #change to a polar plot 
        ax = plt.subplot(111, projection='polar')
        #set the zero angle to the top of the plot
        ax.set_theta_zero_location('N')
        #set the clockwise direction to be the positive direction
        ax.set_theta_direction(-1)
        #plot the histogram
        ax.hist(np.radians(angles), bins=50, density=False)
        
        fig.savefig(self.appfolder + "/angle_distribution.png", dpi=100)
        plt.close(fig)
        
        #display the polar plot on the canvas on the center of the 512*256 canvas
        self.img_ang = Image.open(self.appfolder + "/angle_distribution.png")
        self.imgTK_ang  = ImageTk.PhotoImage(self.img_ang)
        self.canvas4angle.create_image(256, 128, image=self.imgTK_ang, anchor="center")
        
        
    def display_image(self, image):
        # display the image on the canvas with gray colormap
        # save the image
        self.img = Image.fromarray(image)
        self.img = self.img.convert("L")
        self.imgTK = ImageTk.PhotoImage(self.img)
        self.canvas.create_image(0, 0, image=self.imgTK, anchor="nw")

        # save the image as a png file under self.appfolder
        self.img.save(self.appfolder + "/averagedTiff.png")

        # Enabling circle drawing on the image.
        self.canvas.bind("<Button-1>", self.draw_circle)

    def draw_circle(self, event):
        # create a txt file called circlecenter.txt under self.appfolder
        # to save the position of the circle center
        # if the txt file already exists, delete it and create a new one
        self.circlecenterfile = self.appfolder + "/circlecenter.txt"
        if os.path.exists(self.circlecenterfile):
            os.remove(self.circlecenterfile)
        with open(self.circlecenterfile, "w") as f:
            f.write("")

        self.app.log_message("Draw a circle on the image...")
        # draw a circle on the image whose center is the position of the left mouse click
        # the size of the circle can be changed by moving the mouse
        # the position of the circle can be changed by pressing up down left right key
        # the position can be saved by pressing space
        # muilple circles can be drawn on the image by repeating the above steps

        # get the position of the left mouse click
        self.x = event.x
        self.y = event.y
        self.r = 0
        self.circle = self.canvas.create_oval(
            self.x - self.r,
            self.y - self.r,
            self.x + self.r,
            self.y + self.r,
            outline="red",
        )
        self.canvas.bind("<B1-Motion>", self.change_circle)
        self.canvas.bind("<Up>", self.move_up)
        self.canvas.bind("<Down>", self.move_down)
        self.canvas.bind("<Left>", self.move_left)
        self.canvas.bind("<Right>", self.move_right)
        self.canvas.bind("<space>", self.save_position)
        self.canvas.focus_set()  #'focus' the canvas to make the key binding work

    def change_circle(self, event):
        # change the size of the circle
        self.r = ((event.x - self.x) ** 2 + (event.y - self.y) ** 2) ** 0.5
        self.canvas.coords(
            self.circle,
            self.x - self.r,
            self.y - self.r,
            self.x + self.r,
            self.y + self.r,
        )

    def move_up(self, event):
        # move the circle up
        self.y -= 1
        self.canvas.coords(
            self.circle,
            self.x - self.r,
            self.y - self.r,
            self.x + self.r,
            self.y + self.r,
        )

    def move_down(self, event):
        # move the circle down
        self.y += 1
        self.canvas.coords(
            self.circle,
            self.x - self.r,
            self.y - self.r,
            self.x + self.r,
            self.y + self.r,
        )

    def move_left(self, event):
        # move the circle left
        self.x -= 1
        self.canvas.coords(
            self.circle,
            self.x - self.r,
            self.y - self.r,
            self.x + self.r,
            self.y + self.r,
        )

    def move_right(self, event):
        # move the circle right
        self.x += 1
        self.canvas.coords(
            self.circle,
            self.x - self.r,
            self.y - self.r,
            self.x + self.r,
            self.y + self.r,
        )

    def save_position(self, event):
        self.spaceclicks += 1  # count the number of space clicks

        # display the position of the circle center on the text widget
        self.app.log_message(
            "Center position of the drawed circle:" + str(self.x) + " " + str(self.y)
        )

        # updating the position of the circle center on the image
        self.circlecenter = (1 - 1 / self.spaceclicks) * self.circlecenter + (
            1 / self.spaceclicks
        ) * np.array([self.x, self.y])
        # keep 1 decimal places
        self.circlecenter = np.round(self.circlecenter, 0)

        # log the number of space clicks and the position of the circle center
        
        self.app.log_message("Number of space clicks:" + str(self.spaceclicks))
        self.app.log_message("Updated center position:" + str(self.circlecenter))    

        # save the position to the created txt file without overwriting previous position
        with open(self.circlecenterfile, "a") as f:
            f.write(str(self.x) + " " + str(self.y) + "\n")


# main
if __name__ == "__main__":
    root = tk.Tk()
    app = CenterDetector(
        master=root,
        folder="/home/zilong/Desktop/2D2P/Data/162_10072023",
        appfolder="/home/zilong/Desktop/2D2P/Data/162_10072023/APP",
    )
    app.mainloop()
    
    
