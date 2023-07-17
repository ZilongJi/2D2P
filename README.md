# 2D2P
Pipeline for processing **2 D**imensional virtual reality **2 P**hoton (**2D2P**) calcium imaging data.

2D2P includes the following modules so far:
- rotation center detection
- zdrift calculation

## Installation (only tested for linux system)
1. Install a [Anaconda](https://www.anaconda.com/download#linux) distribution of python
2. Create a new environment with `conda create --name 2D2P python=3.8`
3. Activate the new environment by run `conda activate 2D2P`
4. Create a folder named `2D2P` on your computer and `cd 2D2P`
5. Install [ScanImageTiffIO](https://github.com/rhayman/ScanImageTiffIO) (following the instructions there. Note that `libtiff` will be installed automatically when do cmake)
6. Git clone [2D2P](https://github.com/ZilongJi/2D2P) and install all the dependencies (will create a pip install later..)
7. I use vscode to code which can be obtained here: [vscode](https://code.visualstudio.com/docs/setup/linux)

## Data acquisition flowchart
<img src="https://github.com/ZilongJi/2D2P/blob/main/Figures/2DVR2P%20Flowchart.png" width=50% height=50%>
Click to zoom in for details. 

## Using the GUI
<img src="https://github.com/ZilongJi/2D2P/blob/main/Figures/2D2PAPP.png" width=50% height=50%>

## Troubleshooting
- [ ] 15/07/2023: Analysis pipeline tested. But z-drift correlation does not work. Need to add image registration to unrotated tiff. 
