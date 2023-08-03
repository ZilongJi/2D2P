import os
import glob

def get_imaging_files(datafolder, namelist, readVRlogs=True):
    '''
    get the triple of data files
    Args:
        datafolder: the folder containing the data files
        namelist: a list of names, e.g., [00003, 00004, 00005]
    Returns:
        A dictionary, with each element as a list of two files (readVRlogs=False) or three files (readVRlogs=True)
        1, tiff file
        2, RELog file
        3, VRLog file
    '''
    #get all the tiff files 
    tifffiles = glob.glob(datafolder + "/*.tif")
    #remove tifffiles with key words "stack"
    tifffiles = [x for x in tifffiles if "stack" not in x]
    #then only keep tiff files with the names in namelist
    tifffiles = [x for x in tifffiles if any(y in x for y in namelist)]
    #sort the tifffiles by the number in the file name
    tifffiles.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    #get all the RElog files under the parent folder begin with 'RE' and with an extension of .txt
    RElogfiles = glob.glob(datafolder + "/RE*.txt")
    #remove RElogfiles with key words "stack"
    RElogfiles = [x for x in RElogfiles if "stack" not in x]

    #get all the VRlog files under the parent folder begin with numbers and with an extension of .txt
    VRlogfiles = glob.glob(datafolder + "/[0-9]*.txt")

    #pair the tiff files and RElog files together which share the same key word
    allfiles = []
    for tifffile in tifffiles:
        #extract the key word from the tifffile
        #for example, '/home/zilong/Desktop/2D2P/Data/183_25072023/25072023_00005.tif' then extract '00005'
        key = tifffile.split("/")[-1].split(".")[0].split("_")[-1]
        #find the RElogfile containing the key word
        RElogfile = [x for x in RElogfiles if key in x][0]
        if readVRlogs:
            #find the VRlogfile containing the key word
            VRlogfile = [x for x in VRlogfiles if key in x][0]
            #pair the tifffile and RElogfile together
            pair = [tifffile, RElogfile, VRlogfile]
        else:
            pair = [tifffile, RElogfile]
            
        #append the pair to allfiles
        allfiles.append(pair)
        
    return allfiles

def get_rotary_center(centerfile):
    '''
    get the rotary center from the centerfile
    Args:
        centerfile: the file containing the rotary center
    Returns:
        the rotary center 
    '''
    
    with open(centerfile, "r") as f:
        # read the last row
        last_line = f.readlines()[-1]
        # assign the x and y coordinates to self.rotx and self.roty
        rotx = float(last_line.split()[0])
        roty = float(last_line.split()[1])
    
    rotCenter = [rotx, roty]
    
    return rotCenter
    
    
    
