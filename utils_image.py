#%%
import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import time
import tifffile
from datetime import datetime, timedelta

import suite2p
from suite2p.registration import register, rigid

from scanimagetiffio import SITiffIO
from utils_io import get_imaging_files, get_rotary_center

def getMeanTiff_randomsampling(S, frac=0.1):
    """
    get the mean frame by averaging over recording per steps
    get the median value of the mean frame if return_median=True, for histogram matching purpose
    Input:
        S: the SITiffIO object
        frac: the fraction of frames to average over
    Output:
        meanFrame (np.array, uint8): the mean frame
        median_val (float): the median value of the mean frame (only returned if return_median=True)
    """
    nframes = S.get_n_frames()
    
    #randomly select frac of frames to average over
    num = int(nframes*frac)
    print('Randomly select {} frames to average over...'.format(num))
    indexs = np.random.choice(nframes, num, replace=False)

    meanframe = np.zeros(S.get_frame(1).shape, dtype=np.float64)
    for i, ind in enumerate(indexs):
        meanframe += S.get_frame(ind+1)/num
    
    #get the median value of the mean frame
    median_val = np.median(meanframe)
    
    #set the border (10 pixels) to median value
    #othersiwse the border will be very bright and dominate the mean value
    meanframe[:10,:] = median_val
    meanframe[-10:,:] = median_val
    meanframe[:,:10] = median_val
    meanframe[:,-10:] = median_val

    return meanframe.astype(np.int16)
    
def getMeanTiff_equalsampling(S, numBins):
    """
    get the mean frame by sampling frames equally from each angle bin

    Args:
        S (_type_): _description_
        numBins (_type_): _description_
    """
    
    angles = S.get_all_theta()
    
    #split 0-360 for numBins bins, each bin is 360/numBins degrees
    #find the number of elements of angles that fall into each bin

    #make a list of the bin edges
    bin_edges =  np.linspace(0,360,numBins+1)

    #make a list of the number of elements in each bin
    bin_counts = np.histogram(angles, bins=bin_edges)[0]

    #find the minimum in the list of bin counts
    #and through a warning if the minimum is less than 10
    if np.min(bin_counts) < 50:
        print('WARNING: some bins have less than 50 elements, with the least is: ', np.min(bin_counts))

    #for each bin, randomly sample min(bin_counts) elements from the bin
    #and save all the indices in a list
    bin_sample_list = []
    for i in range(numBins):
        #find the indices of the elements in the bin
        bin_indices = np.where((angles > bin_edges[i]) & (angles < bin_edges[i+1]))[0]
        #randomly sample np.min(bin_counts) elements from the bin without replacement, and convert to list
        bin_sample =  np.random.choice(bin_indices, np.min(bin_counts), replace=False).tolist()
        #add bin_sample into bin_sample_list
        bin_sample_list += bin_sample
    
    #extract frames from S accroding to the bin_sample_list
    #and average them all together
    #to save space,, we running average the frames

    #initialize the running average
    meanframe = np.zeros(S.get_frame(1).shape, dtype=np.float64)
    #loop through the bin_sample_list
    for i in bin_sample_list:
        if i==0:
            continue
        #add the frame to the running average devide by the number of frames
        meanframe += S.get_frame(i)/len(bin_sample_list)

    #get the median value of the mean frame
    median_val = np.median(meanframe)   
    
    #set the border (10 pixels) to median value
    #othersiwse the border will be very bright and dominate the normalization
    meanframe[:10,:] = median_val
    meanframe[-10:,:] = median_val
    meanframe[:,:10] = median_val
    meanframe[:,-10:] = median_val

    return meanframe.astype(np.int16)

def UnrotateCropFrame(Array, Angle, rotCenter): 
    """
    Unrotate each frame with the corresponding rotation angle and the rotation center 
    and crop the largest inner rectangle from the unrotated frame

    Args:
        Array (array): a 3D array of frames
        Angle (list): a list of rotation angles
        rotCenter (list): the rotation center from the center detection module
    """
    
    #for each frame, unrotate it with the corresponding rotation angle using Image.rotate
    #and store in an array
    NewFrames = []
    
    for i in range(Array.shape[0]):
        frame = Array[i,:,:]
        #unrotate the frame
        unrotatedFrame = Image.fromarray(frame).rotate(Angle[i], center=rotCenter)
        #crop the largest inner rectangle from the unrotated frame
        croppedFrame = cropLargestRecT(unrotatedFrame, rotCenter)
        #convert the PIL image back to int16
        croppedFrame = np.array(croppedFrame, dtype=np.int16)
        NewFrames.append(croppedFrame)      
        
    #convert the list to UnrotatedFrames to array
    NewFrames = np.array(NewFrames)
    
    return NewFrames

def cropLargestRecT(img, cropcenter):
    """
    crop the largest inner rectangle from PIL image under rotation
    Input: 
        img, PIL Image
        cropcenter: the image crop center
    Output:
        cropImg
    """
    width, height = img.size
    centerH, centerV = cropcenter
    minval = min(centerH-0, width-centerH, centerV-0, height-centerV)
    #size of the largest inner rectangle is np.sqrt(2)*minval
    maxHalfSize = np.floor(np.sqrt(2)*minval/2)
    left = centerH-maxHalfSize; right = centerH+maxHalfSize
    upper = centerV-maxHalfSize; lower = centerV+maxHalfSize
    cropImg = img.crop((left, upper, right, lower))
    return cropImg

def RegFrame(frames):
    '''
    Perform image registration on the frames
    Input:
        frames: a list of frames
    Output:
        mean_img: the mean image of the registered frames
    '''
    
    # prepare configurations using the ops dictionary (need to be add to the GUI later)
    ops = suite2p.default_ops()
    ops['batch_size'] = 200 # we will decrease the batch_size in case low RAM on computer
    ops['block_size'] = [64,64]
    ops['fs'] = 30 # sampling rate of recording, determines binning for cell detection
    ops['tau'] = 0.7 # timescale of gcamp to use for deconvolution

    #perform image registration with built-in suite2p function
    regframes = np.zeros_like(frames)
    output = register.compute_reference_and_register_frames(frames, f_align_out=regframes, refImg=None, ops=ops)
    
    refImg, rmin, rmax, mean_img, rigid_offsets, nonrigid_offsets, zest = output
    
    return mean_img, regframes

def compute_zpos_sp(Zstack, regFrames, ops):
    """
    compute z position of registrated frames gievb z-stacks, adapted from suite2p code:
    
    https://github.com/MouseLand/suite2p/blob/main/suite2p/registration/zalign.py

    Args:
        Zstack (3D array): size [nplanes *Ly*Lx] 
        regFrames (3D array): size [nframes *Ly*Lx]
        ops (dict): default_ops
    Returns:
        ymax (2D array): size [nplanes *nframes], the y position of the maximum correlation
        xmax (2D array): size [nplanes *nframes], the x position of the maximum correlation
        zcorr (2D array): size [nplanes *nframes], the maximum correlation
    """
    
    nplanes, zLy, zLx = Zstack.shape
    nFrames, Ly, Lx = regFrames.shape
    
    if nFrames>100:
        nbatch = 100 # number of frames to process at a time
    else:
        nbatch = nFrames
    
    refAndMasks = []
    for Z in Zstack:
        maskMul, maskOffset = rigid.compute_masks(
            refImg=Z,
            maskSlope=3*ops['smooth_sigma'],
        )
        
        cfRefImag = rigid.phasecorr_reference(
            refImg=Z,
            smooth_sigma=ops['smooth_sigma']
        )
        
        cfRefImag = cfRefImag[np.newaxis, :, :]
        
        refAndMasks.append((maskMul, maskOffset, cfRefImag))
    
    ymax = np.zeros((nplanes, nFrames), np.int32)
    xmax = np.zeros((nplanes, nFrames), np.int32)
    zcorr = np.zeros((nplanes, nFrames), np.float32)
    
    t0 = time.time()
    nfr = 0
    while True:
        data = np.float32(regFrames[nfr:nfr+nbatch])
        inds = np.arange(nfr, nfr + data.shape[0], 1, int)
        if (data.size == 0) | (nfr >= nFrames):
            break
        for z, ref in enumerate(refAndMasks):
            
            maskMul, maskOffset, cfRefImg = ref
            cfRefImg = cfRefImg.squeeze()
            
            ymax[z,inds],xmax[z,inds],zcorr[z, inds] = rigid.phasecorr(
                data = rigid.apply_masks(data=data, maskMul=maskMul, maskOffset=maskOffset),
                cfRefImg = cfRefImg,
                maxregshift = ops['maxregshift'],
                smooth_sigma_time=ops['smooth_sigma_time'],
            )
            if z%10 == 1:
                print("%d planes, %d/%d frames, %0.2f sec." %
                      (z, nfr, nFrames, time.time() - t0))
        
        print("%d planes, %d/%d frames, %0.2f sec." %
              (z, nfr, nFrames, time.time() - t0))
        nfr += data.shape[0]
    return ymax, xmax, zcorr

def get_meanZstack(S, volume, stacks, frames, Rotcenter, ImgReg=False):
    """
    get the mean frame of Zstacks
    Args:
        S [SITIFFIO]: SITIFFIO object
        volume [int]: volume of the Zstack
        stacks [int]: number of stacks
        frames [int]: number of frames
        Rotcenter [x,y list]: rotation center
        ImgReg [bool]: if True, do image registration
    Return:
        meanZstack [stacks*Ly*Lx]: the mean frame of Zstack
    """
    print("Extract the mean frame of Zstacks...")
    
    #count the time for the function
    t0 = time.time()

    Angles = S.get_all_theta()
    
    #figure out the size of the cropped image
    frame1 = S.get_frame(1)
    angle1 = Angles[0]
    UnrotFrame1 = Image.fromarray(frame1).rotate(angle1, center=Rotcenter)
    croppedFrame1 = cropLargestRecT(UnrotFrame1, Rotcenter)
    w, h = croppedFrame1.size
    
    #generate an empty array to store the mean frame of Zstack
    meanZstacks = np.zeros((stacks, h, w))
    
    #loop through all the stacks
    for stack_i in range(stacks):
        print("Processing stack {}".format(stack_i))
        #create a temporary array to store all the frames in the current stack
        temp_stack_frames = np.zeros((volume*frames, w, h), dtype=np.int16)
        
        #get all the index of the frames belonging to the current stack
        #init an empty array to store the index
        inds = np.zeros((volume*frames), dtype=np.int32)
        for vi in range(volume):
            ind_in_vloume = vi*stacks*frames+np.arange(stack_i*frames,(stack_i+1)*frames,1)
            inds[vi*frames:(vi+1)*frames] = ind_in_vloume
    
        #loop through the index and get the frames
        for i,ind in enumerate(inds):
            #get the frame, unrotate and crop
            frame_i = S.get_frame(ind+1)
            angle_i = Angles[ind]
            UnrotFrame_i = Image.fromarray(frame_i).rotate(angle_i, center=Rotcenter)
            croppedFrame_i = cropLargestRecT(UnrotFrame_i, Rotcenter)
            croppedFrame_i = np.array(croppedFrame_i, dtype=np.int16)

            #add the current frame to temp_stack
            temp_stack_frames[i] = croppedFrame_i
            
        #do image registration if ImgReg is True
        if ImgReg:
            meanImg, _ = RegFrame(temp_stack_frames)
        else:
            meanImg = np.mean(temp_stack_frames, axis=0)
        
        meanZstacks[stack_i] = np.int16(meanImg)
        
    print("Getting the mean Z stack frames -- Done! Time used: {}".format(time.time()-t0))
    
    return meanZstacks

def findFOV(zstacks, Img, maxrotangle=30):
    '''
    find the field of view for multi-day imaging
    Args:
        zstacks (3D array): size [nplanes, height, width] 
        Img (2D array): size [height, width]
        degreerange (list): range of rotation angle
    Returns:
        ymax (int): y position of the FOV
        xmax (int): x position of the FOV
        zcorr (float): z correlation
    '''
    ops = suite2p.default_ops()
    
    w, h = Img.shape
    
    #get the last two dimension the meanZ and meanRegImg
    _, wZ, hZ = zstacks.shape
    w, h = Img.shape
    if w>wZ:
        #cut the meanRegImg to the same size as meanZ, but with the same center
        #Img = Img[:, -wZ:, -hZ:]
        wpad = (w-wZ)//2
        hpad = (h-hZ)//2
        Img = Img[wpad:-wpad, hpad:-hpad]
    elif w<wZ:
        #center pad each edge of meanRegImg to the same size as meanZ
        wpad = (wZ-w)//2
        hpad = (hZ-h)//2
        Img = np.pad(Img, ((wpad, wpad), (hpad, hpad)), mode="constant", constant_values=0)
    else:
        pass
    
    neww, newh = Img.shape
    #create an empty array to store the rotated images
    rotImgs = np.zeros((2*maxrotangle+1, neww, newh))
    #all rotation angles
    rotAngles = np.arange(-maxrotangle, maxrotangle+1, 1)
    for i, ang in enumerate(rotAngles):
        rotImg = Image.fromarray(Img).rotate(ang)
        rotImg = np.array(rotImg)
        #save
        rotImgs[i] = rotImg
    
    #do phase correlation
    ymax, xmax, zcorr = compute_zpos_sp(zstacks, rotImgs, ops)
    
    return ymax, xmax, zcorr

def hist_match(source, template):
    """
    Adjust the pixel values of a grayscale image such that its histogram
    matches that of a target image

    Arguments:
    -----------
        source: np.ndarray
            Image to transform; the histogram is computed over the flattened
            array
        template: np.ndarray
            Template image; can have different dimensions to source
    Returns:
    -----------
        matched: np.ndarray
            The transformed output image
    """

    oldshape = source.shape
    source = source.ravel()
    template = template.ravel()

    # get the set of unique pixel values and their corresponding indices and
    # counts
    s_values, bin_idx, s_counts = np.unique(source, return_inverse=True,
                                            return_counts=True)
    t_values, t_counts = np.unique(template, return_counts=True)

    # take the cumsum of the counts and normalize by the number of pixels to
    # get the empirical cumulative distribution functions for the source and
    # template images (maps pixel value --> quantile)
    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles /= s_quantiles[-1]
    t_quantiles = np.cumsum(t_counts).astype(np.float64)
    t_quantiles /= t_quantiles[-1]

    # interpolate linearly to find the pixel values in the template image
    # that correspond most closely to the quantiles in the source image
    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)

    return interp_t_values[bin_idx].reshape(oldshape)

def UnrotateTiff(datafolder, namelist, readVRlogs=False):
    '''
    Unrotate all the imaging tiff files in namelist, crop the tiff files and save them into a folder named UnrotTiff
    Args:
        datafolder: the folder containing the data files
        namelist: a list of names, e.g., [00003, 00004, 00005]
    '''
    
    print('Unrotate Imaging Tiff files...')
    
    #1, create a folder named UnrotTiff under the datafolder folder to save data
    UnrotTiffFolder = os.path.join(datafolder, "UnrotTiff/")
    #if the folder exists, remove it, and then create a new one
    if os.path.exists(UnrotTiffFolder):
        shutil.rmtree(UnrotTiffFolder)
    os.makedirs(UnrotTiffFolder) 
    
    #2, get all the tiff files
    allfiles = get_imaging_files(datafolder, namelist, readVRlogs=False)
    
    #3, get the rotary center from the centerfile
    circlecenterfilename = os.path.join(datafolder, "DP_exp/circlecenter.txt")
    #through out an error if the file does not exist
    if not os.path.exists(circlecenterfilename):
        raise ValueError("The rotary center file does not exist! Make sure DP_exp folder exists and the rotary center file is in it!")
    
    rotCenter = get_rotary_center(circlecenterfilename)
    print("Rotation center is at ({}, {})".format(rotCenter[0], rotCenter[1]))

    #4, get the median value of the mean frame for each tiff file for later histogram matching
    
    #open a txt file to store the median values
    with open(UnrotTiffFolder+"medianVals.txt", "w") as f:
        f.write("filename\tmedian\n")
    
    all_medians = []
    for i, (tiff, _)  in enumerate(allfiles):
        print('Get the median value of tiff file: ', tiff)
        S = SITiffIO()
        S.open_tiff_file(tiff, "r")
        meanFrame = getMeanTiff_randomsampling(S, frac=0.01)
        medianVal = np.median(meanFrame)
        #store the median value
        all_medians.append(medianVal)
        #save the median values into a txt file under UnrotTiffFolder
        with open(UnrotTiffFolder+"medianVals.txt", "a") as f:
            f.write(tiff.split("/")[-1].split(".")[0]+"\t"+str(medianVal)+"\n")
    
    #get the histgram offset 
    histoffset = [all_medians[0]-all_medians[i] for i in range(len(all_medians))]
        
    #5, for each pair of tiff file and RElog file, unrotate the tiff file, match the histogram, and save it into the UnrotTiff folder
    for i, (tiff_file, RElog_file) in enumerate(allfiles):
        
        #time
        t = time.time()

        #geterante the unrotated tiff file name
        unrotated_tiff_file = UnrotTiffFolder+tiff_file.split("/")[-1].split(".")[0] + "_unrot.tif"
        #print processing file name
        print("Unrotating tiff file: " + tiff_file)

        #get histogram offset
        offset = histoffset[i]
    
        #unrotate the tiff file and save it into the UnrotTiff folder
        S = SITiffIO()
        S.open_tiff_file(tiff_file, "r")
        S.open_tiff_file(unrotated_tiff_file, "w") #for writing
        S.open_rotary_file(RElog_file)
        S.interp_times()  # might take a while...
        
        N = S.get_n_frames()
        Alltheta = S.get_all_theta()

        for i in range(N):
            frame_i = S.get_frame(i+1)
            theta_i = Alltheta[i]
            #unrotate the frame
            unrotatedFrame = Image.fromarray(frame_i).rotate(theta_i, center=rotCenter)
            #crop the largest inner rectangle from the unrotated frame
            croppedFrame = cropLargestRecT(unrotatedFrame, rotCenter)
            #convert the PIL image back to int16
            croppedFrame = np.array(croppedFrame, dtype=np.int16)+offset

            S.write_frame(croppedFrame, i)
        
        #time
        print("Time for unrotating current tiff file: " + str(time.time()-t))

#%%
if __name__ == "__main__":
    pass