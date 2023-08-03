#%%
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import time
import tifffile
from datetime import datetime, timedelta
import suite2p
from suite2p.registration import register, rigid
from scanimagetiffio import SITiffIO

def getMeanTiff_randomsampling(S, frac=0.1, return_median=False):
    """
    get the mean frame by averaging over recording per steps
    get the median value of the mean frame if return_median=True, for histogram matching purpose
    Input:
        S: the SITiffIO object
        frac: the fraction of frames to average over
        return_median: whether to return the median value of the mean frame
    Output:
        meanFrame (np.array, uint8): the mean frame
        median_val (float): the median value of the mean frame (only returned if return_median=True)
    """
    nframes = S.get_n_frames()
    
    #randomly select frac of frames to average over
    num = int(nframes*frac)
    print('Randomly select {} frames to average over...'.format(num))
    indexs = np.random.choice(nframes, num, replace=False)

    meanframe = np.zeros((S.get_frame(1).shape[0], S.get_frame(1).shape[1]))
    for i, ind in enumerate(indexs):
        meanframe += S.get_frame(ind+1)/num
    
    #get the median value of the mean frame
    median_val = np.median(meanFrame)
    
    '''
    #set the border (10 pixels) to median value
    #othersiwse the border will be very bright and dominate the normalization
    meanFrame[:10,:] = np.median(meanFrame)
    meanFrame[-10:,:] = np.median(meanFrame)
    meanFrame[:,:10] = np.median(meanFrame)
    meanFrame[:,-10:] = np.median(meanFrame)
    
    #normalize the meanFrame to 0-255
    meanFrame = meanFrame - np.min(meanFrame)
    meanFrame = meanFrame/np.max(meanFrame)
    meanFrame = meanFrame*255
    
    #change to uint8
    meanFrame = np.uint8(meanFrame)
    '''
    
    if return_median:
        return meanframe.astype(np.int16), median_val
    else:
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
    #make a list of the bin centers
    bin_centers = (bin_edges[:-1] + bin_edges[1:])/2

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
    meanframe = np.zeros(S.get_frame(1).shape)
    #loop through the bin_sample_list
    for i in bin_sample_list:
        #get the frame
        frame = S.get_frame(i)
        #add the frame to the running average devide by the number of frames
        meanframe += frame/len(bin_sample_list)

    #set the border (10 pixels) to median value
    #othersiwse the border will be very bright and dominate the normalization
    meanframe[:10,:] = np.median(meanframe)
    meanframe[-10:,:] = np.median(meanframe)
    meanframe[:,:10] = np.median(meanframe)
    meanframe[:,-10:] = np.median(meanframe)

    #normalize the running_average to 0-255
    meanframe = meanframe - np.min(meanframe)
    meanframe = meanframe/np.max(meanframe)
    meanframe = meanframe*255

    #change to uint8
    meanframe = np.uint8(meanframe)
    
    return meanframe
    

def getFramesandTimeStamps(tiffpath):
    """
    #get the acquisition time of each frame in the tiff file using tifffile
    #get each frame as well
    tiffpath: the path to the tiff file
    """
    
    Timestamps = []
    Frames = []
    Epochs = []
    with tifffile.TiffFile(tiffpath) as tif:
        for page in tif.pages:
            #get each frame
            Frames.append(page.asarray())
            
            #get the acquisition time of each frame
            desp = page.tags['ImageDescription'].value
            #desp is a string, need to convert to list
            #I want to find the value of the key "frameTimestamps_sec"
            #and the value of the key "epoch" 
            desp = desp.split('\n')
            for line in desp:
                if line.startswith('frameTimestamps_sec'):
                    frametimestamps = line.split('=')[1]
                    #convert string to float and store in a list
                    frametimestamps = float(frametimestamps)
                    #frametimestamps represents seconds, convert to datetime object
                    frametimestamps = timedelta(seconds=frametimestamps)
                    Timestamps.append(frametimestamps)
                if line.startswith('epoch'):
                    epoch = line.split('=')[1]
                    #change epoch which is a list like [2023  6 19 16 58 56.387] to Datetime object
                    epoch = datetime.strptime(epoch, ' [%Y  %m %d %H %M %S.%f]')
                    Epochs.append(epoch)

    #For each element in Timestamps, add the corresponding epoch to it
    #and store in a list
    acquistionTime = [Timestamps[i] + Epochs[i] for i in range(len(Timestamps))]
    
    return Frames, acquistionTime

def getTimeandRotAngle(tiffpath, relogfile):
    """
    Get the time stamp and rotation angle from the log file
    Args:
        tiffpath (_type_): _description_
        relogfile (_type_): _description_
    """
    #load the relog txt file and read the lines
    #each line looks like: 2023-06-19 16:57:22.108 Rot=247.089130 Trigger=0.000000
    #save the time (beofre first space ) 
    #and rotation angle (value after Rot=) in two lists
    RETimeStamps = []
    RERotAngles = []
    with open(relogfile, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.split(' ')
            #get the time stamp
            RETimeStamp = line[0] + ' ' + line[1]
            #convert to datetime object
            RETimeStamp = datetime.strptime(RETimeStamp, '%Y-%m-%d %H:%M:%S.%f')
            RETimeStamps.append(RETimeStamp)
            
            #get the rotation angle
            RERotAngle = line[2].split('=')[1]
            #convert to float
            RERotAngle = float(RERotAngle)
            RERotAngles.append(RERotAngle)   
    
    return RETimeStamps, RERotAngles
    

def getRotAngle(acquistionTimeStamps, RETimeStamps, RERotAngles):
    """
    get the rotation angle for each of the frame
    Args:
        acquistionTimeStamps (list): _description_
        RETimeStamps (list): _description_
        RERotAngles (list): _description_
    """
    #for each element in acquistionTimeStamps, find the closest element in RETimeStamps
    #and get the corresponding rotation angle
    RotAngles = []
    for i in range(len(acquistionTimeStamps)):
        #find the closest element in RETimeStamps
        #find the index of the closest element in RETimeStamps
        idx = (np.abs(np.array(RETimeStamps) - acquistionTimeStamps[i])).argmin()
        #get the corresponding rotation angle
        RotAngle = RERotAngles[idx]
        RotAngles.append(RotAngle)
    
    return RotAngles

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
    
def UnrotateFrame_tiffile(tiffpath, relogfile, rotCenter=[256,256]):
    """
    Unrotate each frame with the corresponding rotation angle, as well as a rotation center
    using python tifffile package (much slower than SITiffIO)
    
    Args:
        tiffpath: the path to the tiff file
        relogfile: the path to the relog file
        rotCenter: the rotation center, default is image center which is [256,256]
    """
 
     #get the acquisition time of each frame in the tiff file using tifffile
    Frames, acquistionTimeStamps = getFramesandTimeStamps(tiffpath)
    
    #get the time stamp and rotation angle from the log file
    RETimeStamps, RERotAngles = getTimeandRotAngle(tiffpath, relogfile)
    
    #get the rotation angle for each of the frame
    RotAngles = getRotAngle(acquistionTimeStamps, RETimeStamps, RERotAngles)
    
    #for each frame in Frames, unrotate it with the corresponding rotation angle using Image.rotate
    #and store in a list
    UnrotatedFrames = []
    for i in range(len(Frames)):
        UnrotatedFrame = Image.fromarray(Frames[i]).rotate(RotAngles[i], center=rotCenter)
        #crop the largest inner rectangle from the unrotated frame
        UnrotatedFrame = cropLargestRecT(UnrotatedFrame, rotCenter)
        #convert the PIL image back to int16
        UnrotatedFrame = np.array(UnrotatedFrame, dtype=np.int16)
        UnrotatedFrames.append(UnrotatedFrame)
    
    return UnrotatedFrames

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

#%%
if __name__ == "__main__":
    #%% read the imaging tiff and display the mean frame
    #data folder

    dataFolder = '/home/zilong/Desktop/2PAnalysis/2PData/from_Guifen/162_REtest/'
    tifffilepath = dataFolder+ 'imaging_testing_162_00001.tif'
    vrlogfile = dataFolder + '20230524-144206.06.txt'
    relogfile = dataFolder + 'REdata_20230524_151401.txt'

    # dataFolder = '/home/zilong/Desktop/2PAnalysis/2PData/from_Guifen/162_test2_2blocks_19062023/'
    # tifffilepath = dataFolder+ '162_block1_19062023__00001.tif'
    # vrlogfile = dataFolder + '20230619-162203.03.txt'
    # relogfile = dataFolder + 'REdata_20230619_170256.txt'

    #read the data
    S = SITiffIO()
    S.open_tiff_file(tifffilepath, 'r')
    S.open_log_file(vrlogfile)
    S.open_rotary_file(relogfile)
    S.interp_times() # might take a while... 
    
    #%%
    meanframe = getMeanTiff(S, frac=0.1)
    #display the mean frame

    plt.imshow(meanframe, cmap='gray')  
    
    #%% read a smaller tiff file and calculate the rotation angle of each frame
    dataFolder = '/home/zilong/Desktop/2PAnalysis/2PData/from_Guifen/162_test2_2blocks_19062023/'
    tiffpath = dataFolder+ '162_zstack_19062023__00002.tif'
    relogfile = dataFolder + 'REdata_20230619_165713.txt'

    #%% unrotate each frame with tifffile
    newFrames_Tifffile = UnrotateFrame_tiffile(tiffpath, relogfile, rotCenter=[256,256])
    
    #%% unrotate each frame with SITiffIO
    newFrames_SITiffIO = UnrotateFrame_SITiffIO(tiffpath, relogfile, rotCenter=[256,256])
    
    #%% display the mean frame
    meanframe = np.mean(newFrames_Tifffile, axis=0)
    plt.imshow(meanframe, cmap='gray')
    
    #reshape the newFrames to [5,41,10, width, height]
    newFrames_array = np.array(newFrames_Tifffile)
    newFrames_array = newFrames_array.reshape([5,41,10,newFrames_array.shape[1], newFrames_array.shape[2]])

    #average across the first and third dimension
    meanStacks = np.mean(newFrames_array, axis=(0,2))
    #display images in the meanStacks
    for i in range(meanStacks.shape[0]):
        plt.imshow(meanStacks[i,:,:], cmap='gray')
        plt.show()  
    #%% 
    dataFolder = '/home/zilong/Desktop/2PAnalysis/2PData/from_Guifen/162_test2_2blocks_19062023/'
    tiffpath = dataFolder+ '162_block1_19062023__00001.tif'
    relogfile = dataFolder + 'REdata_20230619_170256.txt' 
    
    #get the acquisition time of each frame in the tiff file using tifffile
    Frames, acquistionTimeStamps = getFramesandTimeStamps(tiffpath) 
    
    

# %%
