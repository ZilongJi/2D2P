#%%
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tifffile
from datetime import datetime, timedelta
from tqdm import tqdm
from scanimagetiffio.scanimagetiffio import SITiffIO

def getMeanTiff_randomsampling(S, frac=0.1):
    """
    get the mean frame by averaging over recording per steps
    Input:
        S: the SITiffIO object
        frac: the fraction of frames to average over
    Output:
        meanFrame
    """
    nframes = S.get_n_frames()
    
    #randomly select frac of frames to average over
    num = int(nframes*frac)
    print('Randomly select {} frames to average over...'.format(num))
    indexs = np.random.choice(nframes, num, replace=False)

    frames = np.zeros((num, S.get_frame(1).shape[0], S.get_frame(1).shape[1]))
    for i, ind in enumerate(indexs):
        frames[i,:,:] = S.get_frame(ind+1)

    #average over the frames
    meanFrame = np.mean(frames, axis=0)
    
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
    
    return meanFrame

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
        print('WARNING: some bins have less than 50 elements')

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

def UnrotateFrame_SITiffIO(tiffpath, relogfile, rotCenter=[256,256]): 
    """
    Unrotate each frame with the corresponding rotation angle, as well as a rotation center
    using SITiffIO

    Args:
        tiffpath (_type_): _description_
        relogfile (_type_): _description_
        rotCenter (list, optional): _description_. Defaults to [256,256].
    """
    
    #read the tiff file using SITiffIO
    S = SITiffIO()
    S.open_tiff_file(tiffpath, "r")
    S.open_rotary_file(relogfile)
    S.interp_times()  # might take a while...
    
    N = S.get_n_frames() #number of frames
    RotAngles = S.get_all_theta() #all rotation angles
    
    #for each frame, unrotate it with the corresponding rotation angle using Image.rotate
    #and store in a list
    UnrotatedFrames = []
    for i in range(N):
        frame = S.get_frame(i+1)
        #unrotate the frame
        unrotatedFrame = Image.fromarray(frame).rotate(RotAngles[i], center=rotCenter)
        #crop the largest inner rectangle from the unrotated frame
        unrotatedFrame = cropLargestRecT(unrotatedFrame, rotCenter)
        #convert the PIL image back to int16
        unrotatedFrame = np.array(unrotatedFrame, dtype=np.int16)
        UnrotatedFrames.append(unrotatedFrame)       
    
    return UnrotatedFrames
    
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
