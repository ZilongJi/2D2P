import os
import cv2
import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

from utils_io import get_imaging_files
from scanimagetiffio import SITiffIO   

def plot_trajectory(datafolder, filenamelist):
    '''
    Plot the position of the particles in the xy plane.
    '''
    
    #get file triples, including Tiff file, VR logs and RE logs
    allfiles = get_imaging_files(datafolder, filenamelist, readVRlogs=True)
    
    #creata allX and allZ allTime and allTheta as a dictionary
    allX = {}
    allZ = {}
    allTime = {}
    allTheta = {}

    for session in range(len(allfiles)):
        tifffile, relogfile, vrlogfile  = allfiles[session]
        print ('processing: \n' + tifffile.split('/')[-1] + '\n' + relogfile.split('/')[-1] + '\n' + vrlogfile.split('/')[-1]+'...')
        
        S = SITiffIO()
        S.open_tiff_file(tifffile, "r")
        S.open_rotary_file(relogfile)
        S.open_log_file(vrlogfile)
        S.interp_times()  # might take a while...
        
        X = S.get_all_raw_x()
        Z = S.get_all_raw_z()
        time =S.get_tiff_times()
        theta = S.get_all_theta()
        
        #add the position and time to the dictionary
        allX[session] = X
        allZ[session] = Z
        allTime[session] = time
        allTheta[session] = theta
        
    #do plot
    fig = plt.figure(figsize=(8, 4), dpi=300)

    #choose colormap when plot, equally sampled from tab20c accodring to the number of sessions
    colors = plt.cm.tab20c(np.linspace(0, 1, len(allX)))

    #plot the trajectory of each session in subplot 1, but with different colors
    ax1 = fig.add_subplot(121, projection='3d')
    
    for i in range(len(allX)):
        ax1.plot(allX[i], allZ[i], i,alpha = 0.5, linewidth = 1, color = colors[i])
    #square the plot
    ax1.set_aspect('equal', 'box')
    #xlabel and ylabel and zlabel
    ax1.set_xlabel('VR X')
    ax1.set_ylabel('VR Y')
    ax1.set_zlabel('Session ID')
    #set zticks to integers, every 2 integers
    ax1.set_zticks(range(len(allX))[::2])
    #change the view angle
    ax1.view_init(20, 40)


    #plot the merged trajectory (allX, allZ) in subplot 2, but with different colors
    ax2 = fig.add_subplot(122)
    for i in range(len(allX)):
        label_i = str(i)+' ('+str(len(allX[i]))+')'
        ax2.plot(allX[i], allZ[i], label = label_i, alpha = 0.5, linewidth = 1, color = colors[i])
    #put the label out of the plot
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    #square the plot
    ax2.set_aspect('equal', 'box')
    #xlabel and ylabel
    ax2.set_xlabel('VR X')
    ax2.set_ylabel('VR Y')

    #tight layout
    fig.tight_layout()

    #create a folder under the datafolder/UnrotTiff named 2D2P to save the plot
    savefolder = os.path.join(datafolder, 'UnrotTiff/', '2D2P')
    if not os.path.exists(savefolder):
        os.makedirs(savefolder)
    #save the plot
    plt.savefig(os.path.join(savefolder, 'trajectory.png'))
    
    #save allX, allZ, allTime to pickle file
    trajectory = []
    trajectory.append(allX)
    trajectory.append(allZ)
    trajectory.append(allTime)
    trajectory.append(allTheta)

    with open(os.path.join(savefolder,'trajectory.pickle'), 'wb') as f:
        pickle.dump(trajectory, f)
        
    return allX, allZ, allTime, allTheta, fig
    
def BoxCarSmooth(signal, win=5, boxtype='backward'):
    '''
    boxcar smooth of thesignal
    '''
    N = len(signal)
    smoothSignal = np.zeros_like(signal)
    if boxtype=='backward':
        #pad signal with zero on at the beginning
        signal = np.pad(signal, pad_width=(2*win,0), mode='edge')
    elif boxtype=='center':
        #pad signal with zero on at the beginning and the end
        signal = np.pad(signal, pad_width=(win,win), mode='edge')    
    else:
        ValueError("choose correct smoothing type")
        
    for i in range(N):
        smoothSignal[i] = np.mean(signal[i:i+2*win+1])
    return smoothSignal  

def getLinearSpeed(X, Z, timestamps, boxcar_size=5):
    '''
    calculate the linear speed of the animal
    '''
    #smooth x and z for calculating the speed below
    SmoothX = BoxCarSmooth(X, win=boxcar_size, boxtype='center')
    SmoothZ = BoxCarSmooth(Z, win=boxcar_size, boxtype='center')
    diffX = np.diff(SmoothX)
    diffZ = np.diff(SmoothZ)
    #calculate the velocity along X and Z
    diffTimeStamps = np.diff(timestamps) 
    vX = np.divide(diffX, diffTimeStamps)
    vZ = np.divide(diffZ, diffTimeStamps)

    vX = np.concatenate((np.asarray([0]), vX))  
    vZ = np.concatenate((np.asarray([0]), vZ))
    #speed
    linearspeed = np.sqrt(vX**2+vZ**2)
    
    return linearspeed

def getAngularSpeed(theta, timestamps):
    '''
    calculate the angular speed of the animal
    Input:
        theta: the rotation angle of the animal
        timestamps: the time stamps of each frame
    Output:
        angspeed: the angular speed of the animal
    '''
    
    #calculate the difference of successive theta values
    diffTheta = np.diff(theta) 
    
    #note that all thetas are wrapped to [0,2pi], so we should deal with the boundary
    #it is impossible that diff value is larger than pi (or smaller than -pi)
    #otherwise it will be a too quick rotation
    clockCrossBoundaryIdx = diffTheta<-np.pi
    anticlockCrossBoundaryIdx = diffTheta>np.pi

    #1, therefore, for clockwise rotation, e.g., 6.20-->0.3, we replace 0.3-6.20 with 0.3-6.20+2*pi
    diffTheta[clockCrossBoundaryIdx] = diffTheta[clockCrossBoundaryIdx]+2*np.pi
    #2, for counter clockwise rotation, e.g., 0.3-->6.20, we replace 6.20-0.3 with 6.20-0.3-2*pi
    diffTheta[anticlockCrossBoundaryIdx] = diffTheta[anticlockCrossBoundaryIdx]-2*np.pi  
    
    #calculate the difference of successive time points
    diffTimeStamps = np.diff(timestamps)
    
    angspeed = np.divide(diffTheta, diffTimeStamps)
    #add zero at the begining to make the dimension consistant
    angspeed = np.concatenate((np.asarray([0]), angspeed))    
    
    return angspeed
    
     
    
def getTuningMap(spks, X, Z, timestamps, VRsize=(1,1), binsize=(0.025,0.025), 
                 sigma=3/2.5, speed_thres=0.025, boxcar_size=5, visit_thres=0.1,
                 peak_thres=100, cell_id = None, datafolder=None):
    '''
    plot the neural firing tuning map of a cell
    Input:
        spks: deconvolved calcium activity
        X: x-position (np.array)
        Z: y-position (np.array)
        timestamps: (np.array)
        VRsize: size of the virtual reality box (tuple)
        binsize: size of the bins (tuple)
        sigma: sigma of the Gaussian filter (float)
        speed_thres: speed threshold (float)
        boxcar_size: size of the boxcar window for smoothing the speed (int)
        visit_thres: time threshold for a bin to be considered visited (float)
    Output:
        Firing_Rate_In_Position: the firing rate map (np.array)
    Hyperparameter setting is inspired by this paper (Zong et al, 2022, Cell): https://www.sciencedirect.com/science/article/pii/S0092867422001970
    '''
    #1, calculate the smoothed moving speed
    linearspeed = getLinearSpeed(X, Z, timestamps, boxcar_size=boxcar_size)
    
    #2, removing period when animal's speed is below a threshold
    
    ind_thres = linearspeed>=speed_thres
    
    spks = spks[ind_thres]
    X = X[ind_thres]
    Z = Z[ind_thres]
    #calculate diffTimeStamps and add 0 at the beginning to make sure dimension does not change in one line
    diffTimeStamps =  np.concatenate((np.asarray([0]), np.diff(timestamps)))
    diffTimeStamps = diffTimeStamps[ind_thres]
    
    #removing edge effect
    X[X==1]=1-1e-5
    Z[Z==1]=1-1e-5
    Pos_X = np.int32(X/binsize[0])
    Pos_Z = np.int32(Z/binsize[1])    
    
    # Here, the total time spent moving in each position bin 
    # and the sumed amplitude of the deconvolved calcium activity in each position bin
    # are calculated
    Time_In_Position = np.zeros((np.int32(VRsize[0]/binsize[0]), np.int32(VRsize[1]/binsize[1])))
    Spks_In_Position = np.zeros_like(Time_In_Position)
    total_y_bins = np.int32(VRsize[0]/binsize[0])
    for i in range(len(X)):
        # Time_In_Position[total_y_bins-1-Pos_Z[i],Pos_X[i]] += diffTimeStamps[i]
        # Spks_In_Position[total_y_bins-1-Pos_Z[i],Pos_X[i]] += spks[i]
        Time_In_Position[Pos_Z[i],Pos_X[i]] += diffTimeStamps[i]
        Spks_In_Position[Pos_Z[i],Pos_X[i]] += spks[i]
           
    #Here, I calculate the actual firing rate (calcium activty/time) and save it as Field_Data
    Firing_Rate_In_Position = Spks_In_Position/(Time_In_Position+1e-5)
    
    #smooth the firing rate matrix with a Gaussian filter
    Firing_Rate_In_Position = gaussian_filter(Firing_Rate_In_Position, sigma=sigma)
    
    #find Time_In_Position<visit_thres and set Firing_Rate_In_Position to np.nan
    Firing_Rate_In_Position[Time_In_Position<visit_thres]=np.nan
    
    #plot the map only plot those peak values larger than peak_thres
    if np.nanmax(Firing_Rate_In_Position)>peak_thres:
        plt.figure(figsize=(3, 3), dpi=300)
        plt.imshow(Firing_Rate_In_Position, cmap='inferno')
        plt.xlabel('VR X')
        plt.ylabel('VR Y')
        plt.xticks(np.linspace(0, Firing_Rate_In_Position.shape[0], 5), np.linspace(0, 1, 5))
        plt.yticks(np.linspace(0, Firing_Rate_In_Position.shape[1], 5), np.linspace(0, 1, 5))
        plt.colorbar(label='Calcium activity')
        
        #save the figure
        savefolder = os.path.join(datafolder, 'UnrotTiff/', '2D2P/', 'firingmaps/')
        if not os.path.exists(savefolder):
            os.makedirs(savefolder)
        #save the plot
        plt.savefig(os.path.join(savefolder, 'firingmap_'+str(cell_id)+'.png'))
        plt.close()    
    
    return Firing_Rate_In_Position


def make_movie(images, meanImage, savefolder, moviename, boxcar_size=30, fs=30, Traj_x=None, Traj_y=None, ifmask=False):
    '''
    Make a movie from a list of images
    Input:
        images: list of images
        meanImage: mean image of the movie
        savefolder: folder to save the movie
        moviename: name of the movie
        boxcar_size: size of the boxcar filter
    '''
    framesize = images[0].shape

    # Create the video writer
    out = cv2.VideoWriter(os.path.join(savefolder, moviename), cv2.VideoWriter_fourcc(*'DIVX'), fs, framesize, isColor=True)

    #aligment information
    std_base = np.std(meanImage)
    mean_base = np.mean(meanImage)
    lb = mean_base-5*std_base
    up = mean_base+5*std_base

    # Initialize a circular buffer for rolling mean
    rolling_mean_buffer = np.zeros((boxcar_size,) + framesize, dtype=np.float32)
    rolling_mean_index = 0

    for i, img in enumerate(images):
        # Add the current frame to the rolling mean buffer
        rolling_mean_buffer[rolling_mean_index] = img.astype(np.float32)
        rolling_mean_index = (rolling_mean_index + 1) % boxcar_size

        # Calculate the rolling mean
        mean_frame = np.mean(rolling_mean_buffer, axis=0)
    
        #shift the mean value to align with the reference frame
        mean_value = np.mean(mean_frame)
        diff = mean_value-mean_base
        mean_frame = mean_frame-diff

        # Clip values outside the specified range
        lb = mean_base - 5 * std_base
        up = mean_base + 5 * std_base
        mean_frame = np.clip(mean_frame, lb, up)

        # Normalize the image between 0 and 255
        mean_frame = cv2.normalize(mean_frame, None, 0, 255, cv2.NORM_MINMAX)
        
        #if ifmask is True, then set 'white' to mean_frame when img is 0
        if ifmask:
            mean_frame[img==0]= 0
        
        mean_frame = np.uint8(mean_frame)
        #to RGB
        mean_frame = cv2.cvtColor(mean_frame, cv2.COLOR_GRAY2RGB) 
               
        if Traj_x is not None:
            #add the trajectory as a red circle to the mean_frame
            x = Traj_x[i]
            y = Traj_y[i]
            
            cv2.circle(mean_frame, (y, x), 5, (0, 0, 255), -1)
            
        # Write the smoothed frame to the video
        out.write(mean_frame)

    out.release()



    
    