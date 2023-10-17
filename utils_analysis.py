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

def getTuningMap(spks, X, Z, timestamps, VRsize=(1, 1), binsize=(0.025, 0.025),
                 sigma=3/2.5, speed_thres=0.025, boxcar_size=5, visit_thres=0.1,
                 peak_thres=100, cell_id=None, datafolder=None, return_all=False):
    '''
    Calculate the tuning map of a neuron
    Input:
        spks: spike times
        X: X position of the animal
        Z: Z position of the animal
        timestamps: time stamps of each frame
        VRsize: size of the VR
        binsize: size of the bin
        sigma: sigma of the Gaussian filter
        speed_thres: threshold of the speed
        boxcar_size: size of the boxcar filter
        visit_thres: threshold of the visit
        peak_thres: threshold of the peak
        cell_id: the id of the neuron
        datafolder: the folder to save the plot
        return_all: if True, also return all_calcium_mean and prob_visit
    '''
    
    
    # 1, calculate the smoothed moving speed
    linearspeed = getLinearSpeed(X, Z, timestamps, boxcar_size=boxcar_size)
    
    # 2, removing periods when the animal's speed is below a threshold
    ind_thres = linearspeed >= speed_thres
    spks = spks[ind_thres]
    X = X[ind_thres]
    Z = Z[ind_thres]
    #calculate diffTimeStamps and add 0 at the beginning to make sure dimension does not change in one line
    diffTimeStamps =  np.concatenate((np.asarray([0]), np.diff(timestamps)))
    diffTimeStamps = diffTimeStamps[ind_thres]

    # Calculate bin indices and remove edge effect
    X[X == 1] = 1 - 1e-5
    Z[Z == 1] = 1 - 1e-5
    Pos_X = (X / binsize[0]).astype(int)
    Pos_Z = (Z / binsize[1]).astype(int)

    # Calculate Time_In_Position and Spks_In_Position using bin indices
    n_bins_x = int(VRsize[0] / binsize[0])
    n_bins_z = int(VRsize[1] / binsize[1])
    
    Time_In_Position = np.zeros((n_bins_z, n_bins_x))
    Spks_In_Position = np.zeros((n_bins_z, n_bins_x))
    
    np.add.at(Time_In_Position, (Pos_Z, Pos_X), diffTimeStamps)
    np.add.at(Spks_In_Position, (Pos_Z, Pos_X), spks)

    # Here, I calculate the mean calcium activity and save it as ave_calcium_in_bin
    ave_calcium_in_bin = np.divide(Spks_In_Position, (Time_In_Position + 1e-5))
    
    # Smooth the firing rate matrix with a Gaussian filter
    ave_calcium_in_bin_gs = gaussian_filter(ave_calcium_in_bin, sigma=sigma)
    
    # Find Time_In_Position < visit_thres and set Firing_Rate_In_Position to np.nan
    ave_calcium_in_bin_gs[Time_In_Position < visit_thres] = np.nan
    
    ave_calcium_in_bin_raw = ave_calcium_in_bin.copy()
    ave_calcium_in_bin_raw[Time_In_Position < visit_thres] = np.nan
    
    # Plot the map only if those peak values are larger than peak_thres
    if np.nanmax(ave_calcium_in_bin_gs) > peak_thres:
        plt.figure(figsize=(3, 3), dpi=300)
        plt.imshow(ave_calcium_in_bin_gs, cmap='inferno')
        plt.xlabel('VR X')
        plt.ylabel('VR Y')
        plt.xticks(np.linspace(0, ave_calcium_in_bin_gs.shape[0], 5), np.linspace(0, 1, 5))
        plt.yticks(np.linspace(0, ave_calcium_in_bin_gs.shape[1], 5), np.linspace(0, 1, 5))
        plt.colorbar(label='Calcium activity')
        
        # Save the figure
        savefolder = os.path.join(datafolder, 'UnrotTiff/', '2D2P/', 'firingmaps/')
        if not os.path.exists(savefolder):
            os.makedirs(savefolder)
        
        # Save the plot
        plt.savefig(os.path.join(savefolder, 'firingmap_' + str(cell_id) + '.png'))
        plt.close()
    
    if return_all:
        all_calcium_mean = np.sum(Spks_In_Position) / np.sum(Time_In_Position)
        prob_visit = Time_In_Position / np.sum(Time_In_Position)
        prob_visit[Time_In_Position < visit_thres] = np.nan
        
        #cut the border for 5 pixels and keep the remaining 30*30 pixels
        Spks_In_Position_cut = Spks_In_Position[5:-5,5:-5]
        Time_In_Position_cut = Time_In_Position[5:-5,5:-5]
        all_calcium_mean_cut = np.sum(Spks_In_Position_cut) / np.sum(Time_In_Position_cut)
        ave_calcium_in_bin_raw_cut = ave_calcium_in_bin_raw[5:-5,5:-5]
        prob_visit_cut = Time_In_Position_cut / np.sum(Time_In_Position_cut)
        
        return ave_calcium_in_bin_gs, ave_calcium_in_bin_raw, all_calcium_mean, prob_visit, ave_calcium_in_bin_raw_cut, all_calcium_mean_cut, prob_visit_cut
    else:
        return ave_calcium_in_bin_gs
    

def getTuningMap_shuffle(spks_shuffle, X, Z, timestamps, VRsize=(1, 1), binsize=(0.025, 0.025),
                 sigma=3/2.5, speed_thres=0.025, boxcar_size=5, visit_thres=0.1, return_gaussian_filtered=False):
    
    n_shuffle = spks_shuffle.shape[0]
    
    # 1, calculate the smoothed moving speed
    linearspeed = getLinearSpeed(X, Z, timestamps, boxcar_size=boxcar_size)
    
    # 2, removing periods when the animal's speed is below a threshold
    ind_thres = linearspeed >= speed_thres
    spks_shuffle = spks_shuffle[:,ind_thres]
    X = X[ind_thres]
    Z = Z[ind_thres]
    #calculate diffTimeStamps and add 0 at the beginning to make sure dimension does not change in one line
    diffTimeStamps =  np.concatenate((np.asarray([0]), np.diff(timestamps)))
    diffTimeStamps = diffTimeStamps[ind_thres]

    # Calculate bin indices and remove edge effect
    X[X == 1] = 1 - 1e-5
    Z[Z == 1] = 1 - 1e-5
    Pos_X = (X / binsize[0]).astype(int)
    Pos_Z = (Z / binsize[1]).astype(int)

    # Calculate Time_In_Position and Spks_In_Position using bin indices
    n_bins_x = int(VRsize[0] / binsize[0])
    n_bins_z = int(VRsize[1] / binsize[1])
    
    Time_In_Position = np.zeros((n_shuffle, n_bins_z, n_bins_x))
    Spks_In_Position = np.zeros((n_shuffle, n_bins_z, n_bins_x))
    
    # perform np.add.at for each shuffle
    for i in range(n_shuffle):
        np.add.at(Time_In_Position[i], (Pos_Z, Pos_X), diffTimeStamps)
        np.add.at(Spks_In_Position[i], (Pos_Z, Pos_X), spks_shuffle[i])

    # Here, I calculate the mean calcium activity and save it as ave_calcium_in_bin
    ave_calcium_in_bin = np.divide(Spks_In_Position, (Time_In_Position + 1e-5))
    
    if return_gaussian_filtered:
        #Gaussian filter along the last two dimensions of ave_calcium_in_bin
        for i in range(n_shuffle):
            ave_calcium_in_bin[i] = gaussian_filter(ave_calcium_in_bin[i], sigma=sigma)
        ave_calcium_in_bin[Time_In_Position < visit_thres] = np.nan   
        return ave_calcium_in_bin
    
    # Find Time_In_Position < visit_thres and set Firing_Rate_In_Position to np.nan
    ave_calcium_in_bin[Time_In_Position < visit_thres] = np.nan
        
    
    #calculate all_calcium_mean prob_visit for each shuffle
    all_calcium_mean = np.sum(Spks_In_Position, axis=(1,2)) / np.sum(Time_In_Position, axis=(1,2))
    
    prob_visit = Time_In_Position / np.sum(Time_In_Position, axis=(1,2))[:,None,None]
    prob_visit[Time_In_Position < visit_thres] = np.nan
    
    #cut the border for 5 pixels and keep the remaining 30*30 pixels
    Spks_In_Position_cut = Spks_In_Position[:,5:-5,5:-5]
    Time_In_Position_cut = Time_In_Position[:,5:-5,5:-5]
    all_calcium_mean_cut = np.sum(Spks_In_Position_cut, axis=(1,2)) / np.sum(Time_In_Position_cut, axis=(1,2))
    prob_visit_cut = Time_In_Position_cut / np.sum(Time_In_Position_cut, axis=(1,2))[:,None,None]
    ave_calcium_in_bin_cut = ave_calcium_in_bin[:,5:-5,5:-5]
    
    return ave_calcium_in_bin, all_calcium_mean, prob_visit, ave_calcium_in_bin_cut, all_calcium_mean_cut, prob_visit_cut   

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



    
    