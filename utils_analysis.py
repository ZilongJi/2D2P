import os
import cv2
import numpy as np
import pickle
from multiprocessing import Pool
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

def get_indices_circle(i, j, r, n_bins_z, n_bins_x):
    """
    get the indices of the circle with center (i,j) and radius r
    """
    #initialize the indices
    ind = []
    #for each row, get the indices of the circle
    for row in range(i-r, i+r+1):
        #for each column, get the indices of the circle
        for col in range(j-r, j+r+1):
            #if the point is within the circle, then append the index to ind
            if (row-i)**2+(col-j)**2 <= r**2:
                #if the point is out of the matrix, then skip
                if row<0 or row>=n_bins_z or col<0 or col>=n_bins_x:
                    continue
                else:
                    ind.append((row, col))
    return ind

def adaptive_binning(Spks_In_Position, Time_In_Position, fs=30, alpha=1e6):
    """
    adaptive binning technique
    for each spatial bin, expanding a circle around this point until the floowing criteria is met:
    N_spikes > alpha /(N_occ**2*r**2)
    where N_occ is the number of occipancy samples falling within the circle, this equals to 
    Time_In_Position falling within the circle multiplied by the sampling rate fs;
    N_spikes is the number of spikes falling within the circle
    The adaptive spikes equals to fs*N_spikes/N_occ

    Args:
        Spks_In_Position (_type_): _description_
        Time_In_Position (_type_): _description_
        fs (int, optional): _description_. Defaults to 30.
        alpha (_type_, optional): _description_. Defaults to 1e6.

    Returns:
        adaptive_spikes: the adaptive spikes
        radius_matrix: the radius of the circle for each spatial bin
    """
    #get the size of the matrix
    n_bins_z, n_bins_x = Spks_In_Position.shape
    #initialize adaptive_spikes matrix
    adaptive_spikes = np.zeros((n_bins_z, n_bins_x))
    #initialze radius matrix
    radius_matrix = np.zeros((n_bins_z, n_bins_x))
    #calculate the occupancy matrix
    occ = Time_In_Position*fs
    #for each bin, do the adaptive binning calculation
    for i in range(n_bins_z):
        for j in range(n_bins_x):
            for r in range(1, n_bins_x):
                #get the indices of the circle, ind is a list
                ind = get_indices_circle(i, j, r, n_bins_z, n_bins_x)
                #calculate the number of spikes and occupancy within the circle
                #where ind is a list of tuples
                N_spikes = np.sum([Spks_In_Position[tup] for tup in ind])
                N_occ = np.sum([occ[tup] for tup in ind]) + 1e-5 #in case N_occ is 0
                #if the criteria is met, save the adaptive_spikes then break the inner loop
                if N_spikes > alpha/(N_occ**2*r**2):
                    adaptive_spikes[i,j] = fs*N_spikes/N_occ
                    radius_matrix[i,j] = r
                    break

    return adaptive_spikes, radius_matrix

def getTuningMap(fr, X, Z, timestamps, VRsize=(1, 1), binsize=(0.025, 0.025),
                 sigma=3/2.5, speed_thres=0.025, boxcar_size=5, visit_thres=0.1,
                 peak_thres=1, cell_id=None, datafolder=None, return_all=False,
                 apply_adaptive_binning = True):
    '''
    Calculate the tuning map of a neuron
    Input:
        fr: firing rate of the neuron
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
        apply_adaptive_binning: if True, apply adaptive binning
    '''
    
    
    # 1, calculate the smoothed moving speed
    linearspeed = getLinearSpeed(X, Z, timestamps, boxcar_size=boxcar_size)
    
    # 2, removing periods when the animal's speed is below a threshold
    ind_thres = linearspeed >= speed_thres
    fr = fr[ind_thres]
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

    # Calculate Time_In_Position and fr_In_Position using bin indices
    n_bins_x = int(VRsize[0] / binsize[0])
    n_bins_z = int(VRsize[1] / binsize[1])
    
    Time_In_Position = np.zeros((n_bins_z, n_bins_x))
    fr_In_Position = np.zeros((n_bins_z, n_bins_x))
    
    np.add.at(Time_In_Position, (Pos_Z, Pos_X), diffTimeStamps)
    np.add.at(fr_In_Position, (Pos_Z, Pos_X), fr)

    # Here, I calculate the mean calcium activity and save it as ave_calcium_in_bin
    if apply_adaptive_binning:
        ave_calcium_in_bin, radius_matrix = adaptive_binning(fr_In_Position, Time_In_Position, fs=30, alpha=1e8)
    else:
        ave_calcium_in_bin = np.divide(fr_In_Position, (Time_In_Position + 1e-5))
    
    # Smooth the firing rate matrix with a Gaussian filter
    ave_calcium_in_bin_gs = gaussian_filter(ave_calcium_in_bin, sigma=sigma)
    
    # Find Time_In_Position < visit_thres and set Firing_Rate_In_Position to np.nan
    ave_calcium_in_bin_gs[Time_In_Position < visit_thres] = np.nan
    
    ave_calcium_in_bin_raw = ave_calcium_in_bin.copy()
    ave_calcium_in_bin_raw[Time_In_Position < visit_thres] = np.nan
    
    # Plot the map only if those peak values are larger than peak_thres
    if np.nanmax(ave_calcium_in_bin_gs) > peak_thres:
        
        # Save the figure
        savefolder = os.path.join(datafolder, 'UnrotTiff/', '2D2P/', 'firingmaps/')
        if not os.path.exists(savefolder):
            os.makedirs(savefolder)
        
        plt.figure(figsize=(3, 3), dpi=300)
        labelsize = 10
        ticksize = 8
        #imshow the map
        plt.imshow(ave_calcium_in_bin_gs, cmap='inferno')
        #xlabel and ylabel
        plt.xlabel('VR X', fontsize=labelsize)
        plt.ylabel('VR Y', fontsize=labelsize)
        plt.xticks([]); plt.yticks([])
        #add colorbar and label in a latex format Dconv/F_0 /dot s^-1
        cbar = plt.colorbar(label='$Dconv/F_0 \cdot s^{-1}$', shrink=0.8)
        #remove colorbar ticks
        cbar.set_ticks([])
        #set tick labels size as ticksize
        cbar.ax.tick_params(labelsize=labelsize)
        #add peak value as text at the right top corner of the map, do not overlap with the map
        #plt.text(0.5, 1.05, 'Peak={:.2f}'.format(np.nanmax(map)), fontsize=labelsize, transform=plt.gca().transAxes)
        plt.title('Peak={:.2f}'.format(np.nanmax(ave_calcium_in_bin_gs)), fontsize=labelsize)
        
        # Save the plot
        plt.savefig(os.path.join(savefolder, 'firingmap_' + str(cell_id) + '.png'))
        plt.close()
        
        #save the radius matrix as a image with colorbar
        plt.figure(figsize=(3, 3), dpi=300)
        plt.imshow(radius_matrix, cmap='inferno')
        plt.xlabel('VR X', fontsize=labelsize)
        plt.ylabel('VR Y', fontsize=labelsize)
        plt.xticks([]); plt.yticks([])
        plt.colorbar(label='Radius', shrink=0.8)

        # Save the plot
        plt.savefig(os.path.join(savefolder, 'radius_' + str(cell_id) + '.png'))
        plt.close()
    
    if return_all:
        all_calcium_mean = np.sum(fr_In_Position) / np.sum(Time_In_Position)
        prob_visit = Time_In_Position / np.sum(Time_In_Position)
        prob_visit[Time_In_Position < visit_thres] = np.nan
        
        #cut the border for 5 pixels and keep the remaining 30*30 pixels
        fr_In_Position_cut = fr_In_Position[5:-5,5:-5]
        Time_In_Position_cut = Time_In_Position[5:-5,5:-5]
        all_calcium_mean_cut = np.sum(fr_In_Position_cut) / np.sum(Time_In_Position_cut)
        ave_calcium_in_bin_raw_cut = ave_calcium_in_bin_raw[5:-5,5:-5]
        prob_visit_cut = Time_In_Position_cut / np.sum(Time_In_Position_cut)
        
        return ave_calcium_in_bin_gs, ave_calcium_in_bin_raw, all_calcium_mean, prob_visit, ave_calcium_in_bin_raw_cut, all_calcium_mean_cut, prob_visit_cut
    else:
        return ave_calcium_in_bin_gs

# Function to calculate ave_calcium_in_bin for a single shuffle
def process_shuffle(i, fr_In_Position, Time_In_Position, apply_adaptive_binning):
    if apply_adaptive_binning:
        ave_calcium_in_bin_shuffle, _ = adaptive_binning(fr_In_Position[i], Time_In_Position[i], fs=30, alpha=1e8)
        return ave_calcium_in_bin_shuffle
    else:
        return np.divide(fr_In_Position[i], (Time_In_Position[i] + 1e-5))


def getTuningMap_shuffle(fr_shuffle, X, Z, timestamps, VRsize=(1, 1), binsize=(0.025, 0.025),
                 sigma=3/2.5, speed_thres=0.025, boxcar_size=5, visit_thres=0.1, return_gaussian_filtered=False,
                 apply_adaptive_binning = True):
    
    n_shuffle = fr_shuffle.shape[0]
    
    # 1, calculate the smoothed moving speed
    linearspeed = getLinearSpeed(X, Z, timestamps, boxcar_size=boxcar_size)
    
    # 2, removing periods when the animal's speed is below a threshold
    ind_thres = linearspeed >= speed_thres
    fr_shuffle = fr_shuffle[:,ind_thres]
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

    # Calculate Time_In_Position and fr_In_Position using bin indices
    n_bins_x = int(VRsize[0] / binsize[0])
    n_bins_z = int(VRsize[1] / binsize[1])
    
    Time_In_Position = np.zeros((n_shuffle, n_bins_z, n_bins_x))
    fr_In_Position = np.zeros((n_shuffle, n_bins_z, n_bins_x))
    
    # perform np.add.at for each shuffle
    for i in range(n_shuffle):
        np.add.at(Time_In_Position[i], (Pos_Z, Pos_X), diffTimeStamps)
        np.add.at(fr_In_Position[i], (Pos_Z, Pos_X), fr_shuffle[i])

    # # Here, I calculate the mean calcium activity and save it as ave_calcium_in_bin
    # if apply_adaptive_binning:
    #     ave_calcium_in_bin = np.zeros((n_shuffle, n_bins_z, n_bins_x))
    #     for i in range(n_shuffle):
    #         ave_calcium_in_bin_shuffle, _ = adaptive_binning(fr_In_Position[i], Time_In_Position[i], fs=30, alpha=1e8)
    #         ave_calcium_in_bin[i] = ave_calcium_in_bin_shuffle
    # else:
    #     ave_calcium_in_bin = np.divide(fr_In_Position, (Time_In_Position + 1e-5))

    with Pool(processes=16) as pool:
        # Use partial from functools to pass additional arguments to process_shuffle
        import functools
        process_shuffle_partial = functools.partial(process_shuffle, fr_In_Position=fr_In_Position, Time_In_Position=Time_In_Position, apply_adaptive_binning=apply_adaptive_binning)
        ave_calcium_in_bin = pool.map(process_shuffle_partial, range(n_shuffle))

    #convert ave_calcium_in_bin to numpy array
    ave_calcium_in_bin = np.array(ave_calcium_in_bin)
    
    if return_gaussian_filtered:
        #Gaussian filter along the last two dimensions of ave_calcium_in_bin
        for i in range(n_shuffle):
            ave_calcium_in_bin[i] = gaussian_filter(ave_calcium_in_bin[i], sigma=sigma)
        ave_calcium_in_bin[Time_In_Position < visit_thres] = np.nan   
        return ave_calcium_in_bin
    
    # Find Time_In_Position < visit_thres and set Firing_Rate_In_Position to np.nan
    ave_calcium_in_bin[Time_In_Position < visit_thres] = np.nan
        
    
    #calculate all_calcium_mean prob_visit for each shuffle
    all_calcium_mean = np.sum(fr_In_Position, axis=(1,2)) / np.sum(Time_In_Position, axis=(1,2))
    
    prob_visit = Time_In_Position / np.sum(Time_In_Position, axis=(1,2))[:,None,None]
    prob_visit[Time_In_Position < visit_thres] = np.nan
    
    #cut the border for 5 pixels and keep the remaining 30*30 pixels
    fr_In_Position_cut = fr_In_Position[:,5:-5,5:-5]
    Time_In_Position_cut = Time_In_Position[:,5:-5,5:-5]
    all_calcium_mean_cut = np.sum(fr_In_Position_cut, axis=(1,2)) / np.sum(Time_In_Position_cut, axis=(1,2))
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



    
    