import numpy as np
#tqdm
from tqdm import tqdm

def smooth_percentile(Fcorr, TimeWindow, percentile):
    '''
    Get the uncorrected running baseline of Fcorr
    Input:
        Fcorr: [nCells, TimePoints], neuropil corrected signal
        TimeWindow: the window size to calculate the percentile
        percentile: the percentile to calculate
    Output:
        Fs: the running baseline (uncorrected)
    '''
    Length = Fcorr.shape[1]
    Fs = Fcorr.copy()

    HalfSpan = np.ceil(TimeWindow / 2).astype(int)

    #tqdm
    for i in tqdm(range(HalfSpan, Length - HalfSpan)):
        Fs[:,i] = np.percentile(Fcorr[:, i - HalfSpan:i + HalfSpan], percentile, axis=1)

    for i in range(HalfSpan):
        Fs[:, i] = np.percentile(Fcorr[:, i:i + 2 * HalfSpan], percentile, axis=1)

    for i in range(Length - HalfSpan, Length):
        Fs[:, i] = np.percentile(Fcorr[:, i - 2 * HalfSpan:i], percentile, axis=1)

    return Fs

def movstd(Fcorr, TimeWindow):
    '''
    Calculate the moving standard deviation of Fcorr with a moving window of TimeWindow
    Input:
        Fcorr: [nCells, TimePoints], neuropil corrected signal
        TimeWindow: the window size to calculate the percentile
    '''
    
    Length = Fcorr.shape[1]
    Fstd_local = Fcorr.copy()

    HalfSpan = np.ceil(TimeWindow / 2).astype(int)
    #tqdm
    for i in tqdm(range(HalfSpan, Length - HalfSpan)):
        Fstd_local[:, i] = np.std(Fcorr[:, i - HalfSpan:i + HalfSpan], axis=1)

    for i in range(HalfSpan):
        Fstd_local[:, i] = np.std(Fcorr[:, i:i + 2 * HalfSpan], axis=1)
        
    for i in range(Length - HalfSpan, Length):
        Fstd_local[:, i] = np.std(Fcorr[:, i - 2 * HalfSpan:i], axis=1)
        
    return Fstd_local

def get_deltaF_F_and_fr(spks, Fcorr, moving_window=15, framerate=30, percetile=8):
    '''
    Get deltaF_F and firing rate (not the real firing rate but suggested by https://github.com/MouseLand/suite2p/issues/267
    Input:  
        spks: [nCells, TimePoints], deconvolved calcium events
        Fcorr: [nCells, TimePoints], corrected fluorescence signal
        moving_window: the window size to calculate the percentile
        framerate: the framerate of the recording
        percetile: the percentile to calculate
    Output:
        deltaF_F: [nCells, TimePoints], deltaF_F
        fr: [nCells, TimePoints], firing rate
    '''
    
    #diminish the effect that slow fluctuations like photobleaching can ahve on Fcorr. We then 
    #extract a running baseline F0(t), which is estimated as a sum of two components: F0(t) = Fs(t) + Fm
    #Fs(t) denotes the eighth percentile of Fcorr(t) within a +- 15 s window center at t, and m is a constant 
    #value added to Fs(t) to make F0(t) centered around zero in periods without calcium activity (baseline points). 
    #These baseline points were extracted as time points for which the local standard deviation (std) of the signal 
    #(±15s moving window) did not exceed a cutoff of std min +0.1*(std max-std min ), where std min and std max 
    #denote the minimal and maximal standard deviations over all data points
    
    deltaF_F = np.zeros(Fcorr.shape)
    fr = np.zeros(Fcorr.shape)

    #1, get Fs(t), the eighth percentile of Fcorr(t) within a +- 15 s window center at t
    print('Getting the uncorrected running baseline Fs(t)...')
    TimeWindow = 2*moving_window*framerate #+-15s
    Fs = smooth_percentile(Fcorr, TimeWindow, percetile) #dim=[nCells, TimePoints]

    #2, get m, the constant value added to Fs(t) to make F0(t) centered around zero in periods without calcium activity
    print('Getting the constant mean value Fm...')
    Fstd_local = movstd(Fcorr, TimeWindow) #dim=[nCells, TimePoints]
    Fstd_local_min = np.min(Fstd_local, axis=1) #dim=[nCells]
    Fstd_local_max = np.max(Fstd_local, axis=1) #dim=[nCells]
    Fstd_local_10percent = Fstd_local_min + 0.1*(Fstd_local_max-Fstd_local_min) #dim=[nCells]
    
    #for each cell, get m value 
    #initialize Fm as nCells rows
    Fm = np.zeros(Fcorr.shape[0])
    for i in range(Fcorr.shape[0]):
        Fm[i] = np.mean(Fcorr[i, Fstd_local[i,:]< Fstd_local_10percent[i]]-Fs[i, Fstd_local[i,:]< Fstd_local_10percent[i]])

    #add Fs and Fm to get F0
    F0 = Fs + Fm[:, np.newaxis] #dim=[nCells, TimePoints]

    #3, get deltaF_F  which equals to (Fcorr_i-F0)/F0
    #and fr: firing rate (not the real firing rate but suggested by https://github.com/MouseLand/suite2p/issues/267)
    print('Getting deltaF_F and firing rate...')
    for i in range(Fcorr.shape[0]):
        if np.mean(F0[i,:])<0:
            deltaF_F[i,:] = (F0[i,:]-Fcorr[i,:])/(F0[i,:]*-1)
            fr[i,:] = spks[i,:]/(F0[i,:]*-1)
        else:
            deltaF_F[i,:] = (Fcorr[i,:]-F0[i,:])/F0[i,:]
            fr[i,:] = spks[i,:]/F0[i,:]

    #for each row in fr, if the abs value exceed 50, set to 50,; also set negative value to zero
    fr[fr>50] = 0
    fr[fr<0] = 0
            
    return deltaF_F, fr