import mne
import numpy as np
import sys
sys.path.append('/media/mangaldeep/HDD2/workspace/MotionControl_MasterThesis')
from scipy.io import loadmat

# %%
def extractBCI3(runs = 0, person_id = [0]):
    files = ['/media/mangaldeep/HDD3/DataSets/BCI3_Competition/data_set_IVa_aa_mat/100Hz/data_set_IVa_aa.mat',
    '/media/mangaldeep/HDD3/DataSets/BCI3_Competition/data_set_IVa_al_mat/100Hz/data_set_IVa_al.mat',
    '/media/mangaldeep/HDD3/DataSets/BCI3_Competition/data_set_IVa_av_mat/100Hz/data_set_IVa_av.mat',
    '/media/mangaldeep/HDD3/DataSets/BCI3_Competition/data_set_IVa_aw_mat/100Hz/data_set_IVa_aw.mat',
    '/media/mangaldeep/HDD3/DataSets/BCI3_Competition/data_set_IVa_ay_mat/100Hz/data_set_IVa_ay.mat',]

    dat = loadmat(files[person_id], struct_as_record=True)
    extra_ch =['AFp1', 'AFp2', 'FAF5', 'FAF1', 'FAF2', 'FAF6', 'FFC7', 'FFC5', 'FFC3', 'FFC1', 'FFC2', 'FFC4', 'FFC6', 
        'FFC8', 'CFC7', 'CFC5', 'CFC3', 'CFC1', 'CFC2', 'CFC4', 'CFC6', 'CFC8', 'CCP7', 'CCP5', 'CCP3', 'CCP1', 'CCP2',
        'CCP4', 'CCP6', 'CCP8', 'PCP7', 'PCP5', 'PCP3', 'PCP1', 'PCP2', 'PCP4', 'PCP6', 'PCP8', 'PPO7', 'PPO5', 'PPO1',
        'PPO2', 'PPO6', 'PPO8', 'OPO1', 'OPO2', 'OI1', 'OI2', 'I1', 'I2']

    # %%
    sfreq = dat['nfo']['fs'][0][0][0][0]
    EEGdata   = dat['cnt'].T
    EEGdata.astype('float64')
    EEGdata = EEGdata*0.1
    nchannels, nsamples = EEGdata.shape

    chan_names = [s[0] for s in dat['nfo']['clab'][0][0][0]]

    event_onsets  = dat['mrk'][0][0][0][0]
    event_codes   = dat['mrk'][0][0][1][0]
    # event_codes = np.nan_to_num(event_codes)

    labels = np.zeros((1, nsamples), int)
    labels[0, event_onsets] = event_codes

    cl_lab = [s[0] for s in dat['mrk']['className'][0][0][0]]
    cl_lab.append('nan')
    cl1    = cl_lab[0]
    cl2    = cl_lab[1]

    # digitized electrode positions 
    xpos = dat['nfo']['xpos']
    ypos = dat['nfo']['ypos']

    nclasses = len(cl_lab)
    nevents = len(event_onsets)

    # Print some information
    # print('Shape of EEG:', EEGdata.shape)
    # print('Sample rate:', sfreq)
    # print('Number of channels:', nchannels)
    # print('Channel names:', chan_names)
    # print('Number of events:', len(event_onsets))
    # print('Event codes:', np.unique(event_codes))
    # print('Class labels:', cl_lab)
    # print('Number of classes:', nclasses)

    # %%
    # Dictionary to store the trials in, each class gets an entry
    trials = {}

    # The time window to extract for each trial, here 0. -- 3.5 seconds
    win = np.arange(int(0*sfreq), int(3.5*sfreq))

    # Length of the time window
    nsamples = len(win)

    # Loop over the classes (right, foot)
    for cl, code in zip(cl_lab, np.unique(event_codes)):
        
        # Extract the onsets for the class
        cl_onsets = event_onsets[event_codes.astype('str') == str(code)]
        
        # Allocate memory for the trials
        trials[cl] = np.zeros((nchannels, nsamples, len(cl_onsets)))
        
        # Extract each trial
        for i, onset in enumerate(cl_onsets):
            trials[cl][:,:,i] = EEGdata[:, win+onset]
            
    # the dimensionality of the data (channels x time x trials)
    # print('Shape of trials[cl1]:', trials[cl1].shape)
    # print('Shape of trials[cl2]:', trials[cl2].shape)
    # print('Shape of trials[00]:', trials[0].shape)

    # %%
    right_hand  = np.rollaxis(trials[cl1], 2, 0)  
    foot = np.rollaxis(trials[cl2], 2, 0) 
    test = np.rollaxis(trials['nan'], 2, 0) 
    data = np.concatenate([right_hand, foot, test])

    # %%
    Y = np.concatenate([-np.ones(right_hand.shape[0]),
                        np.ones(foot.shape[0]),
                        np.zeros(test.shape[0])])
    # data.shape

    # %%
    # Initialize an info structure
    info = mne.create_info(
            ch_names = chan_names,
            ch_types = ['eeg']*nchannels,
            sfreq    = sfreq )  
    # info.set_montage('standard_1020')
    # print('Event created :', info)


    # Electrode Locations
    xpos = dat['nfo']['xpos'][0][0]
    ypos = dat['nfo']['ypos'][0][0]
    layout_pos = np.concatenate([xpos, ypos], axis = 1)
    layout = mne.channels.generate_2d_layout(
        xy = layout_pos,
        ch_names=chan_names,
        name ='EEG custom layout',
        )

    # %%
    ten_twenty_montage = mne.channels.make_standard_montage('standard_1020')
    new_name = []
    for ch in ten_twenty_montage.ch_names:
        ch = ch.lower()
        new_name.append(ch[0].upper() + ch[1:])

    ten_twenty_montage.ch_names = new_name

    ch_names = info['ch_names']
    new_name = []
    for ch in ch_names:
        ch = ch.lower()
        new_name.append(ch[0].upper() + ch[1:])

    # Remove dots from channel names in raw.info['ch_names']
    new_names = []
    for ch_name in new_name:
        new_names.append(ch_name.split('.')[0])

    mne.rename_channels(info, dict(zip(info['ch_names'], new_names)))
    # info.set_montage(ten_twenty_montage, verbose=False, on_missing='ignore');
    # print(ten_twenty_montage.ch_names)
    # print(chan_names)

    # %%
    raw = mne.io.RawArray(EEGdata, info, verbose = False)
    # raw = raw.pick_channels(ch_names = ten_twenty_montage.ch_names).copy()
    # raw.set_montage('standard_1020', on_missing='ignore')

    eventLength = Y.shape[0]
    ev = dat['mrk']['pos'][0][0][0] #[i*sfreq*3 for i in range(eventLength)]

    event_marker = np.column_stack((np.array(ev,  dtype = int),
                            np.zeros(eventLength,  dtype = int),
                            np.array(Y,  dtype = int)))
    ann = mne.annotations_from_events(event_marker, sfreq, event_desc = Y)
    raw.set_annotations(ann)

    # Locations from Physionet dataset
    # mon = mne.channels.read_dig_fif('/media/mangaldeep/HDD2/workspace/MotionControl_MasterThesis/main/extraction/Physionet_ChLoc_raw.fif')
    locinfo = mne.io.read_raw_fif('/media/mangaldeep/HDD2/workspace/MotionControl_MasterThesis/main/extraction/Physionet_ChLoc_raw.fif', preload = False, verbose = False)
    locinfo.pick_channels(raw.ch_names)

    raw._set_channel_positions(locinfo._get_channel_positions(), locinfo.ch_names)

    return raw

if __name__== "__main__":
    # runs = [3, 4, 7, 8, 11,12]
    # person_id = 1
    raw = extractBCI3()
    print(raw)
