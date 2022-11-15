import numpy as np
import mne
import glob
import  data.brain_atlas  as bm
from data.params import OCIParams

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import plotly.express as px
import brainflow as bf
from scipy.signal import butter,lfilter
from scipy.fft import fft, fftfreq, fftshift
import plotly.io as pio
import plotly.graph_objects as go


def extractOCI(runs= [], person_id = [], Expr_name = 'P2_Day*_125'):
    dCfg = OCIParams()
    sfreq = dCfg.sfreq
    ch_names = bm.oci_Channels
    ch_types = ['eeg'] * len(ch_names)
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types);  
    
    train_x, train_y,events = [],[], []
    Expr_name = 'Raw_'+Expr_name+'.npz'
    filename = f'/media/mangaldeep/HDD2/workspace/MotionControl_MasterThesis/data/{Expr_name}'
    for file in glob.glob(filename):
        train = np.load(file, allow_pickle=True)
        train_x.append(np.float64(train['arr_0']))
        events.append(train['arr_1'])
        train_y.append(train['arr_2'])

    train_x = np.hstack(train_x)
    train_y = np.hstack(train_y)
    event_t = np.hstack(events)

    t = train_x.shape[-1]/sfreq
    t = np.arange(t)
    t = t.reshape(event_t.shape[0],-1)
    event_t = t[:,7]
    
    raw = mne.io.RawArray(train_x, info, verbose=False);
    locinfo = mne.io.read_raw_fif('/media/mangaldeep/HDD2/workspace/MotionControl_MasterThesis/main/extraction/Physionet_ChLoc_raw.fif',
                 preload = False, verbose=False)
    locinfo.pick_channels(raw.ch_names)
    raw._set_channel_positions(locinfo._get_channel_positions(), locinfo.ch_names)
    event_data = np.uint16(np.column_stack((event_t, np.zeros((len(train_y,))),train_y)))
    ann = mne.annotations_from_events(event_data,sfreq=sfreq)
    ann.onset = event_t
    raw = raw.set_annotations(ann)
    return raw

if __name__ == "__main__":
    extractOCI(1,1, Expr_name = 'P2_Day*_80') #['Raw_P2_Day1_80.npz','Raw_P2_Day2_80.npz'])
    '''
    # raw.plot_sensors(show_names = True);

    # locinfo.plot_sensors(show_names = True)


    rawfltrd = raw.filter(5, 39, verbose= False, fir_design='firwin', skip_by_annotation='edge').copy()
    scal = dict(mag=1e-12, grad=4e-11, eeg=1000e-6)
    # rawfltrd.plot(scalings = scal);
    # rawfltrd.plot_psd();

    def normalize(train_x):
        for j in range(train_x.shape[0]):
            # try:
            #     train_x[j,:]-= np.mean(train_x[j,:])
            #     train_x[j,:] = (train_x[j,:]/np.std(train_x[j,:]) )/3
            # except Exception as e:
            #     train_x[j,:] =0
            train_x[j,:] = (train_x[j,:] - train_x[j,:].min()) / (train_x[j,:].max() - train_x[j,:].min())
        return train_x

    raw_nrm = normalize(rawfltrd.get_data())
    raw_nrm = mne.io.RawArray(raw_nrm, info);

    # event_t = [961*i for i in range(len(train_y))]
    event_data = np.uint16(np.column_stack((event_t, np.zeros((len(train_y,))),train_y)))
    event_marker, event_id = event_data[:,0],event_data[:,2]
    # idx = event_marker[:,-1].argsort()
    # event_marker = event_marker[idx,:]
    reject = dict(    eeg=20000000000e-6)     # unit: V (EEG channels))
    event_ids = dict({'right':0, 'left':1, 'none':2}) # Replacing the existing event ids
    # epochs1 = mne.Epochs(rawfltrd, events= event_marker, event_id= event_ids, baseline = (0,0))
    epochs = mne.Epochs(raw_nrm, events= event_data, tmin= -1, tmax= 6, event_id= event_ids,reject = None,
            verbose= True, proj= False) ;# Baseline is default (None,0)
    x_epoch = epochs.get_data()

    clas = 'left'

    # epochs['right'].plot_image(combine = 'mean', title = 'right');
    # epochs[clas].average().plot_image(show_names = 'all', titles = clas, picks = ch_names);
    # # epochs[clas].plot_topo_image(layout=layout, title = clas);
    # epochs[clas].plot_image(title = clas, combine = 'mean');
    # epochs[clas].plot_psd();



    # from data.params import OCIParams
    # datadir = "/media/mangaldeep/HDD2/workspace/MotionControl_MasterThesis/data"
    # EXPR_NAME = 'Processed_'+filename

    # # dCfg = OCIParams()
    # x_epoch = np.expand_dims(x_epoch, axis=1)
    # train_x,test_x,train_y,test_y = train_test_split(x_epoch, np.float16(epochs.events[:,-1]), test_size= 0.1,
    #                                 stratify= epochs.events[:,-1], random_state= 42)
    # # print(channel_datas.shape)
    # print(f"saving data...")
    # # labels = labels[ACTION]*np.ones(self.TOTAL_ITERS)
    # np.savez(f'{datadir}/train/{EXPR_NAME}',train_x, train_y)
    # np.savez(f'{datadir}/test/{EXPR_NAME}',test_x, test_y)
    '''