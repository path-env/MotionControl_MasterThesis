# %%
# as png
# %matplotlib inline 
#  interactable inside ide
# %matplotlib widget
### interactable seperate window
# %matplotlib tk 

# %%
from time import time
import mne
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.preprocessing import MinMaxScaler
#Local Imports

# from main.extraction.physionet_MI import  extractPhysionet, extractBCI3
from data import brain_atlas  as bm
from data.params import BCI3Params, EEGNetParams, OCIParams, PhysionetParams, globalTrial
from main.extraction.oci_MI import extractOCI
# Data Extraction
from main.extraction.bci3_IVa import extractBCI3
from main.extraction.physionet_MI import extractPhysionet

# Feature Extraction
from mne.time_frequency import tfr_morlet
from mne.decoding import CSP
from sklearn.model_selection import train_test_split
from kymatio.sklearn import Scattering2D
from scipy import stats, signal

# Classifier
import torch, torchvision
from torchvision import transforms, models
import torch.nn as nn
import torch.optim as optim

# from torchsummary import summary

import torch.optim.lr_scheduler as lr_scheduler
# torch.manual_seed(42)
# torch.cuda.manual_seed_all(42)

from sklearn.model_selection import ShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.svm import SVC
from sklearn.kernel_approximation import RBFSampler

from main.extraction.data_extractor import DataContainer
from models.neurotec_edu import Neurotech_net
from models.MI_CNN_WT_2019 import TFnet
from models.EEGNet_2018 import EEGnet
from utils.train_net import train_and_validate

# %%
class BrainSignalAnalysis():
    def __init__(self, raw, data_cfg = OCIParams(), analy_cfg = globalTrial(), net_cfg = EEGNetParams()) -> None:
        self.dCfg = data_cfg
        self.aCfg = analy_cfg
        self.nCfg = net_cfg

        # Parameters
        elec_lines_f, L_cutoff, H_cutoff = self.dCfg.elec_lines_f, self.aCfg.mu_rhythm[0], self.aCfg.beta_rhythm[-1] #HZ

        # %%
        self.plot_enable = 0
        ica_ssp_comp = 0

        # %%
        # Remove and Filter signal noises
        # raw.notch_filter(elec_lines_f)
        rawfltrd = raw.filter(L_cutoff, H_cutoff, verbose= False, fir_design='firwin', 
                                skip_by_annotation='edge').copy()

        # Referncing to reference electrodes
        # rawfltrd = rawfltrd.set_eeg_reference(self.dCfg.inion)
        # Check the Power spectral density
        # rawfltrd.plot_psd();
        pick_ch = bm.oci_Channels  # Considering only the channels that map to topo map functionality
        # rawfltrd.pick_channels(pick_ch); # inplace
        # Channel names to Indices
        ch_names = rawfltrd.ch_names
        self.pick_ch_idx = [ch_names.index(i) for i in pick_ch]

        # Duplicate params
        self.scale = dict(mag=1e-12, grad=4e-11, eeg=100e-6)
        mne_plot_raw = dict(scalings=self.scale, clipping='transparent', order=self.pick_ch_idx)

        # %%
        ## Analysis after Spectral filter
        if self.plot_enable ==1:
            rawfltrd.plot(scalings=self.scale, clipping='transparent'); #, order=self.pick_ch_idx
            # rawfltrd.plot(scalings='auto');
            # rawfltrd.plot_psd_topo(show = True);
            # rawfltrd.plot_sensors(show_names=True,kind = '3d', sphere=(0.0, 0.015, 0.033, 0.1));

        self.rawfltrd = rawfltrd
    
    def _matrix2Image(self):
            # self.train_x = scaler.fit_transform(self.train_x)
            cmap = np.uint8(cm.gist_earth(self.train_x)*255)[:,:,:,:3]
            cmap = np.swapaxes(cmap, 1,3)
            cmap = np.swapaxes(cmap, 2,3)

            preprocess = transforms.Compose([
                # transforms.ToPILImage(),
                transforms.Resize(256),
                transforms.CenterCrop(227),
                # transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

            self.train_x = torch.from_numpy(cmap.astype(np.float32))
            # input_image = Image.fromarray(cmap)
            input_tensor = preprocess(self.train_x)
            input_tensor-=input_tensor.min()
            input_tensor/=input_tensor.max()
            self.train_x = input_tensor #.unsqueeze(0) # create a mini-batch as expected by the model

            # # Plot the proceessed image
            # import cv2
            # test = self.train_x[0,:,:,:].swap_axes(0,2).numpy()
            # cv2.imshow('Test', test)
            # cv2.waitKey(0)
            return None

    def normalize(self, x):
        # Normalizing
        for ep in range(x.shape[0]):
            for ch in range(x.shape[1]):
                x[ep,ch,:] = (x[ep,ch,:] - x[ep,ch,:].min()) / (x[ep,ch,:].max() - x[ep,ch,:].min())
            # try:
            #     train_x[j,:]-= np.mean(train_x[j,:])
            #     train_x[j,:] = (train_x[j,:]/np.std(train_x[j,:]) )/3
            # except Exception as e:
            #     train_x[j,:] =0
        return x
    
    # %%
    # Remove the artifacts
    def artifact_removal(self, method, save_epochs = False):
        rawfltrd = self.rawfltrd.copy()
        rawfltrd = rawfltrd.resample(self.dCfg.sfreq) # for OpenBCI headset
        if method.find('locl')!= -1:
            pick_ch = bm.oci_Channels  # Considering only the channels that map to topo map functionality
            rawfltrd.pick_channels(pick_ch); # inplace

        if method.find('ssp')!=-1:
            eog_proj, events = mne.preprocessing.compute_proj_eog(rawfltrd, n_grad=0, n_mag=0, n_eeg=self.dCfg.ssp_n_eeg, average=True, verbose=False, ch_name=  self.dCfg.eog_ref_ch, reject=None) # returns EOG Proj and events of blinks
            rawfltrd.add_proj(projs=eog_proj);

        # ##### Common Average Reference(CAR) (Projector)
        ## TODO- Remove bads prior
        if method.find('car')!=-1:
            rawfltrd.set_eeg_reference(ref_channels='average', projection=True, verbose = False);
            ## Analysis after Projections
            if self.plot_enable ==1:
                # rawfltrd_ssp.plot_projs_topomap();
                # rawfltrd_car.info['projs']
                # Compare after and before projection
                # rawfltrd.plot(scalings=self.scale, clipping='transparent', order=pick_ch_idx);
                rawfltrd.plot(scalings=self.scale, clipping='transparent', butterfly= False, title='Raw Filtered (Without Projection))', proj= False, order=self.pick_ch_idx);
                rawfltrd.plot(scalings=self.scale, clipping='transparent', butterfly= False, title='Raw Filtered(With Projection)', proj= True, order=self.pick_ch_idx);

            # rawfltrd_proj = rawfltrd_car.apply_proj()

        # #### Artifact Removal : Independent Component Analysis(ICA)
        if method.find('ica')!=-1:
            # cov = mne.Covariance()
            # rawfltrd_ica = rawfltrd_car.apply_proj().copy()
            ica = mne.preprocessing.ICA(n_components=self.dCfg.ica_n_comp, noise_cov= None, random_state=2,
                                     method='picard',max_iter=500, verbose=False)
            # Create an instance of RAW
            # rawfltrd_ica = raw.copy()
            rawfltrd.apply_proj()
            ica.fit(rawfltrd, verbose=False);
            # n_components  = 10 then ica.exclude = [1,2]
            ica.exclude = []
            # Using EOG Channel to select ICA Components
            ica.exclude , ex_scores = ica.find_bads_eog(rawfltrd, ch_name= self.dCfg.eog_ref_ch, verbose = False);#,threshold=2);

            # %%
            ## TODO: ICA template matching for Multiple subjects
            ## TODO: Plot ICA evoked.

            # %%
            if len(ica.exclude) ==0:
                print('!!!!!!!!!!! No components excluded !!!!!!!!!!!!!!!!')
            # ica.exclude = []#=[1,6]
            # ica.plot_scores(ex_scores);
            # ica.plot_properties(rawfltrd_ica, picks = ica.exclude)

            # %%
            ## Analysis after ICA
            if self.plot_enable ==1:
                # ica.plot_components();
                # ica.plot_sources(rawfltrd_ica);
                # ica.plot_overlay(rawfltrd_ica, exclude = ica.exclude);
                # ica.plot_properties(rawfltrd_ica, picks = ica.exclude);
                ica.plot_scores(ex_scores);

            # ica.exclude = [0,1,2,3] # manually exclude ICA components based on observation on plots above
            rawfltrd = ica.apply(rawfltrd, verbose=False) # Reconstructed sensor data (In Senso space)
            if self.plot_enable ==1:
                rawfltrd.plot(scalings=self.scale, clipping='transparent', title='ICA  on RAW', proj= False); # , order=pick_ch_idx
                # rawfltrd.plot(scalings=self.scale, clipping='transparent', title='Raw Filtered- w Projection', proj=True); #, order=pick_ch_idx
                # rawfltrd_ssp.plot(scalings=self.scale, clipping='transparent', title='Raw Filtered- wo Projection', proj=False);# , order=pick_ch_idx

            # %%
            # rawfltrd_proj = rawfltrd_car.copy()
            # rawfltrd_proj.apply_proj()
            # rawfltrd_proj.plot(scalings=self.scale, clipping='transparent', title='ICA+SSP', proj=False);

            # %%
            if self.plot_enable ==1:
                # rawfltrd_ica.plot_psd(fmin=L_cutoff, fmax=H_cutoff,picks=['C4','C2','C6']);
                # rawfltrd_proj.plot_psd(fmin=L_cutoff, fmax=H_cutoff,picks=['C4','C2','C6']);
                pass

        # %% 
        # ### Create Epcohs from events
        # Capture events from annotations
        event_data = mne.events_from_annotations(rawfltrd, verbose=False)
        event_marker, event_ids = event_data
        idx = event_marker[:,-1].argsort()
        event_marker = event_marker[idx,:]
        event_ids = self.dCfg.event_dict# Replacing the existing event ids
        self.classes = list(event_ids.keys())
        # epochs1 = mne.Epochs(rawfltrd, events= event_marker, event_id= event_ids, baseline = (0,0))
        epochs = mne.Epochs(rawfltrd, events= event_marker, tmin= self.dCfg.tmin, tmax=self.dCfg.tmax, event_id= event_ids, on_missing = 'ignore',  
                        verbose= False, proj= True, reject = None, baseline=self.dCfg.baseline, preload = True) # Baseline is default (None,0)
        # epochs.equalize_event_counts() # Shape = epochs x chan x timepnts
        self.rawfltrd = rawfltrd
        # epochs.load_data()
        # epochs._data = self.normalize(epochs.get_data())
        self.epochs = epochs
        # Evoked data
        # T0 = epochs['T0'].average() # Shape = chan x timepnts
        # T1 = epochs['T1'].average()
        # T2 = epochs['T2'].average()

        # %%
        # evoked = epochs.average()
        # if self.plot_enable ==1:
        #     ica.plot_sources(evoked);
        #     ica.plot_overlay(evoked);

        # %%
        # epochs.to_data_frame(index=['condition', 'epoch'],long_format=True)
        if self.plot_enable ==1:
            self.epochs.plot();
            # epochs.plot_drop_log();

        # %%
        ## Data whitening
        if self.plot_enable==1:   
            noise_cov = mne.compute_covariance(self.epochs, tmax=0., method='shrunk', rank=None, verbose='error')                                
            self.epochs[self.classes[0]].plot_white(noise_cov=noise_cov); # step of scaling the whitened plots to show how well the assumption of Gaussian noise is satisfied by the data
        if save_epochs == True:
            self.epochs.save(f'main/preproc/{self.dCfg.name}_{method}_epo.fif', overwrite= True)
            print('Epochs saved')

    # Extract the features
    def feature_extraction(self,method, epoch_file = None, save_feat = False):
        frequencies =  np.arange(5,40,1) #np.logspace(*np.log10([5, 30]), num=25)
        chpicks = [1] #baseline = (-0.5, 0.)
        if epoch_file !=None:
            # self.epochs.delete()
            self.epochs = mne.read_epochs(epoch_file)
            self.classes = list(self.epochs.event_id.keys())
            # self.epochs = self.epochs[self.classes[:2]] # for binary classification
            
        # self.epochs.load_data()
        self.labels = self.epochs.events[:,-1]
        if not self.labels.__contains__(0):
            self.labels = self.labels-1
        # if len(np.unique(self.labels)) ==3: # for binary classification
        #     self.labels = np.concatenate([np.zeros(self.epochs[self.classes[0]].get_data().shape[0]), 
        #                             np.ones(self.epochs[self.classes[1]].get_data().shape[0])])
        # %%
        if method.find('_RAW')!=-1:
            self.features = self.epochs.get_data()

        # %% 
        ## Time- Frequency
        if method.find('_TF')!=-1:
            # 1/f removal- Power Normalization - Epochs TFR
            baset = [self.epochs.baseline[1] , -0.2]
            power = tfr_morlet(self.epochs, freqs = frequencies, n_cycles = frequencies/2, return_itc= False, average=False)
            shp = power.data.shape
            powerdB = np.zeros((shp[0],shp[2],shp[3]))
            powerEp = powerdB.copy()
            t = np.arange(np.abs(baset[1]-baset[0])*self.epochs.info['sfreq'], dtype = np.int64)
            for ep in range(power.data.shape[0]):
                EPpower  = np.mean(power.data[ep,:,:,:],axis=0)
                for f in range(power.data.shape[2]):
                    baseline = np.mean(EPpower[f,t])
                    activity = EPpower[f,:]
                    powerdB[ep,f,:] = 10*np.log10(activity/baseline)
                    powerEp[ep,f,:] = activity

            powerdB = self.normalize(powerdB.copy())
            self.features = powerdB
            # features = features.reshape(features.shape[0],-1)           
            # print(self.features.shape)
            # print(self.labels.shape)

        # %%
        ## Common Spatial Patterns
        if method.find('_CSP')!=-1:
            self.feat_ext = CSP(n_components=self.dCfg.csp_n_comp, reg='ledoit_wolf', log=True, transform_into= 'average_power', #log=None, norm_trace=False, rank='full',cov_est = 'epoch')#
                        cov_est = 'concat',rank='full', norm_trace= True)
            # self.features = self.feat_ext.fit_transform(self.epochs.get_data(), self.labels)
            self.features = self.epochs.get_data()

        ## Wavelet Scattering Transform
        if method.find('_WST')!=-1:
            M,N = self.epochs.get_data().shape[1:]
            self.feat_ext = Scattering2D(J=self.dCfg.wst_scale,shape=(M,N),L=self.dCfg.wst_noAngles)
            # self.epoch_data = self.epochs.get_data().reshape(self.epochs.get_data().shape[0],-1)
            self.features = self.epochs.get_data()

        ## Mean, skewness, variance, kurtosis, zerocrossing, area under signal, peak to peak, 
        ## amplitude spectral density, power spectral density, power of each freq band
        if method.find('_STAT')!=-1:
            # break into segments
            epoch_data = self.epochs.get_data()
            n_segments = 5 #*2
            n_s,n_ch,n_T = epoch_data.shape
            window_size = int((n_T/n_segments)) # 50% overlap
            t = 0
            samples = []
            for ep in range(n_s):
                features = []
                for ch in range(n_ch):
                    ch_feat = []
                    seg_size = ((int(window_size//2)))*2 + 8
                    for t in range(0,n_T, int(window_size//2)):
                        seg = epoch_data[ep,ch,t:t+window_size]
                        # mean
                        M = np.mean(seg)# , axis = 2)
                        # variance
                        V = np.var(seg) #, axis =2)
                        # skew
                        S = stats.skew(seg) #, axis = 2)
                        # kurtosis
                        K = stats.kurtosis(seg)#, axis = 2)
                        # zerocrossing
                        Z = len(np.where(seg>0)[0])
                        # area under signal
                        A = np.trapz(seg, dx =1 )#, axis =2)
                        # p2p
                        P = np.max(seg) -np.min(seg)#, axis =2) - np.min(seg, axis =2)
                        # Amp spectral den
                        Psd = signal.periodogram(seg)[1]#, axis =2)
                        Asd = np.sqrt(Psd)
                        Psd = ((Psd - Psd.min()) / (Psd.max() - Psd.min()).tolist())[1:]
                        Asd = ((Asd - Asd.min()) / (Asd.max() - Asd.min()).tolist())[1:]
                        # power 
                        PFB = np.trapz(Psd)#, axis=2)
                    
                        seg = [M,V,S,K,Z,A,P,*Psd,*Asd,PFB]
                        if len(seg) == seg_size:
                            ch_feat.append(list(seg))
                            
                    features.append(ch_feat)
                samples.append(features)
            self.features = np.array(samples)

            n_s,n_ch,n_seg,n_f = self.features.shape
            # Feature Normalisation acrosss channels for spatial info
            for ep in range(n_s):
                for ch in range(n_ch):
                    for f in range(n_f):
                        self.features[ep,ch,:,f] = (self.features[ep,ch,:,f] - self.features[ep,ch,:,f].min(axis=-1)) / (self.features[ep,ch,:,f].max(axis=-1) - self.features[ep,ch,:,f].min(axis=-1))

        if method.find('_IMG')!=-1:
            # nchan = 16
           
            map = np.zeros((7,8))
            positions = [[0,3], [0,4], [1,2], [1,5], 
                    [2,1],[2,6],[3,0],[3,2], [3, 5], [6,7],
                    [4,1],[4,6], [5,2], [5,5],[6,3], [6,4]]
            map_idx = [3,4,10,13,17,22,24,26,29,31,33,38,42,45,51,52]
            epoch = self.epochs.load_data()
            epoch = self.epochs.reorder_channels(bm.oci_Channels)
            data = epoch.get_data()
            features = np.zeros((data.shape[0], data.shape[-1],7,8))
            for ep in range(data.shape[0]):
                for t in range(data.shape[-1]):
                    np.put(map,map_idx,data[ep,:,t])
                    features[ep, t,:,:] =  map 

            features = [features[:,i : i + self.dCfg.IMG_size] for i in range(0, len(features), self.dCfg.IMG_ovrlp)]
            try:
                features = np.stack(features).swapaxes(0,1) # ep x n_seg x step x *IMG
            except ValueError:
                features = features[:-1]
                features = np.stack(features).swapaxes(0,1) 
            
            sh = features.shape
            self.features = features.reshape(sh[0]*sh[1], sh[2], sh[3], sh[4])# ep x (n_segxstep) x n_row x n_col
            self.labels = self.labels.repeat(sh[1])

        # Test train split
        if self.features.shape[0] > 3:
            self.train_x, self.test_x,self.train_y,self.test_y = train_test_split(self.features, self.labels, 
                                    test_size= self.dCfg.test_split, stratify= self.labels, random_state= 42)

            if len(self.train_x.shape) <4:
                self.train_x = np.expand_dims(self.train_x, axis=1)
                self.test_x = np.expand_dims(self.test_x, axis=1)
        # self.train_x = (self.train_x - self.train_x.min()) / (self.train_x.max() - self.train_x.min())
        # self.test_x = (self.test_x - self.test_x.min()) / (self.test_x.max() - self.test_x.min())

            method = method.split('_')[:-2]
            method = '_'.join(method)
            if save_feat == True:
                np.savez(f'data/train/{self.dCfg.name}_{method}',self.train_x, self.train_y)
                np.savez(f'data/test/{self.dCfg.name}_{method}',self.test_x, self.test_y)            
                print(f'Features saved to: data/test|train/{self.dCfg.name}_{method}')
            
        # %%
        # classify
    def classifier(self, method, save_results = True, save_model = True, feat_file = None):
        if feat_file is not None:
            datafile = np.load(f'{feat_file}')
            self.train_x = datafile['arr_0']
            self.train_y = datafile['arr_1']

        if method.find('_CNN')!=-1:
            # self._matrix2Image()

            batch_size = 3
            learning_rate = 0.001

            # Add channel dim for convolutions
            if len(self.train_x.shape) <4:
                self.train_x = np.expand_dims(self.train_x, axis=1)
            # Input data & labels
            data = DataContainer(self.train_x, self.train_y, self.nCfg)
            _,_,n_chan,n_T = data.x.shape
            n_classes = np.unique(self.train_y).shape[0]

            # Model
            model = eval(method.split('_')[-2:-1][0])
            model = model(n_classes, n_chan, n_T, self.dCfg.sfreq)#.float()
            # Loss function
            if n_classes ==2:
                loss = torch.nn.BCEWithLogitsLoss()
            else:
                loss = torch.nn.CrossEntropyLoss()

            # Observe that all parameters are being optimized
            optimizer = optim.SGD(model.parameters(), lr= self.nCfg.lr, momentum= self.nCfg.optim_moment)

            # Decay LR by a factor of 0.1 every 7 epochs
            LRscheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

            # # Optimizer
            # optimizer = torch.optim.Adam(model.parameters())

            tb_info = f'runs/{model._get_name()}/{self.dCfg.name}/{method}'

            # train and test
            train_and_validate(data,model,loss,optimizer, LRscheduler, tb_info, self.dCfg, self.nCfg,epochs = 100)

            if save_model:
                torch.save(model, f"/media/mangaldeep/HDD2/workspace/MotionControl_MasterThesis/models/{model._get_name()+'_modified.pt'}")

        if method.find('_ML')!=-1:
            cv = ShuffleSplit(10, test_size = 0.2, random_state=1)
            # cv_split = cv.split(self.epochs.get_data(), self.train_y)
            # rbf = RBFSampler(gamma=1, random_state=1)
            clf = eval(method.split('_')[-2:-1][0])
            # clf = SVC()
            # pipe = Pipeline([('CSP', self.feat_ext),('CLF', clf)])
            pipe = Pipeline([('scatter', self.feat_ext), ('clf', clf())])
            scores = cross_val_score(pipe, self.train_x , self.train_y, cv = cv, verbose=False)
            class_balance = np.mean(self.train_y == self.train_y[0])
            class_balance = max(class_balance, 1. - class_balance)
            print("Classification accuracy: %f / Chance level: %f" % (np.mean(scores),class_balance))
                           

if __name__ =='__main__':
    '''
    runs = [3, 4, 7, 8, 11, 12]
    person_id = 1
    raw = extractBCI3(runs , person_id)
    raw = extractPhysionet(runs, person_id)
    '''
    data_cfg = OCIParams()
    raw = extractOCI([], 1)
    analy_cfg = globalTrial()
    net_cfg = EEGNetParams()
    artifact_removal_methods =  'locl_ssp_car_ica'
    feat_extract_methods = artifact_removal_methods+'_TF'
    classi_methods = feat_extract_methods+'_CNN'
    
    methods = 'locl_IMG_EEGnet_CNN'
    methods = f'locl_ssp_car_ica_RAW_{0}_{0}'

    start = time()
    bsa = BrainSignalAnalysis(raw,data_cfg, analy_cfg, net_cfg)

    bsa.artifact_removal(methods, save_epochs = False)
    bsa.feature_extraction(methods, save_feat = True)#,
        # epoch_file = '/media/mangaldeep/HDD2/workspace/MotionControl_MasterThesis/main/preproc/BCI3_ssp_car_ica_3_P3_epo.fif')
    # bsa.classifier(methods, save_model = False)#,
    #     feat_file = '/media/mangaldeep/HDD2/workspace/MotionControl_MasterThesis/main/feature_extraction/Train_locl_ssp_car_ica_TF_EEGnet_CNN_[3]_1.npz')
    print(f"Overall time: {time() - start}")