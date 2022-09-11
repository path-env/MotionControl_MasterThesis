# %%
# as png
# %matplotlib inline 
#  interactable inside ide
# %matplotlib widget
### interactable seperate window
# %matplotlib tk 

# %%
import sys
from matplotlib import cm
sys.path.append('/media/mangaldeep/HDD2/workspace/MotionControl_MasterThesis')

# %%
import mne
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

#Local Imports
# from main.extraction.physionet_MI import  extractPhysionet, extractBCI3
from data import brain_atlas  as bm
from data.params import BCI3Params, EEGNetParams, PhysionetParams, globalTrial
# Data Extraction
from main.extraction.bci3_IVa import extractBCI3
from main.extraction.physionet_MI import extractPhysionet

# Feature Extraction
from mne.time_frequency import tfr_morlet
from mne.decoding import CSP

from kymatio.sklearn import Scattering2D

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

from main.extraction.data_extractor import data_container
from models.neurotec_edu import Neurotech_net
from models.MI_CNN_WT_2019 import TFnet
from models.EEGNet_2018 import EEGnet
from models.train_net import train_and_validate

# %%


class BrainSignalAnalysis():
    def __init__(self, raw, data_cfg, analy_cfg, net_cfg,runs, person_id) -> None:
        self.dCfg = data_cfg
        self.aCfg = analy_cfg
        self.nCfg = net_cfg
        self.runs = runs
        self.person_id = person_id

        # Parameters
        elec_lines_f, L_cutoff, H_cutoff = self.dCfg.elec_lines_f, self.aCfg.mu_rhythm[0], self.aCfg.beta_rhythm[-1] #HZ

        # %%
        self.plot_enable = 0
        ica_ssp_comp = 0

        # %%
        # Remove and Filter signal noises
        raw.notch_filter(elec_lines_f)
        rawfltrd = raw.filter(L_cutoff, H_cutoff, verbose= False, fir_design='firwin', skip_by_annotation='edge').copy()

        # Referncing to reference electrodes
        rawfltrd = rawfltrd.set_eeg_reference(self.dCfg.inion)
        # Check the Power spectral density
        # rawfltrd.plot_psd();

        pick_ch = bm.C_Channels  # Considering only the channels that map to topo map functionality
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
            # self.train_data = scaler.fit_transform(self.train_data)
            cmap = np.uint8(cm.gist_earth(self.train_data)*255)[:,:,:,:3]
            cmap = np.swapaxes(cmap, 1,3)
            cmap = np.swapaxes(cmap, 2,3)

            preprocess = transforms.Compose([
                # transforms.ToPILImage(),
                transforms.Resize(256),
                transforms.CenterCrop(227),
                # transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

            self.train_data = torch.from_numpy(cmap.astype(np.float32))
            # input_image = Image.fromarray(cmap)
            input_tensor = preprocess(self.train_data)
            input_tensor-=input_tensor.min()
            input_tensor/=input_tensor.max()
            self.train_data = input_tensor #.unsqueeze(0) # create a mini-batch as expected by the model

            # # Plot the proceessed image
            # import cv2
            # test = self.train_data[0,:,:,:].swap_axes(0,2).numpy()
            # cv2.imshow('Test', test)
            # cv2.waitKey(0)
            return None

        # %%
        # Remove the artifacts
    def artifact_removal(self, method, save_epochs = False):
        rawfltrd = self.rawfltrd.copy()
        if method.find('locl')!= -1:
            pick_ch = bm.C_Channels  # Considering only the channels that map to topo map functionality
            rawfltrd.pick_channels(pick_ch); # inplace

        if method.find('ssp')!=-1:
            eog_proj, events = mne.preprocessing.compute_proj_eog(rawfltrd, n_grad=0, n_mag=0, n_eeg=self.dCfg.ssp_n_eeg, average=True, verbose=False, ch_name=  self.dCfg.eog_ref_ch, reject=None) # returns EOG Proj and events of blinks
            rawfltrd.add_proj(projs=eog_proj);

        # ##### Common Average Reference(CAR) (Projector)
        ## TODO- Remove bads prior
        if method.find('car')!=-1:
            rawfltrd.set_eeg_reference(ref_channels='average', projection=True);
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
            ica = mne.preprocessing.ICA(n_components=self.dCfg.ica_n_comp, noise_cov= None, random_state=2, method='picard',max_iter=500)
            # Create an instance of RAW
            # rawfltrd_ica = raw.copy()
            rawfltrd.apply_proj()
            ica.fit(rawfltrd);
            # n_components  = 10 then ica.exclude = [1,2]
            ica.exclude = []
            # Using EOG Channel to select ICA Components
            ica.exclude , ex_scores = ica.find_bads_eog(rawfltrd, ch_name= self.dCfg.eog_ref_ch);#,threshold=2);

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
            rawfltrd = ica.apply(rawfltrd) # Reconstructed sensor data (In Senso space)
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
        event_data = mne.events_from_annotations(rawfltrd)
        event_marker, event_ids = event_data
        idx = event_marker[:,-1].argsort()
        event_marker = event_marker[idx,:]
        event_ids = self.dCfg.event_dict# Replacing the existing event ids
        self.classes = list(event_ids.keys())
        # epochs1 = mne.Epochs(rawfltrd, events= event_marker, event_id= event_ids, baseline = (0,0))
        self.epochs = mne.Epochs(rawfltrd, events= event_marker, tmin= self.dCfg.tmin, tmax=self.dCfg.tmax, event_id= event_ids, verbose= True, proj= True, reject = None) # Baseline is default (None,0)
        self.epochs.equalize_event_counts() # Shape = epochs x chan x timepnts

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
        noise_cov = mne.compute_covariance(self.epochs, tmax=0., method='shrunk', rank=None, verbose='error')
        if self.plot_enable==1:                                   
            self.epochs[self.classes[0]].plot_white(noise_cov=noise_cov); # step of scaling the whitened plots to show how well the assumption of Gaussian noise is satisfied by the data
        if save_epochs == True:
            self.epochs.save(f'main/preproc/{self.dCfg.name}_{method}_{self.runs}_P{self.person_id}_epo.fif', overwrite= True)
            print('Epochs saved')

        # %%
        # Extract the features
    def feature_extraction(self,method, epoch_file = None, save_feat = False):
        frequencies =  np.arange(5,30,1) #np.logspace(*np.log10([5, 30]), num=25)
        chpicks = [1] #baseline = (-0.5, 0.)
        if epoch_file !=None:
            # self.epochs.delete()
            self.epochs = mne.read_epochs(epoch_file)
            self.classes = list(self.epochs.event_id.keys())
            # self.epochs = self.epochs[self.classes[:2]] # for binary classification

        self.label = self.epochs.events[:,-1]
        if not self.label.__contains__(0):
            self.label = self.label-1
        # if len(np.unique(self.label)) ==3: # for binary classification
        #     self.label = np.concatenate([np.zeros(self.epochs[self.classes[0]].get_data().shape[0]), 
        #                             np.ones(self.epochs[self.classes[1]].get_data().shape[0])])
        # %%
        if method.find('_RAW')!=-1:
            self.train_data = self.epochs.get_data()

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

            self.train_data = powerdB
            # train_data = train_data.reshape(train_data.shape[0],-1)           
            # print(self.train_data.shape)
            # print(self.label.shape)

        # %%
        ## Common Spatial Patterns
        if method.find('_CSP')!=-1:
            self.feat_ext = CSP(n_components=self.dCfg.csp_n_comp, reg='ledoit_wolf', log=True, transform_into= 'average_power', #log=None, norm_trace=False, rank='full',cov_est = 'epoch')#
                        cov_est = 'concat',rank='full', norm_trace= True)
            # self.train_data = self.feat_ext.fit_transform(self.epochs.get_data(), self.label)
            self.train_data = self.epochs.get_data()

        ## Wavelet Scattering Transform
        if method.find('_WST')!=-1:
            M,N = self.epochs.get_data().shape[1:]
            self.feat_ext = Scattering2D(J=self.dCfg.wst_scale,shape=(M,N),L=self.dCfg.wst_noAngles)
            # self.epoch_data = self.epochs.get_data().reshape(self.epochs.get_data().shape[0],-1)
            self.train_data = self.epochs.get_data()

        self.train_data = (self.train_data - self.train_data.min()) / (self.train_data.max() - self.train_data.min())

        if save_feat == True:
            np.savez(f'main/feature_extraction/{self.dCfg.name}_{method}_{self.runs}_{self.person_id}',self.train_data, self.label)
            print(f'Features saved to: main/feature_extraction/{self.dCfg.name}_{method}_{self.runs}_{self.person_id}')
            
        # %%
        # classify
    def classifier(self, method, save_results = True, save_model = True, feat_file = None):
        if feat_file is not None:
            datafile = np.load(f'{feat_file}')
            self.train_data = datafile['arr_0']
            self.label = datafile['arr_1']

        if method.find('_CNN')!=-1:
            # self._matrix2Image()

            batch_size = 3
            learning_rate = 0.001

            # Add channel dim for convolutions
            if len(self.train_data.shape) <4:
                self.train_data = np.expand_dims(self.train_data, axis=1)
            # Input data & labels
            data = data_container(self.train_data, self.label, self.nCfg)
            _,_,n_chan,n_T = data.x.shape
            n_classes = np.unique(self.label).shape[0]

            # Model
            model = eval(method.split('_')[-2:-1][0])
            model = model(n_classes, n_chan,self.dCfg.sfreq)#.float()
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
            # cv_split = cv.split(self.epochs.get_data(), self.label)
            # rbf = RBFSampler(gamma=1, random_state=1)
            clf = eval(method.split('_')[-2:-1][0])
            # clf = SVC()
            # pipe = Pipeline([('CSP', self.feat_ext),('CLF', clf)])
            pipe = Pipeline([('scatter', self.feat_ext), ('clf', clf())])
            scores = cross_val_score(pipe, self.train_data , self.label, cv = cv, verbose=False)
            class_balance = np.mean(self.label == self.label[0])
            class_balance = max(class_balance, 1. - class_balance)
            print("Classification accuracy: %f / Chance level: %f" % (np.mean(scores),class_balance))
                           

if __name__ =='__main__':
    runs = [3]# [3, 4, 7, 8, 11, 12]
    person_id = 1
    data_cfg = BCI3Params()
    analy_cfg = globalTrial()
    net_cfg = EEGNetParams()
    # artifact_removal_methods =  'ssp_car_ica'
    # feat_extract_methods = artifact_removal_methods+'_TF'
    # classi_methods = 'EEGnet_CNN'
    methods = 'locl_ssp_car_ica_RAW_EEGnet_CNN'
    methods = 'locl_ssp_car_ica_WST_LDA_ML'
    raw = extractBCI3(runs , person_id)
    # raw = extractPhysionet(runs, person_id)
    
    bsa = BrainSignalAnalysis(raw,data_cfg, analy_cfg, net_cfg, runs, person_id)

    bsa.artifact_removal(methods, save_epochs = False)
    bsa.feature_extraction(methods, save_feat = True)#,
        # epoch_file = '/media/mangaldeep/HDD2/workspace/MotionControl_MasterThesis/main/preproc/BCI3_ssp_car_ica_3_P3_epo.fif')
    bsa.classifier(methods, save_model = True)#,
    #     feat_file = '/media/mangaldeep/HDD2/workspace/MotionControl_MasterThesis/main/feature_extraction/Train_locl_ssp_car_ica_TF_EEGnet_CNN_[3]_1.npz')