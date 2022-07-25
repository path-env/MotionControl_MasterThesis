# %%
# as png
# %matplotlib inline 
#  interactable inside ide
# %matplotlib widget
### interactable seperate window
# %matplotlib tk 

# %%
import sys
sys.path.append('/media/mangaldeep/HDD2/workspace/MotionControl_MasterThesis')
# os.getcwd()

# %%
import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Local Imports
# from main.extraction.physionet_MI import  extractPhysionet, extractBCI3
from data import brain_atlas  as bm

#%%
class Preproc():
    def __init__(self, raw, data_cfg, analy_cfg) -> None:
        pick_ch = bm.C_Channels  # Considering only the channels that map to topo map functionality
        raw.pick_channels(pick_ch); # inplace

        # Parameters
        elec_lines_f, L_cutoff, H_cutoff = data_cfg.elec_lines_f, analy_cfg.mu_rhythm[0], analy_cfg.beta_rhythm[-1] #HZ

        # Channel names to Indices
        ch_names = raw.ch_names
        self.pick_ch_idx = [ch_names.index(i) for i in pick_ch]

        # Duplicate params
        self.scale = dict(mag=1e-12, grad=4e-11, eeg=100e-6)
        mne_plot_raw = dict(scalings=self.scale, clipping='transparent', order=self.pick_ch_idx)

        # %%
        self.plot_enable = 0
        ica_ssp_comp = 0

        # %%
        # Remove and Filter signal noises
        raw.notch_filter(elec_lines_f)
        rawfltrd = raw.filter(L_cutoff, H_cutoff, verbose= False, fir_design='firwin', skip_by_annotation='edge').copy()

        # Referncing to reference electrodes
        # rawfltrd = rawfltrd.set_eeg_reference(['Iz'])
        # Check the Power spectral density
        # rawfltrd.plot_psd();

        # %%
        ## Analysis after Spectral filter
        if self.plot_enable ==1:
            rawfltrd.plot(scalings=self.scale, clipping='transparent'); #, order=self.pick_ch_idx
            # rawfltrd.plot(scalings='auto');
            # rawfltrd.plot_psd_topo(show = True);
            # rawfltrd.plot_sensors(show_names=True,kind = '3d', sphere=(0.0, 0.015, 0.033, 0.1));

        self.rawfltrd = rawfltrd
        # %% [markdown]
        # #### Artifact Removal : Projectors: SSP & CAR

        # %% [markdown]
        # ##### Signal Space Projection(SSP)

        # %%
        # Create an instance of RAW
    def run(self, method, data_cfg, run, patient_id):
        rawfltrd = self.rawfltrd.copy()
        if method.find('ssp')!=-1:
            eog_proj, events = mne.preprocessing.compute_proj_eog(rawfltrd, n_grad=0, n_mag=0, n_eeg=data_cfg.ssp_n_eeg, average=True, verbose=False, ch_name =['Fpz','Fp1','Fp2'], reject=None) # returns EOG Proj and events of blinks
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
            ica = mne.preprocessing.ICA(n_components=data_cfg.ica_n_comp, noise_cov= None, random_state=2, method='picard',max_iter=500)
            # Create an instance of RAW
            # rawfltrd_ica = raw.copy()
            rawfltrd.apply_proj()
            ica.fit(rawfltrd);
            # n_components  = 10 then ica.exclude = [1,2]
            ica.exclude = []
            # Using EOG Channel to select ICA Components
            ica.exclude , ex_scores = ica.find_bads_eog(rawfltrd, ch_name=['Fpz','Fp1','Fp2']);#,threshold=2);

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

        # %% [markdown]
        # ### Create Epcohs from events
        ## TODO: Choose between rawfltrd/ rawfltrd_proj/ rawfltrd_ica
        # Capture events from annotations
        event_data = mne.events_from_annotations(rawfltrd)
        event_marker, event_ids = event_data
        event_ids = data_cfg.event_dict# Replacing the existing event ids
        # epochs1 = mne.Epochs(rawfltrd, events= event_marker, event_id= event_ids, baseline = (0,0))
        epochs = mne.Epochs(rawfltrd, events= event_marker, tmin= data_cfg.tmin, tmax=data_cfg.tmax, event_id= event_ids, verbose= True, proj= True, reject = None) # Baseline is default (None,0)
        # epochs.equalize_event_counts() # Shape = epochs x chan x timepnts

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
            epochs.plot();
            # epochs.plot_drop_log();

        # %%
        ## Data whitening
        noise_cov = mne.compute_covariance(epochs, tmax=0., method='shrunk', rank=None, verbose='error')
        if self.plot_enable==1:                                   
            T1.plot_white(noise_cov=noise_cov); # step of scaling the whitened plots to show how well the assumption of Gaussian noise is satisfied by the data
        epochs.save(f'main/preproc/{data_cfg.name}_{method}_{run}_P{patient_id}_epo.fif', overwrite= True)


# %% [markdown]
# ### Extraction
def preproc(raw):
    pick_ch = bm.C_Channels  # Considering only the channels that map to topo map functionality
    raw.pick_channels(pick_ch); # inplace

    # Parameters
    delta,theta,alpha,beta,gamma = [0,3],[3,6],[6,12],[12,25],[25,50]
    mu_rhythm, beta_rhythm = [7,13], [13,30]
    elec_lines, L_cutoff, H_cutoff = 60, mu_rhythm[0], beta_rhythm[-1] #HZ

    # Channel names to Indices
    ch_names = raw.ch_names
    pick_ch_idx = [ch_names.index(i) for i in pick_ch]

    # Duplicate params
    scale = dict(mag=1e-12, grad=4e-11, eeg=100e-6)
    mne_plot_raw = dict(scalings=scale, clipping='transparent', order=pick_ch_idx)

    # %%
    plot_enable = 0
    ica_ssp_comp = 0

    # %% [markdown]
    # #### Remove Line noise and apply band pass

    # %%
    # Remove and Filter signal noises
    raw.notch_filter(elec_lines)
    rawfltrd = raw.filter(L_cutoff, H_cutoff, verbose= False, fir_design='firwin', skip_by_annotation='edge').copy()

    # Referncing to reference electrodes
    rawfltrd = rawfltrd.set_eeg_reference(['Iz'])
    # Check the Power spectral density
    # rawfltrd.plot_psd();

    # %%
    ## Analysis after Spectral filter
    if plot_enable ==1:
        rawfltrd.plot(scalings=scale, clipping='transparent'); #, order=pick_ch_idx
        # rawfltrd.plot(scalings='auto');
        # rawfltrd.plot_psd_topo(show = True);
        # rawfltrd.plot_sensors(show_names=True,kind = '3d', sphere=(0.0, 0.015, 0.033, 0.1));

    # %% [markdown]
    # #### Artifact Removal : Projectors: SSP & CAR

    # %% [markdown]
    # ##### Signal Space Projection(SSP)

    # %%
    # Create an instance of RAW
    rawfltrd_ssp = rawfltrd.copy()
    eog_proj, events = mne.preprocessing.compute_proj_eog(rawfltrd_ssp, n_grad=0, n_mag=0, n_eeg=2, average=True, verbose=False, ch_name = 'Fpz', reject=None) # returns EOG Proj and events of blinks
    rawfltrd_ssp.add_proj(projs=eog_proj);

    # mne.preprocessing.ssp.make_eeg_average_ref_proj()


    # %% [markdown]
    # ##### Common Average Reference(CAR) (Projector)

    # %%
    ## TODO- Remove bads prior
    rawfltrd_car = rawfltrd_ssp.copy()
    rawfltrd_car.set_eeg_reference(ref_channels='average', projection=True);

    # %%
    ## Analysis after Projections
    if plot_enable ==1:
        # rawfltrd_ssp.plot_projs_topomap();
        # rawfltrd_car.info['projs']
        # Compare after and before projection
        # rawfltrd.plot(scalings=scale, clipping='transparent', order=pick_ch_idx);
        rawfltrd_car.plot(scalings=scale, clipping='transparent', butterfly= False, title='Raw Filtered (Without Projection))', proj= False, order=pick_ch_idx);
        rawfltrd_car.plot(scalings=scale, clipping='transparent', butterfly= False, title='Raw Filtered(With Projection)', proj= True, order=pick_ch_idx);

    # %%
    # rawfltrd_proj = rawfltrd_car.apply_proj()

    # %% [markdown]
    # #### Artifact Removal : Independent Component Analysis(ICA)

    # %%
    # cov = mne.Covariance()
    rawfltrd_ica = rawfltrd_car.apply_proj().copy()
    ica = mne.preprocessing.ICA(n_components = 20, noise_cov= None, random_state=2, method='picard',max_iter=500)
    # Create an instance of RAW
    # rawfltrd_ica = raw.copy()
    # rawfltrd_ica.apply_proj()
    ica.fit(rawfltrd_ica);
    # n_components  = 10 then ica.exclude = [1,2]
    ica.exclude = []
    # Using EOG Channel to select ICA Components
    ica.exclude , ex_scores = ica.find_bads_eog(rawfltrd_ica, ch_name=['Fpz']);#,threshold=2);

    # %%
    ## TODO: ICA template matching for Multiple subjects
    ## TODO: Plot ICA evoked.

    # %%
    ica.exclude = []#=[1,6]
    # ica.plot_scores(ex_scores);
    # ica.plot_properties(rawfltrd_ica, picks = ica.exclude)

    # %%
    ## Analysis after ICA
    if plot_enable ==1:
        # ica.plot_components();
        # ica.plot_sources(rawfltrd_ica);
        # ica.plot_overlay(rawfltrd_ica, exclude = ica.exclude);
        # ica.plot_properties(rawfltrd_ica, picks = ica.exclude);
        ica.plot_scores(ex_scores);

    # %%
     # ica.exclude = [0,1,2,3] # manually exclude ICA components based on observation on plots above
    rawfltrd_ica = ica.apply(rawfltrd_ica) # Reconstructed sensor data (In Senso space)
    if plot_enable ==1:
        rawfltrd_ica.plot(scalings=scale, clipping='transparent', title='ICA  on RAW', proj= False); # , order=pick_ch_idx
        # rawfltrd.plot(scalings=scale, clipping='transparent', title='Raw Filtered- w Projection', proj=True); #, order=pick_ch_idx
        rawfltrd_ssp.plot(scalings=scale, clipping='transparent', title='Raw Filtered- wo Projection', proj=False);# , order=pick_ch_idx

    # %%
    # rawfltrd_proj = rawfltrd_car.copy()
    # rawfltrd_proj.apply_proj()
    # rawfltrd_proj.plot(scalings=scale, clipping='transparent', title='ICA+SSP', proj=False);

    # %%
    if plot_enable ==1:
        # rawfltrd_ica.plot_psd(fmin=L_cutoff, fmax=H_cutoff,picks=['C4','C2','C6']);
        # rawfltrd_proj.plot_psd(fmin=L_cutoff, fmax=H_cutoff,picks=['C4','C2','C6']);
        pass

    # %% [markdown]
    # #### Artifact Removal: SSP vs CAR

    # %%
    ## Compare ICA vs SSP
    if ica_ssp_comp ==1:
        # Comparison using data
        rawfltrd_df = rawfltrd.to_data_frame(index=['time'])
        rawfltrd_proj_df = rawfltrd_proj.to_data_frame(index=['time'])
        rawfltrd_ica_df = rawfltrd_ica.to_data_frame(index=['time'])
        rawfltrd_ica_df.compare(rawfltrd_proj_df, align_axis=0)

    # %%
    # comparison using plots
    if ica_ssp_comp ==1:
        dummy = rawfltrd.copy()
        data1 = rawfltrd.get_data()
        data2 = rawfltrd_ica.get_data()
        data3 = rawfltrd_proj.get_data()
        dummy._data = data3 - data2
        dummy.plot(clipping='transparent', title='DIfference between ICA and SSP', proj=False);

    # %% [markdown]
    # ### Create Epcohs from events

    # %%
    ## TODO: Choose between rawfltrd/ rawfltrd_proj/ rawfltrd_ica
    raw_epoch = rawfltrd_ica.copy()
    # Capture events from annotations
    event_data = mne.events_from_annotations(raw_epoch)
    event_marker, event_ids = event_data
    event_ids = dict({'T1':2, 'T2':3}) # Replacing the existing event ids

    # %%
    event_marker

    # %%
    # epochs1 = mne.Epochs(rawfltrd, events= event_marker, event_id= event_ids, baseline = (0,0))
    epochs = mne.Epochs(raw_epoch, events= event_marker, tmin=-2, tmax=4, event_id= event_ids, verbose= False, proj= True, reject = None) # Baseline is default (None,0)
    # epochs.equalize_event_counts() # Shape = epochs x chan x timepnts

    # Evoked data
    # T0 = epochs['T0'].average() # Shape = chan x timepnts
    T1 = epochs['T1'].average()
    T2 = epochs['T2'].average()

    # %%
    epochs.get_data().shape

    # %%
    evoked = epochs.average()
    if plot_enable ==1:
        ica.plot_sources(evoked);
        ica.plot_overlay(evoked);

    # %%
    raw_epoch.info['projs']

    # %%
    ## Task  Analysis
    if plot_enable==1:
        task = T2
        title = str(task)[11:11+2] + ' ICA'
        # task.plot_topomap();
        # task.plot_white(); # Noise cov required
        # task.plot_field(); # requires  surf maps
        # task.plot_sensors();
        task.plot_topo();
        task.plot_joint(times=[0.0, 0.2, 0.3]);#,picks=['C4','C2','C6','C1','C3','C5']);
        task.plot_image(titles=f'{title} Image',show_names='all');
        # task.plot(proj= True, titles = '{task} - Projs - True',spatial_colors=True);
        # task.plot(proj= False, titles = '{task} -  Projs - False',spatial_colors=True);
        # task.plot(proj= 'reconstruct', titles = '{task} -  Projs - reconstruct',spatial_colors=True);
        # task.plot_topomap();
        # task.plot(gfp= "only"); # population standard deviation of the signal across channels
        ## Compare regions
        # mne.channels.combine_channels({task}, roi_dict, method='mean')
        ## Compare conditions
        # evoked = dict(T1 = list(epochs[title].iter_evoked()), T2 = list(epochs['T2'].iter_evoked()), T0=list(epochs['T0'].iter_evoked()))
        # mne.viz.plot_compare_evokeds(evoked, combine='mean');
        # task_t0 = mne.combine_evoked([task, T0], weights=[1,-1])
        # task_t0.plot_joint();

    # %%
    # epochs.to_data_frame(index=['condition', 'epoch'],long_format=True)
    if plot_enable ==1:
        epochs.plot();
        # epochs.plot_drop_log();

    # %%
    # Amplitudes and latency measures
    channel, latency, amplitude = T1.get_peak(mode='pos', return_amplitude=True)
    print(channel, latency, amplitude)

    # %%
    channel, latency, amplitude = T2.get_peak(mode='pos', return_amplitude=True)
    print(channel, latency, amplitude)

    # %%
    ## Data whitening
    noise_cov = mne.compute_covariance(epochs, tmax=0., method='shrunk', rank=None,
                                       verbose='error')
    if plot_enable==1:                                   
        T1.plot_white(noise_cov=noise_cov); # step of scaling the whitened plots to show how well the assumption of Gaussian noise is satisfied by the data
    

    # %%
    epochs.save('Physionet_ssp_car_ica_epo.fif', overwrite= True)

    # %% [markdown]
    # ### Feature Extraction:


