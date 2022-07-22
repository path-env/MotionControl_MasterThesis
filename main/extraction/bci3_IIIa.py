#%%
# as png
# %matplotlib inline 
#  interactable inside ide
# %matplotlib widget
### interactable seperate window
%matplotlib tk 

import mne
import numpy as np
import sys
sys.path.append('/media/mangaldeep/HDD2/workspace/MotionControl_MasterThesis')
from scipy.io import loadmat
# %% [markdown]
# ### BCI3_IIIa = 'https://www.bbci.de/competition/iii/desc_IIIa.pdf'
#     ''' 
#     4 classes (left hand, right hand, foot, tongue)
#     60 - Channel EEG
#     3 - Volunteers
#     250- Samples/sec (filterd down to 1 to 50 HZ)
#     60 - Experiments per class
#     

# TIme: 
# 0 to 2 : Blank Screen 
# 2 to 3 : Beep for attention
# 3 to 4 : Directional arrow (command for imagery)
# 4 to 7 : Imgery task period 

#     Annotataions:
#         T0 corresponds to rest
#         T1 corresponds to onset of motion (real or imagined) of
#             the left fist (in runs 3, 4, 7, 8, 11, 12)
#             both fists (in runs 5, 6, 9, 10, 13, 14)
#         T2 corresponds to onset of motion (real or imagined) of
#             the right fist (in runs 3, 4, 7, 8, 11, 12)
#             both feet (in runs 5, 6, 9, 10, 13, 14)
#     '''
#%%

# def extractBCI3(person_id = [0]):
person_id = [0]
files = ['/media/mangaldeep/HDD3/DataSets/BCI3_Competition/data_set_IVa_aa_mat/100Hz/data_set_IVa_aa.mat',
'/media/mangaldeep/HDD3/DataSets/BCI3_Competition/data_set_IVa_al_mat/100Hz/data_set_IVa_al.mat',
'/media/mangaldeep/HDD3/DataSets/BCI3_Competition/data_set_IVa_av_mat/100Hz/data_set_IVa_av.mat',
'/media/mangaldeep/HDD3/DataSets/BCI3_Competition/data_set_IVa_aw_mat/100Hz/data_set_IVa_aw.mat',
'/media/mangaldeep/HDD3/DataSets/BCI3_Competition/data_set_IVa_ay_mat/100Hz/data_set_IVa_ay.mat',]


# files = ['/media/mangaldeep/HDD3/DataSets/BCI3_Competition/BCI_3a/BCI3_3a2.gdf',
#         '/media/mangaldeep/HDD3/DataSets/BCI3_Competition/BCI_3a/BCI3_3a3.gdf']
n_channels = 32
sampling_freq = 200  # in Hertz
info = mne.create_info(n_channels, sfreq=sampling_freq)
print(info)

for i in person_id:
    matdata = loadmat(files[i])
    raw = mne.io.RawArray(matdata, info)
# raw = [mne.io.read_raw_gdf(files[i], preload = True) for i in person_id]
# raw  = mne.concatenate_raws(raw)

    # return raw
#%%
# raw = extractBCI3(person_id = [0])


# %%
raw.plot_psd();
# %%
