# %%
import os
import sys
sys.path.append('/media/mangaldeep/HDD2/workspace/MotionControl_MasterThesis')


# %%

import mne
# %% [markdown]
# ### physionet = 'https://physionet.org/files/eegmmidb/1.0.0/'
#     ''' 
#     64 - Channel EEG
#     109 - Volunteers
#     160- Samples/sec
#     14 - Experiments (Each recording 1 to 2 minutes) 
#             T1: Left/Fist,    T2: Right/Feet
#             1.Baseline, eyes open (T0)
#             2.Baseline, eyes closed (T0)
#             3.Task 1 (open and close left or right fist) (T1/T2)
#             4.Task 2 (imagine opening and closing left or right fist) (T1/T2)
#             5.Task 3 (open and close both fists or both feet) (T1/T2)
#             6.Task 4 (imagine opening and closing both fists or both feet) (T1/T2)
#             7.Task 1 (T1/T2)
#             8.Task 2 (T1/T2)
#             9.Task 3 (T1/T2)
#             10.Task 4 (T1/T2)
#             11.Task 1 (T1/T2)
#             12.Task 2 (T1/T2)
#             13.Task 3 (T1/T2)
#             14.Task 4 (T1/T2)
#     Annotataions:
#         T0 corresponds to rest
#         T1 corresponds to onset of motion (real or imagined) of
#             the left fist (in runs 3, 4, 7, 8, 11, 12)
#             both fists (in runs 5, 6, 9, 10, 13, 14)
#         T2 corresponds to onset of motion (real or imagined) of
#             the right fist (in runs 3, 4, 7, 8, 11, 12)
#             both feet (in runs 5, 6, 9, 10, 13, 14)
#     '''

# %%
# Mapping to the task in  hand
'''
T1 - Left = Take Left
T2 - Right = Take Right
T1 - Fist = Brake
T2 - Feet = Accelerate
'''
def extract(runs, person_id):
    # %%
    # runs = list(range(1,15))
    fistLR_openclose = [1,2,3,4,7,8,11,12]
    fist_feet_openclose = [1,2,5,6,9,10,13,14]
    # person_id = 5
    # path2 = '/media/mangaldeep/HDD3/DataSets/Physionet'
    path = '/media/mangaldeep/HDD3/DataSets/mne_data'
    fname = mne.datasets.eegbci.load_data(person_id, runs, path=path)

    # %%
    # runs = [3, 4, 7, 8, 11,12]
    raw = [mne.io.read_raw_edf(fname[i], preload=True, verbose=False) for i,_ in enumerate(runs)]
    # raw = mne.io.read_raw_edf(fname[run_id], preload=True, verbose=False)
    raw = mne.concatenate_raws(raw)
    # %% [markdown]
    # #### Channel Name Modification

    # %%
    # Dataset lacks EEG electrode locations. 
    ten_twenty_montage = mne.channels.make_standard_montage('standard_1020')

    # Montage channel names are different from dataset
    # FP3 --> Fp3
    ch_loc = ten_twenty_montage._get_ch_pos()
    ch_names = ten_twenty_montage.ch_names

    new_names = []
    for ch in ch_names:
        ch = ch.lower()
        new_names.append(ch[0].upper() + ch[1:])

    ten_twenty_montage.ch_names = new_names

    # Remove dots from channel names in raw.info['ch_names']
    new_names = []
    for ch_name in raw.info['ch_names']:
        new_names.append(ch_name.split('.')[0])

    mne.rename_channels(raw.info, dict(zip(raw.ch_names, new_names)))
    # Set Montage 10_20 system
    raw.set_montage(ten_twenty_montage, verbose=False);

    # Sphere not oriented with 10-20 system
    # raw.plot_sensors(kind = '3d', sphere=(0.0, 0.015, 0.033, 0.1));
    # %%
    return raw


if __name__== "__main__":
    runs = [3, 4, 7, 8, 11,12]
    person_id = 1
    raw = extract(runs, person_id)
    print(raw)
