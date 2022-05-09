#%%
## Brain channels location to functions
## 64 Channelss - 10-20 System
FL_Channels = ['Fc5',
 'Fc3',
 'Fc1',
 'Fcz',
 'F7',
 'F5',
 'F3',
 'F1',
 'Fz',
 'Fp1',
 'Fpz',
 'Ft7',
 'Af7',
 'Af3',
 'Afz'
 ]

FR_Channels = ['Fcz',
 'Fc2',
 'Fc4',
 'Fc6',
 'Fz',
 'F2',
 'F4',
 'F6',
 'F8',
 'Fp2',
 'Fpz',
 'Ft8',
 'Afz',
 'Af4',
 'Af8'
 ]

CL_Channels= [ 'C5',
 'C3',
 'C1',
 'Cz',
 'Fc5',
 'Fc3',
 'Fc1',
 'Fcz'
  ]

CR_Channels= ['Cz',
 'C2',
 'C4',
 'C6',
 'Fcz',
 'Fc2',
 'Fc4',
 'Fc6'
 ]

PL_Channels = ['P7',
 'P5',
 'P3',
 'P1',
 'Pz',
 'Cp5',
 'Cp3',
 'Cp1',
 'Cpz',
 'Tp7',
  'Po7',
 'Po3',
 'Poz'
 ]

PR_Channels = ['Pz',
 'P2',
 'P4',
 'P6',
 'P8',
 'Cpz',
 'Cp2',
 'Cp4',
 'Cp6',
 'Tp8',
 'Poz',
 'Po4',
 'Po8'
 ]

O_Channels = [ 'O1',
 'Oz',
 'O2',
  'Po7',
 'Po3',
 'Poz',
  'Po4',
 'Po8',

 ]

TL_Channels = ['Ft7',
'T7','Tp7']

TR_Channels = ['Ft8',
'T8','Tp8']

refchannels = ['T9', 'T10','Iz']

#%%
F_Channels = list(set(FL_Channels + FR_Channels))
C_Channels = list(set(CL_Channels + CR_Channels))
P_Channels = list(set(PL_Channels + PR_Channels))
T_Channels = list(set(TL_Channels + TR_Channels))

# %%
