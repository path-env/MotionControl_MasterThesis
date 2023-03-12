
#%% Parameters for NeuroTechNet
class NeuroTechNetParams():
    def __init__(self):        
        self.eeg_sample_count = 240 # How many samples are we training
        self.learning_rate = 1e-3 # How hard the network will correct its mistakes while learning
        self.eeg_sample_length = 226 # Number of eeg data points per sample
        self.number_of_classes = 1 # We want to answer the "is this a P300?" question
        self.hidden1 = 500 # Number of neurons in our first hidden layer
        self.hidden2 = 1000 # Number of neurons in our second hidden layer
        self.hidden3 = 100 # Number of neurons in our third hidden layer
        self.output = 10 # Number of neurons in our output layer

class EEGNetParams():
    def __init__(self) -> None:
        self.name = 'EEGNet'
        self.epochs = 10
        self.val_split = 0.40
        self.test_split = 0.5
        self.val_bs = 1
        self.train_bs = 1
        self.eeg_sample_count = 240 # How many samples are we training
        self.lr = 0.00153835 # How hard the network will correct its mistakes while learning
        self.gamma = 0.1
        self.optim_moment= 0.9
        self.num_wrkrs = 3
        self.eeg_sample_length = 226 # Number of eeg data points per sample
        self.step_size = 100

class TFNetParams():
    def __init__(self) -> None:
        self.name = 'TFNet'
        self.epochs = 100
        self.val_split = 0.25
        self.test_split = 0.1
        self.val_bs = 3
        self.train_bs = 3
        self.eeg_sample_count = 240 # How many samples are we training
        self.lr = 0.001 # How hard the network will correct its mistakes while learning
        self.gamma = 0.1
        self.optim_moment= 0.9
        self.num_wrkrs = 3
        self.eeg_sample_length = 226 # Number of eeg data points per sample
        self.step_size = 50

class ATTNnetParams():
    def __init__(self) -> None:
        self.name = 'ATTNnet'
        self.epochs = 300
        self.val_split = 0.25
        self.test_split = 0
        self.val_bs = 1
        self.train_bs = 1
        self.eeg_sample_count = 240 # How many samples are we training
        self.lr = 0.01 # How hard the network will correct its mistakes while learning
        self.gamma = 0.1
        self.optim_moment= 0.9
        self.num_wrkrs = 3
        self.eeg_sample_length = 226 # Number of eeg data points per sample
        self.step_size = 50

class CasCnnRnnnetParams():
    def __init__(self) -> None:
        self.name = 'CasCnnRnnnet'
        self.epochs = 100
        self.val_split = 0.25
        self.test_split = 0
        self.val_bs = 1
        self.train_bs = 1
        self.eeg_sample_count = 240 # How many samples are we training
        self.lr = 0.1 # How hard the network will correct its mistakes while learning
        self.gamma = 0.1
        self.optim_moment= 0.9
        self.num_wrkrs = 3
        self.eeg_sample_length = 226 # Number of eeg data points per sample
        self.step_size = 50

######################################################
# Data configuration parameters
#%% Physionet Params
class PhysionetParams():
    def __init__(self) -> None:
        self.name ='Physionet'
        self.sfreq = 125# resampled for OpenBBCI headset #160 #HZ
        self.inion = ['Iz']
        self.elec_lines_f = 60 #HZ
        self.ssp_n_eeg = 2 # No . of EEG SSP components
        self.ica_n_comp = 10
        self.csp_n_comp = 4
        self.eog_ref_ch =  ['Fp1','Fp2']#['Fpz']
        self.event_dict = dict({'T0':1, 'T1':2, 'T2':3}) 
        self.baseline = (None,0)
        self.wst_scale, self.wst_noAngles = 2,8
        self.tmin, self.tmax = -2,4
        self.feat_tmin, self.feat_tmax = 1,4
        self.test_split = 0.1
        self.IMG_size = 10
        self.IMG_ovrlp = 5
#%% BCI Params
class BCI3Params():
    def __init__(self) -> None:
        self.name ='BCI3IVa'
        self.sfreq = 125# resampled for OpenBBCI headset #100 #HZ
        self.inion = ['I1', 'I2']
        self.elec_lines_f = 49 #HZ  # Error with filtering
        self.ssp_n_eeg = 2 # No . of EEG SSP components
        self.ica_n_comp = 10
        self.csp_n_comp = 4
        self.eog_ref_ch = ['Fp1','Fp2']#['Fpz','Fp1','Fp2']
        self.event_dict = dict(right = 1, foot = 2) #, test = 2)
        self.baseline = (None,0)
        self.wst_scale, self.wst_noAngles = 2,8
        self.tmin, self.tmax =-1,2 #-0.5, 1
        self.feat_tmin, self.feat_tmax = -0.5, 1
        self.test_split = 0.1
        self.IMG_size = 10
        self.IMG_ovrlp = 5

#%% OpenBCI Head gear params
class OCIParams():
    def __init__(self) -> None:
        self.name ='OCIParams'
        self.sfreq = 125#
        self.inion = ['']
        self.elec_lines_f = 1#HZ
        self.ssp_n_eeg = 2 # No . of EEG SSP components
        self.ica_n_comp = 10
        self.csp_n_comp = 4
        self.eog_ref_ch =  ['Fp1','Fp2']#['Fpz']
        self.event_dict_rec = dict({'right':0, 'left':1, 'none':2})
        self.event_dict = dict({'right':1, 'left':2, 'none':3})
        self.baseline = (None, -2)
        self.wst_scale, self.wst_noAngles = 2,8
        self.tmin, self.tmax = -6,3
        self.feat_tmin, self.feat_tmax = -1,3
        self.test_split = 0.2
        self.IMG_size = 10
        self.IMG_ovrlp = 5


#%% Analysis configuration Partameters
class globalTrial():
    def __init__(self) -> None:
        self.delta,self.theta,self.alpha,self.beta,self.gamma = [0,3],[3,6],[6,12],[12,25],[25,50]
        self.mu_rhythm, self.beta_rhythm = [7,13], [13,39]
