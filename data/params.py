
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

#%% Physionet Params
class PhysionetParams():
    def __init__(self) -> None:
        self.name ='Physionet'
        self.elec_lines_f = 60 #HZ
        self.ssp_n_eeg = 2 # No . of EEG SSP components
        self.ica_n_comp = 20
        self.event_dict = dict({'T1':2, 'T2':3}) 
        self.tmin, self.tmax =-2, 4

#%% BCI Params
class BCI3Params():
    def __init__(self) -> None:
        self.name ='BCI3'
        self.elec_lines_f = 0.51 #HZ 
        self.ssp_n_eeg = 2 # No . of EEG SSP components
        self.ica_n_comp = 20
        self.event_dict = dict(right = 1, foot = 3, test = 2)
        self.tmin, self.tmax =-0.5, 1

#%% Analysis configuration Partameters
class globalTrial():
    def __init__(self) -> None:
        self.delta,self.theta,self.alpha,self.beta,self.gamma = [0,3],[3,6],[6,12],[12,25],[25,50]
        self.mu_rhythm, self.beta_rhythm = [7,13], [13,30]
