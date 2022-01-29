
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