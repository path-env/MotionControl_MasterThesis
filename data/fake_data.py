from matplotlib import rcParams
import torch 
import matplotlib.pyplot as plt

from models.neurotec_edu import Neurotech_net
from data.params import NeuroTechNetParams


NTNetparams = NeuroTechNetParams()
## Create sample data using the parameters
sample_positives = [None, None] # Element [0] is the sample, Element [1] is the class
sample_positives[0] = torch.rand(int(NTNetparams.eeg_sample_count / 2), NTNetparams.eeg_sample_length) * 0.50 + 0.25
sample_positives[1] = torch.ones([int(NTNetparams.eeg_sample_count / 2), 1], dtype=torch.float32)

sample_negatives = [None, None] # Element [0] is the sample, Element [1] is the class
sample_negatives_low = torch.rand(int(NTNetparams.eeg_sample_count / 4), NTNetparams.eeg_sample_length) * 0.25
sample_negatives_high = torch.rand(int(NTNetparams.eeg_sample_count / 4), NTNetparams.eeg_sample_length) * 0.25 + 0.75
sample_negatives[0] = torch.cat([sample_negatives_low, sample_negatives_high], dim = 0)
sample_negatives[1] = torch.zeros([int(NTNetparams.eeg_sample_count / 2), 1], dtype=torch.float32)

samples = [None, None] # Combine the two
samples[0] = torch.cat([sample_positives[0], sample_negatives[0]], dim = 0)
samples[1] = torch.cat([sample_positives[1], sample_negatives[1]], dim = 0)

## Create test data that isn't trained on
test = [None, None]
test_positives = [None, None]
test_positives[0] = torch.rand(10, NTNetparams.eeg_sample_length) * 0.50 + 0.25 # Test 10 good samples
test_positives[1] = torch.ones([test_positives[0].shape[0], 1])

test_negatives = [None, None]
test_negatives_low = torch.rand(5, NTNetparams.eeg_sample_length) * 0.25 # Test 5 bad low samples
test_negatives_high = torch.rand(5, NTNetparams.eeg_sample_length) * 0.25 + 0.75 # Test 5 bad high samples
test_negatives[0] = torch.cat([test_negatives_low, test_negatives_high], dim = 0)
test_negatives[1] = torch.zeros([test_negatives[0].shape[0], 1])
test[0] = torch.vstack((test_negatives[0], test_positives[0]))
test[1] = torch.vstack((test_negatives[1], test_positives[1]))

# print("We have created a sample dataset with " + str(samples[0].shape[0]) + " samples")
# print("Half of those are positive samples with a score of 100%")
# print("Half of those are negative samples with a score of 0%")
# print("We have also created two sets of 10 test samples to check the validity of the network")

# rcParams['figure.figsize'] = 15, 5

# plt.title("Sample Data Set")
# plt.plot(list(range(0, NTNetparams.eeg_sample_length)), sample_positives[0][0], color = "#bbbbbb", label = "Samples")
# plt.plot(list(range(0, NTNetparams.eeg_sample_length)), sample_positives[0].mean(dim = 0), color = "g", label = "Mean Positive")
# plt.plot(list(range(0, NTNetparams.eeg_sample_length)), sample_negatives_high[0], color = "#bbbbbb")
# plt.plot(list(range(0, NTNetparams.eeg_sample_length)), sample_negatives_high.mean(dim = 0), color = "r", label = "Mean Negative")
# plt.plot(list(range(0, NTNetparams.eeg_sample_length)), sample_negatives_low[0], color = "#bbbbbb")
# plt.plot(list(range(0, NTNetparams.eeg_sample_length)), sample_negatives_low.mean(dim = 0), color = "r")
# plt.plot(list(range(0, NTNetparams.eeg_sample_length)), [0.75] * NTNetparams.eeg_sample_length, color = "k")
# plt.plot(list(range(0, NTNetparams.eeg_sample_length)), [0.25] * NTNetparams.eeg_sample_length, color = "k")
# plt.legend()
# plt.show()

def extractFakeData():
    return samples, test