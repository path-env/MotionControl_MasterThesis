import torch
from params import NeuroTechNetParams as NTNetparams

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
test_positives = torch.rand(10, NTNetparams.eeg_sample_length) * 0.50 + 0.25 # Test 10 good samples
test_negatives_low = torch.rand(5, NTNetparams.eeg_sample_length) * 0.25 # Test 5 bad low samples
test_negatives_high = torch.rand(5, NTNetparams.eeg_sample_length) * 0.25 + 0.75 # Test 5 bad high samples
test_negatives = torch.cat([test_negatives_low, test_negatives_high], dim = 0)

print("We have created a sample dataset with " + str(samples[0].shape[0]) + " samples")
print("Half of those are positive samples with a score of 100%")
print("Half of those are negative samples with a score of 0%")
print("We have also created two sets of 10 test samples to check the validity of the network")