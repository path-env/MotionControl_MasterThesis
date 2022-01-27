from torch import nn, optim

from sklearn.model_selection import StratifiedKFold

from data_extractor import data_container

#%% May have to change this!!!!
from models.DeepRNN import DRNN
from models.neurotec_edu import Neurotech_net
model = Neurotech_net()
# Loss Function
criteria = nn.MSELoss()
# Optimizer
optimizer = optim.SGD(model.parameters(), lr=0.1)
#%%
# skf = StratifiedKFold(n_splits = 3)
# full_data = full_data[1:,:]
# for train_index, test_index in skf.split(full_data, gt):
#     X_train, X_test = full_data[train_index], full_data[test_index]
#     y_train, y_test = gt[train_index], gt[test_index]

# create batches
dataset = data_container()
for epoch in range(100):
    for x,y in zip(dataset.x, dataset.y):
        optimizer.zero_grad()
        yhat = model(x)
        loss = criteria(yhat, y)
        loss.backward()
        optimizer.step()