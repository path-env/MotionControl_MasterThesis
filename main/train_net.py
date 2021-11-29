import torch
from torch import nn, optim
import torch.nn.functional as F

from sklearn.model_selection import StratifiedKFold

from main.data_extractor import data_container
from networks.DeepRNN import DRNN

dataset = data_container()
model = DRNN()
# model = nn.Linear(64,64)
criteria = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# skf = StratifiedKFold(n_splits = 3)
# full_data = full_data[1:,:]
# for train_index, test_index in skf.split(full_data, gt):
#     X_train, X_test = full_data[train_index], full_data[test_index]
#     y_train, y_test = gt[train_index], gt[test_index]

# create batches

for epoch in range(100):
    for x,y in zip(dataset.x, dataset.y):
        optimizer.zero_grad()
        yhat = model(x)
        loss = criteria(yhat, y)
        loss.backward()
        optimizer.step()