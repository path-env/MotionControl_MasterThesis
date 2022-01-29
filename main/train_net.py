# from torch import nn, optim
import torch
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import StratifiedKFold

from data_extractor import data_container

#%% May have to change this!!!!
from models.DeepRNN import DRNN
from models.neurotec_edu import Neurotech_net
from data.params import NeuroTechNetParams
model = Neurotech_net()
params = NeuroTechNetParams()
# Loss Function
criteria = torch.nn.MSELoss()
# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=params.learning_rate)
# Tensorboard writer
writer = SummaryWriter('runs/fake_data')

#%% Model Training
# create batches
dataset = data_container(check_net = True)
writer.add_graph(model, dataset.train_x)
for n_iter,epoch in enumerate(range(1000)):
    # for x,y in zip(dataset.train_x, dataset.train_y):
    optimizer.zero_grad()
    yhat = model(dataset.train_x)
    loss = criteria(yhat, dataset.train_y)
    loss.backward()
    optimizer.step()
    writer.add_scalar('Loss/train', loss, n_iter)
    # writer.add_scalar('Loss/test', np.random.random(), n_iter)
    # writer.add_scalar('Accuracy/train', np.random.random(), n_iter)
    # writer.add_scalar('Accuracy/test', np.random.random(), n_iter)
# writer.add_scalar()
writer.close()

# torch.save(model, "/media/mangaldeep/HDD2/workspace/MotionControl_MasterThesis/models/model_check")
#%% Model Testing
# import torch
# model = torch.load("/media/mangaldeep/HDD2/workspace/MotionControl_MasterThesis/models/model_check")
test_results = model(dataset.test_x).data.tolist()
