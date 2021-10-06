import torch
import numpy as np

if torch.cuda.is_available():
    dev = torch.device("cuda")
    x = torch.ones(5, device=dev)
    print(x)
    y = torch.ones(5)
    print(y)
    y = y.to(dev)
    print(y)
    z = x+y
    print(z)
    z = z.to("cpu")
    z = z.numpy()
    print(z)

#%% Autograd
import torch
x = torch.randn(3, requires_grad=True)
y= x+2
z = y*y*2
z = z.mean()
z = z.backward()
print(y.retain_grad()) 
# %%
weight = torch.ones(4, requires_grad=True)
weight.grad.zero_()
 
optimizer = torch.optim.SGD(weight , lr=0.01)
optimizer.step()
optimizer.zero_grad( )

#%%
# first model
# loss and opptimizer
# Training loop
#   -forward pass
#   -backward pass
#   -update weights
import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

# prepare data
x_np, y_np = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=1)

x = torch.from_numpy(x_np.astype(np.float32))
y = torch.from_numpy(y_np.astype(np.float32))
y = y.view(y.shape[0], 1)

n_samples, n_features = x.shape

#model
model = nn.Linear(n_features, 1)

#loss
criteria = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

#training loop
for epoch in range(100):
    # forward
    y_predicted = model(x)
    loss = criteria(y_predicted, y)

    # backward
    loss.backward()

    #update
    optimizer.step()
    optimizer.zero_grad()

    if (epoch+1)% 10 ==0:
        print(f'epoch: {epoch+1}, loss = {loss.item():.4f}')

predict = model(x).detach().numpy()
plt.plot(x_np, y_np,'ro')
plt.plot(x_np,predict,'b')
plt.show()
# %%
class LogisticRegression(nn.Module):

    def __init__(self, n_imput_features):
        super(LogisticRegression,self).__init__()
        self.linear = nn.Linear(n_imput_features, 1)
    
    def forwardpass(self,x):
        y_predicted = torch.sigmoid(self.linear(x))
        return y_predicted

model = LogisticRegression(n_features)