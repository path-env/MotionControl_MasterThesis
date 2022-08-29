# load_ext tensorboard
import torch, torchvision
from torchvision import transforms, models
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchsummary import summary
import tensorboard


def TF_net(classes):
    alexnet = models.alexnet(weights='DEFAULT')
    # model = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained=True)
    # alexnet.eval();
    alexnet

    # # Freesze model parameters
    # for param in alexnet.parameters():
    #     param.requires_grad = False

    # change final layer of alexnet model
    alexnet.classifier[6] = nn.Linear(4096,1)
    alexnet.classifier.add_module('7',nn.Sigmoid())
    return alexnet

if __name__ =='__main__':
    model =  TF_net(['right','left'])
    print(model)