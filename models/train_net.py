import sys
sys.path.append('/media/mangaldeep/HDD2/workspace/MotionControl_MasterThesis')
# from torch import nn, optim
import time
import torch
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import StratifiedKFold

from extraction.data_extractor import data_container

#%% May have to change this!!!!
from models.DeepRNN import DRNN
from models.neurotec_edu import Neurotech_net
from data.params import NeuroTechNetParams

def train_and_validate(model,train_loader, validation_loader, loss_criterion, optimizer, LRscheduler, epochs=25):
    '''
    Function to train and validate
    Parameters
        :param model: Model to train and validate
        :param loss_criterion: Loss Criterion to minimize
        :param optimizer: Optimizer for computing gradients
        :param epochs: Number of epochs (default=25)
  
    Returns
        model: Trained Model with best validation accuracy
        history: (dict object): Having training loss, accuracy and validation loss, accuracy
    '''
    
    start = time.time()
    writer = SummaryWriter('runs/fake_data')
    history = []

    best_acc = 0.0
    # device = 'cuda' if torch.cuda.is_available() else "cpu"
    device = 'cpu'
    # model = model.cuda() if torch.cuda.is_available() else model

    # writer.add_graph(model, train_loader.x)
    for epoch in range(epochs):
        train_data_size = 0 #train_loader.dataset.x.shape[0]
        validation_data_size = 0 #validation_loader.dataset.x.shape[0]
        epoch_start = time.time()
        print("Epoch: {}/{}".format(epoch+1, epochs))
        
        # Set to training mode
        model.train()
        
        # Loss and Accuracy within the epoch
        train_loss = 0.0
        train_acc = 0.0
        
        valid_loss = 0.0
        valid_acc = 0.0
        
        for i, (inputs, labels) in enumerate(train_loader):

            inputs = inputs.to(device)
            labels = labels.to(device).float()
            
            # Clean existing gradients
            optimizer.zero_grad()
            
            # Forward pass - compute outputs on input data using the model
            outputs = model(inputs).squeeze(0)
            labels = labels.view_as(outputs)
            
            # Compute loss
            loss = loss_criterion(outputs, labels)
            
            # Backpropagate the gradients
            loss.backward()
            
            # Update the parameters
            optimizer.step()
            
            # Compute the total loss for the batch and add it to train_loss
            train_loss += loss.item() * inputs.size(0)
            
            # Compute the accuracy
            ret, predictions = torch.max(outputs.data, -1)
            correct_counts = predictions.eq(labels.data.view_as(predictions))
            
            # Convert correct_counts to float and then compute the mean
            acc = torch.mean(correct_counts.type(torch.FloatTensor))
            
            # Compute total accuracy in the whole batch and add to train_acc
            train_acc += acc.item() * inputs.size(0)
            
            train_data_size+= train_loader.batch_size
            # print("Batch number: {:03d}, Training: Loss: {:.4f}, Accuracy: {:.4f}".format(i, loss.item(), acc.item()))

        LRscheduler.step()    

        # Validation - No gradient tracking needed
        with torch.no_grad():

            # Set to evaluation mode
            model.eval()

            # Validation loop
            for j, (inputs, labels) in enumerate(validation_loader):
                inputs = inputs.to(device)
                labels = labels.to(device).float()

                # Forward pass - compute outputs on input data using the model
                outputs = model(inputs).squeeze(0)
                labels = labels.view_as(outputs)

                # Compute loss
                loss = loss_criterion(outputs, labels)

                # Compute the total loss for the batch and add it to valid_loss
                valid_loss += loss.item() * inputs.size(0)

                # Calculate validation accuracy
                ret, predictions = torch.max(outputs.data, -1)
                correct_counts = predictions.eq(labels.data.view_as(predictions))

                # Convert correct_counts to float and then compute the mean
                acc = torch.mean(correct_counts.type(torch.FloatTensor))

                # Compute total accuracy in the whole batch and add to valid_acc
                valid_acc += acc.item() * inputs.size(0)

                validation_data_size+= validation_loader.batch_size
                #print("Validation Batch number: {:03d}, Validation: Loss: {:.4f}, Accuracy: {:.4f}".format(j, loss.item(), acc.item()))
            
        # Find average training loss and training accuracy
        avg_train_loss = train_loss/train_data_size 
        avg_train_acc = train_acc/train_data_size

        # Find average training loss and training accuracy
        avg_valid_loss = valid_loss/validation_data_size 
        avg_valid_acc = valid_acc/validation_data_size

        history.append([avg_train_loss, avg_valid_loss, avg_train_acc, avg_valid_acc])
                
        epoch_end = time.time()
        
        # print(train_data_size)
        # print(validation_data_size)
        print("Epoch : {:03d}, Training: Loss: {:.4f}, Accuracy: {:.4f}%, \n\t\tValidation : Loss : {:.4f}, Accuracy: {:.4f}%, Time: {:.4f}s".format(epoch+1, avg_train_loss, avg_train_acc*100, avg_valid_loss, avg_valid_acc*100, epoch_end-epoch_start))
        
        # Save if the model has best accuracy till now
        #torch.save(model, dataset+'_model_'+str(epoch)+'.pt')
    writer.close()       
    return model, history

#%%
if __name__ == "__main__":
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