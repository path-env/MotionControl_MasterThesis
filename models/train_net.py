import sys
sys.path.append('/media/mangaldeep/HDD2/workspace/MotionControl_MasterThesis')
# from torch import nn, optim
import time
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
# from sklearn.model_selection import StratifiedKFold
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from main.extraction.data_extractor import data_container

from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score

#%% May have to change this!!!!
from models.DeepRNN import DRNN
from models.neurotec_edu import Neurotech_net
from data.params import NeuroTechNetParams

torch.backends.cudnn.benchmark = True

def train_and_validate(data, model, loss_fn, optim, LRscheduler, tb_info, dCfg, nCfg, epochs=25, shuffle = True):
    ds_size = len(data)
    idx = list(range(ds_size))
    if shuffle :
        # np.random.seed(random_seed)
        np.random.shuffle(idx)

    val_split = int(np.floor(nCfg.val_split * ds_size))
    test_split = int(np.floor(nCfg.test_split  * ds_size))
    val_idx, test_idx,train_idx = idx[:val_split], idx[val_split:val_split+test_split], idx[val_split+test_split:]

    # Train and validate shuffle split
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(val_idx)
    test_sampler = SubsetRandomSampler(test_idx)

    train_loader = DataLoader(data, batch_size= nCfg.train_bs, sampler=train_sampler, pin_memory=True, num_workers= nCfg.num_wrkrs)
    validation_loader = DataLoader(data, batch_size= nCfg.val_bs, sampler=valid_sampler, pin_memory=True, num_workers= nCfg.num_wrkrs)
    test_loader = DataLoader(data, batch_size=1, sampler=test_sampler, pin_memory=True, num_workers= nCfg.num_wrkrs)


    start = time.time()
    tb_comment = f'batch_size={nCfg.train_bs}, lr={nCfg.lr}'
    tb = SummaryWriter(tb_info[0], comment= tb_comment)
    history = []

    best_acc = 0.0
    device = 'cuda' if torch.cuda.is_available() else "cpu"
    # device = 'cpu'
    
    model = model.to(device, dtype = torch.float16)
    
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

        overall_label_train,overall_predic_train = [],[]
        overall_label_val,overall_predic_val = [],[]
        f1, prec, recal, prec_recal= 0,0,0,0
        # labels, preds = [],[]
        
        for i, (inputs, labels) in enumerate(train_loader):

            inputs = inputs
            labels = labels

            # Clean existing gradients
            optim.zero_grad(set_to_none= True)
            
            # Forward pass - compute outputs on input data using the model
            outputs = model(inputs).flatten().half()
            # labels = labels.view_as(outputs)
            
            # print(f'Labels:{labels}, Outputs:{outputs}')
            # Compute loss
            loss = loss_fn(outputs, labels)
            
            # Backpropagate the gradients
            loss.backward()
            
            # Update the parameters
            optim.step()
            
            # Compute the total loss for the batch and add it to train_loss
            train_loss += loss.item() * inputs.size(0)
            
            # Compute the accuracy
            # _, predictions = torch.max(outputs.data, -1)
            predictions = torch.round(outputs)
            correct_counts = (predictions == labels)
            
            # Convert correct_counts to float and then compute the mean
            acc = torch.mean(correct_counts.type(torch.FloatTensor))
            
            # Compute total accuracy in the whole batch and add to train_acc
            train_acc += acc.item() * inputs.size(0)
            
            train_data_size+= inputs.size(0)
            # print("Batch number: {:03d}, Training: Loss: {:.4f}, Accuracy: {:.4f}".format(i, loss.item(), acc.item()))
            
            overall_label_train.append(labels.tolist())
            overall_predic_train.append(predictions.tolist())
            # f1 += f1_score(labels, outputs)
        # tb.add_pr_curve()
        LRscheduler.step()    

        # Validation - No gradient tracking needed
        with torch.inference_mode():

            # Set to evaluation mode
            model.eval()

            # Validation loop
            for j, (inputs, labels) in enumerate(validation_loader):
                inputs = inputs
                labels = labels

                # Forward pass - compute outputs on input data using the model
                outputs = model(inputs).flatten().half()
                # labels = labels.view_as(outputs)

                # Compute loss
                loss = loss_fn(outputs, labels)

                # Compute the total loss for the batch and add it to valid_loss
                valid_loss += loss.item() * inputs.size(0)

                # Calculate validation accuracy
                predictions = torch.round(outputs)
                correct_counts = (predictions == labels)

                # Convert correct_counts to float and then compute the mean
                acc = torch.mean(correct_counts.type(torch.FloatTensor))

                # Compute total accuracy in the whole batch and add to valid_acc
                valid_acc += acc.item() * inputs.size(0)

                validation_data_size+= inputs.size(0)
                #print("Validation Batch number: {:03d}, Validation: Loss: {:.4f}, Accuracy: {:.4f}".format(j, loss.item(), acc.item()))

                overall_label_val.append(labels.tolist())
                overall_predic_val.append(predictions.tolist())

        #Metrics
        # Find average training loss and training accuracy
        avg_train_loss = train_loss/train_data_size 
        avg_train_acc = train_acc/train_data_size

        # Find average training loss and training accuracy
        avg_valid_loss = valid_loss/validation_data_size 
        avg_valid_acc = valid_acc/validation_data_size
        
        overall_label_train = np.concatenate(overall_label_train)
        overall_predic_train = np.concatenate(overall_predic_train)
        overall_label_val = np.concatenate(overall_label_val)
        overall_predic_val = np.concatenate(overall_predic_val)

        f1_train = f1_score(overall_label_train, overall_predic_train)
        acc_train = accuracy_score(overall_label_train, overall_predic_train)
        prec_train = precision_score(overall_label_train, overall_predic_train)
        recl_train = recall_score(overall_label_train, overall_predic_train)
        roc_train = roc_auc_score(overall_label_train, overall_predic_train)

        f1_val = f1_score(overall_label_val, overall_predic_val)
        acc_val  = accuracy_score(overall_label_val, overall_predic_val)
        prec_val = precision_score(overall_label_val, overall_predic_val)
        recl_val = recall_score(overall_label_val, overall_predic_val)
        roc_val = roc_auc_score(overall_label_val, overall_predic_val)

        tb.add_scalar('f1_train', f1_train, epoch)
        tb.add_scalar('acc_train', acc_train, epoch)
        tb.add_scalar('prec_train', prec_train, epoch)
        tb.add_scalar('recl_train', recl_train, epoch)
        tb.add_scalar('ROC_train', roc_train, epoch)
        tb.add_scalar('train_loss', avg_train_loss, epoch)

        tb.add_scalar('f1_val', f1_val, epoch)
        tb.add_scalar('acc_val', acc_val, epoch)
        tb.add_scalar('prec_val', prec_val, epoch)
        tb.add_scalar('recl_val', recl_val, epoch)        
        tb.add_scalar('ROC_val', roc_val, epoch)
        tb.add_scalar('valid_loss', avg_valid_loss,epoch)
        # tb.add_scalar('valid_acc',avg_valid_acc,epoch)
        # tb.add_scalar('train_acc', avg_train_acc, epoch)
        
        for name, weight in model.named_parameters():
            tb.add_histogram(name, weight, epoch)
            tb.add_histogram(f'{name}.grad', weight.grad, epoch)

        history.append([avg_train_loss, avg_valid_loss, avg_train_acc, avg_valid_acc])
                
        epoch_end = time.time()
        
        print("Epoch : {:03d}, Training: Loss: {:.4f}, Accuracy: {:.4f}%, \n\t\tValidation: Loss : {:.4f}, Accuracy: {:.4f}%, Time: {:.4f}s".format(epoch+1, avg_train_loss, avg_train_acc*100, avg_valid_loss, avg_valid_acc*100, epoch_end-epoch_start))

    if nCfg.test_split ==0.0:
        label_test, predic_test = [],[]
        test_acc = 0
        model.eval()
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(test_loader):
                inputs = inputs
                outputs = model(inputs).half()
                outputs = torch.round(outputs).flatten()
                correct_counts = (outputs == labels)
                acc = torch.mean(correct_counts.type(torch.FloatTensor))
                test_acc += acc.item() * inputs.size(0)

                label_test.append(labels.tolist())
                predic_test.append(outputs.tolist())

            overall_label_test = np.array(label_test).flatten()
            overall_predic_test  = np.array(predic_test).flatten()
            f1_test = f1_score(overall_label_test, overall_predic_test)
            acc_test= accuracy_score(overall_label_test, overall_predic_test)
            prec_test = precision_score(overall_label_test, overall_predic_test)
            recl_test = recall_score(overall_label_test, overall_predic_test)
            roc_test = roc_auc_score(overall_label_test, overall_predic_test)

            tb.add_scalar('f1_test',f1_test,i)
            tb.add_scalar('acc_test',acc_test,i)
            tb.add_scalar('prec_test',prec_test,i)
            tb.add_scalar('recl_test',recl_test,i)
            tb.add_scalar('roc_test',roc_test,i)
    # tb.add_pr_curve('PRcurve_train',overall_label_train,overall_predic_train)
    # tb.add_pr_curve('PRcurve_val',overall_label_val,overall_predic_val)
    tb.add_graph(model, inputs)
    # Save if the model has best accuracy till now
    #torch.save(model, dataset+'_model_'+str(epoch)+'.pt')
    tb.close()       
    return model, history

#%%
if __name__ == "__main__":
    data = data_container(0,0, check_net= True)
    train_loader = torch.utils.data.DataLoader(data, batch_size=3)
    validation_loader = torch.utils.data.DataLoader(data, batch_size=3)

    model = Neurotech_net()

    params = NeuroTechNetParams()
    # Loss Function
    loss = torch.nn.MSELoss()
    # optim
    optim = torch.optim.Adam(model.parameters(), lr=params.learning_rate)

    LRscheduler = lr_scheduler.StepLR(optim, step_size=7, gamma=0.1)

    tb_info = ('trial','comment')
    train_and_validate(model, train_loader, validation_loader, loss,optim, LRscheduler, tb_info,epochs = 25)
    # Tensorboard writer
    # writer = SummaryWriter('runs/fake_data')

    # #%% Model Training
    # # create batches
    # dataset = data_container(0,0,check_net = True)
    # writer.add_graph(model, dataset.x)
    # for n_iter,epoch in enumerate(range(1000)):
    #     # for x,y in zip(dataset.x, dataset.y):
    #     optim.zero_grad(set_to_none= True)
    #     yhat = model(dataset.x)
    #     loss = loss(yhat, dataset.y)
    #     loss.backward()
    #     optim.step()
    #     writer.add_scalar('Loss/train', loss, n_iter)
    #     # writer.add_scalar('Loss/test', np.random.random(), n_iter)
    #     # writer.add_scalar('Accuracy/train', np.random.random(), n_iter)
    #     # writer.add_scalar('Accuracy/test', np.random.random(), n_iter)
    # # writer.add_scalar()
    # writer.close()

    # # torch.save(model, "/media/mangaldeep/HDD2/workspace/MotionControl_MasterThesis/models/model_check")
    # #%% Model Testing
    # # import torch
    # # model = torch.load("/media/mangaldeep/HDD2/workspace/MotionControl_MasterThesis/models/model_check")
    # test_results = model(dataset.test_x).data.tolist()