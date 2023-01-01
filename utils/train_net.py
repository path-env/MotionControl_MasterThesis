import time
import numpy as np

import torch
import optuna

from torch.utils.tensorboard import SummaryWriter
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from main.extraction.data_extractor import DataContainer

from sklearn import metrics

#%% May have to change this!!!!
from models.DeepRNN import SEQnet
from models.neurotec_edu import Neurotech_net
from data.params import NeuroTechNetParams

torch.backends.cudnn.benchmark = True

def train_and_validate(data, model, loss_fn, optim, LRscheduler, tb, dCfg, nCfg, epochs=25, shuffle = True, opt_trial = False):
    device = 'cuda' if torch.cuda.is_available() else "cpu"
    dtype = torch.float32 if device == 'cuda' else torch.float32
    # device = 'cpu'
    model = model.to(device, dtype = dtype)
    
    train_loader, val_loader, test_loader = data.get_loaders()
    start = time.time()
    # tb_comment = f'batch_size={nCfg.train_bs}, lr={nCfg.lr}'
    # tb = SummaryWriter(tb_info[0], filename_suffix= tb_info[1])
    history = {'f1_train': [], 'acc_train':[], 'prec_train':[], 'recl_train':[], 'train_loss':[],
        'f1_val': [], 'acc_val':[], 'prec_val':[], 'recl_val':[], 'val_loss':[]}
    lr = []
    best_acc = 0.0
    print('###############Train and Validate######################')
    for epoch in range(epochs):
        train_data_size = data.train_idx.shape[0]
        validation_data_size = data.val_idx.shape[0]
        epoch_start = time.time()

        # Loss and Accuracy within the epoch
        train_loss, train_acc = 0.0, 0.0        
        valid_loss, valid_acc = 0.0, 0.0
        
        overall_label_train,overall_predic_train = [],[]
        overall_label_val,overall_predic_val = [],[]

        # Training Loop
        model.train()
        prob_train = []
        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device, dtype = dtype)
            labels = labels.to(device)
            optim.zero_grad(set_to_none= True)
            outputs = model(inputs)#.flatten()
            # print(outputs.flatten())
            loss = loss_fn(outputs, labels)
            loss.backward()
            optim.step()
            train_loss += loss.item() * inputs.size(0)
            
            # Compute the accuracy
            # _, predictions = torch.max(outputs.data, -1)
            # predictions = torch.round(outputs)
            # correct_counts = (predictions == labels)
            # acc = torch.mean(correct_counts.type(torch.FloatTensor))
            # train_acc += acc.item() * inputs.size(0)
            # train_data_size+= inputs.size(0)
            # print("Batch number: {:03d}, Training: Loss: {:.4f}, Accuracy: {:.4f}".format(i, loss.item(), acc.item()))
            
            overall_label_train.append(labels.tolist())
            # prob_train.append(outputs)
            if model.n_classes >1: # Multiclass classification
                overall_predic_train.append(torch.argmax(torch.softmax(outputs, dim =1),dim=1).tolist())
            else: # Binary classification
                overall_predic_train.append(torch.round(torch.sigmoid(outputs)).tolist())
        LRscheduler.step()    
        lr = (LRscheduler.get_last_lr()[0])

        # Validation - No gradient tracking needed
        prob_val = []
        with torch.inference_mode():
            model.eval()
            for j, (input, labels) in enumerate(val_loader):
                input = input.to(device, dtype = dtype)
                labels = labels.to(device)
                outputs = model(input)#.flatten()
                loss = loss_fn(outputs, labels)
                valid_loss += loss.item() * input.size(0)

                # Calculate validation accuracy
                # predictions = torch.round(outputs)
                # correct_counts = (predictions == labels)
                # acc = torch.mean(correct_counts.type(torch.FloatTensor))
                # valid_acc += acc.item() * input.size(0)
                # validation_data_size+= input.size(0)
                #print("Validation Batch number: {:03d}, Validation: Loss: {:.4f}, Accuracy: {:.4f}".format(j, loss.item(), acc.item()))

                overall_label_val.append(labels.tolist())
                # prob_val.append(outputs)
                if model.n_classes >1: # Multiclass classification
                    overall_predic_val.append(torch.argmax(torch.softmax(outputs, dim =1),dim=1).tolist())
                else: # Binary classification
                    overall_predic_val.append(torch.round(torch.sigmoid(outputs)).tolist())

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
        
        label, target_names = list(dCfg.event_dict.values()), list(dCfg.event_dict.keys())
        # dd = metrics.classification_report(overall_label_train, overall_predic_train, output_dict=True,
        #     labels= label, target_names= target_names)

        prec_train, recl_train, f1_train, _ = metrics.precision_recall_fscore_support(overall_label_train, 
                        overall_predic_train, zero_division=1, average='weighted', labels=label)
        acc_train  = metrics.accuracy_score(overall_label_train, overall_predic_train)
        # roc_train = metrics.roc_auc_score(overall_label_train, overall_predic_train, average='micro', 
        #     multi_class = 'ovo')
        # cf_train = metrics.multilabel_confusion_matrix(overall_label_train, overall_predic_train)

        tb.add_scalar('f1_train', f1_train, epoch)
        tb.add_scalar('acc_train', acc_train, epoch)
        tb.add_scalar('prec_train', prec_train, epoch)
        tb.add_scalar('recl_train', recl_train, epoch)
        # tb.add_scalar('ROC_train', roc, epoch)
        tb.add_scalar('train_loss', avg_train_loss, epoch)
        tb.add_scalar('lr', np.array(lr), epoch)

        history['f1_train'].append(f1_train)
        history['acc_train'].append(acc_train)
        history['prec_train'].append(prec_train)
        history['recl_train'].append(recl_train)
        history['train_loss'].append(train_loss)

        prec_val, recl_val, f1_val, _ = metrics.precision_recall_fscore_support(overall_label_val, 
                        overall_predic_val, zero_division=1, average='weighted', labels=label)
        acc_val  = metrics.accuracy_score(overall_label_val, overall_predic_val)
        # roc = metrics.roc_auc_score(overall_label_val, overall_predic_val, average='macro', 
        #     multi_class = 'ovr')
        # cf_val = metrics.multilabel_confusion_matrix(overall_label_val, overall_predic_val)
        # # df_cm = pd.DataFrame(cf_val/np.sum(cf_val) * 10, index=[i for i in classes],
        # #                  columns=[i for i in classes])
        # # fig =sn.heatmap(df_cm, annot=True).get_figure()

        tb.add_scalar('f1_val', f1_val, epoch)
        tb.add_scalar('acc_val', acc_val, epoch)
        tb.add_scalar('prec_val', prec_val, epoch)
        tb.add_scalar('recl_val', recl_val, epoch)        
        # tb.add_scalar('ROC_val', roc, epoch)
        tb.add_scalar('valid_loss', avg_valid_loss,epoch)
        # tb.add_figure("Confusion matrix", fig, epoch)

        # tb.add_scalar('valid_acc',avg_valid_acc,epoch)
        # tb.add_scalar('train_acc', avg_train_acc, epoch)
        
        # for name, weight in model.named_parameters():
        #     tb.add_histogram(name, weight, epoch)
        #     tb.add_histogram(f'{name}.grad', weight.grad, epoch)

        # history.append([avg_train_loss, avg_valid_loss, avg_train_acc, avg_valid_acc])
        history['f1_val'].append(f1_val)
        history['acc_val'].append(acc_val)
        history['prec_val'].append(prec_val)
        history['recl_val'].append(recl_val)
        history['val_loss'].append(valid_loss)

        epoch_end = time.time()
        print(f"Epoch : {epoch+1}|{epochs}, Training: Loss: {avg_train_loss:.3f}, Accuracy: {acc_train*100:.3f}%, \n\r\t\tValidation: Loss : {avg_valid_loss:.3f}, Accuracy: {acc_val*100:.3f}%, Time: {epoch_end-epoch_start:.4f}",end='\r')
        print(' ')
        if opt_trial != False:
            opt_trial.report(train_acc, epoch)

            if opt_trial.should_prune():
                raise optuna.exceptions.TrialPruned()

    # Test Loop
    if nCfg.test_split !=0.0:
        tb,_ = test_net(model, test_loader, tb, dCfg)

    # tb.add_pr_curve('PRcurve_train',overall_label_train,overall_predic_train)
    # tb.add_pr_curve('PRcurve_val',overall_label_val,overall_predic_val)
    # tb.add_figure("Confusion matrix", fig, epoch)
    
    tb.add_graph(model, inputs)
    # Save if the model has best accuracy till now
    #torch.save(model, dataset+'_model_'+str(epoch)+'.pt')
    tb.close()    
    return acc_train, acc_val, f1_train, f1_val, 0
    # return np.mean(history['acc_train']), np.mean(history['acc_val']), np.mean(history['f1_train']), np.mean(history['f1_val']) 

def test_net(model, test_loader, tb, dCfg):
    print('###############Testing######################')
    label_test, predic_test = [],[]
    device = 'cuda' if torch.cuda.is_available() else "cpu"
    dtype = torch.float32 if device == 'cuda' else torch.float32
    model = model.to(device, dtype = dtype)
    prob_test = []
    with torch.inference_mode():
        model.eval()
        for i, (inputs, labels) in enumerate(test_loader):
            inputs = inputs.to(device, dtype = dtype)
            outputs = model(inputs)#.half()
            # correct_counts = (outputs == labels.to(device, dtype = dtype))
            # acc = torch.mean(correct_counts.type(torch.FloatTensor))
            # test_acc += acc.item() * inputs.size(0)

            label_test.append(labels.tolist())
            if model.n_classes >1: # Multiclass classification
                predic_test.append(torch.argmax(torch.softmax(outputs, dim =1),dim=1).tolist())
                prob_test.append(torch.softmax(outputs, dim =1).tolist())
            else: # Binary classification
                predic_test.append(torch.round(torch.sigmoid(outputs)).tolist())
                prob_test.append(torch.sigmoid(outputs).tolist())

    overall_label_test = np.array(label_test).flatten()
    overall_predic_test  = np.array(predic_test).flatten()
    prob_test = np.array(prob_test).reshape(len(label_test), model.n_classes)

    prec_test, recl_test, f1_test, _ = metrics.precision_recall_fscore_support(overall_label_test, 
                    overall_predic_test, zero_division=1, average='weighted', labels= list(dCfg.event_dict.values()))

    acc_test  = metrics.accuracy_score(overall_label_test, overall_predic_test)
    roc_test = metrics.roc_auc_score(overall_label_test, prob_test, average='macro', 
        multi_class = 'ovr')
    cf_val = metrics.multilabel_confusion_matrix(overall_label_test, overall_predic_test)

    tb.add_scalar('f1_test',f1_test,i)
    tb.add_scalar('acc_test',acc_test,i)
    tb.add_scalar('prec_test',prec_test,i)
    tb.add_scalar('recl_test',recl_test,i)
    # tb.add_scalar('roc_test',roc_test,i)
    return tb, acc_test, f1_test, roc_test, cf_val

#%%
if __name__ == "__main__":
    data = DataContainer(0,0, check_net= True)
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
