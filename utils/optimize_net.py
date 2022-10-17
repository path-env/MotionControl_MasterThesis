import optuna
import torch
import numpy as np
from torch import optim

from models.EEGNet_2018 import EEGnet
from models.DLSTM_MI_2019 import ATTNnet

import torch.optim.lr_scheduler as lr_scheduler
from data.params import BCI3Params, EEGNetParams, OCIParams, PhysionetParams
from main.extraction.data_extractor import DataContainer 
from utils.train_net import train_and_validate

from torch.utils.tensorboard import SummaryWriter

device = 'cuda' if torch.cuda.is_available() else "cpu"
dtype = torch.float32 if device == 'cuda' else torch.float32

dCfg = OCIParams()
nCfg = EEGNetParams()

filename = 'Physionet__RAW_[3, 4, 7, 8, 11, 12]_1.npz'

filename = 'Physionet_16locl_STAT_[3, 4, 7, 8, 11, 12]_1.npz'

filename = 'OCIParams_locl_ssp_car_ica_RAW_[3, 4, 7, 8, 11, 12]_1.npz'
train = np.load(f'/media/mangaldeep/HDD2/workspace/MotionControl_MasterThesis/data/train/{filename}')
train_x = train['arr_0']
train_y = train['arr_1']

n_classes = len(set(train_y))
bs, lr = 1, 0.001
data = DataContainer(train_x, train_y, nCfg)
n_s,cCh,n_chan,n_T = data.x.shape
# model =  EEGnet(n_classes,n_chan,dCfg.sfreq)

EPOCHS = 50
best_f1 = 0.0   

def objective(trial):
    global best_f1
    param = {
              'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-1, log = True),
              'optimizer': trial.suggest_categorical("optimizer", ["Adam", "SGD","Adamax", "Adadelta",
                                                                    "ASGD", "RAdam"]),
              'F1': trial.suggest_int("F1", 10,30, step =2),
              'F2': trial.suggest_int("F2", 10,30, step =2),
              'D': trial.suggest_int("D", 1,5, step =1),
              'dt': trial.suggest_float("dt", 0.1, 0.7),
              }    
    lr = param['learning_rate']
    # Generate the model.
    # F1,F2, D,dt =  param['F1'], param['F2'], param['D'], param['dt']
    # model = ATTNnet(n_classes, cCh*n_T, n_layers = param['D'], dt = param['dt'])

    model =  EEGnet(n_classes = n_classes, n_chan = n_chan, n_T = n_T,
             sf = dCfg.sfreq, F1 = param['F1'], F2 = param['F2'], D =param['D'], dt = param['dt'])
    optimizer = getattr(optim, param['optimizer'])(model.parameters(), lr= lr)
    # optimizer = optim.Adam(model.parameters(), lr= lr)
    LRscheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    if n_classes == 2:
        loss = torch.nn.BCEWithLogitsLoss() # No sigmoid required at the last layer & in train loop, Binary classification
    else:
        loss = torch.nn.CrossEntropyLoss() # No softmax required at the last layer, Multiclass classification


    tb_comment = f'bs:{nCfg.train_bs}|lr:{lr}|optim: {optimizer}|'
    tb_info = (f'logs/{model._get_name()}/{dCfg.name}/optuna/{param["optimizer"]}', f'{lr}')
    tb = SummaryWriter(tb_info[0])#, filename_suffix= tb_comment)
    acc_train, acc_val, f1_train, f1_val = train_and_validate(data, model, loss, optimizer,LRscheduler, tb, dCfg, nCfg, epochs=EPOCHS, opt_trial=trial)
    if f1_val > best_f1:
        best_f1 = f1_val
        print('Saving the model.....')
        model_scripted = torch.jit.script(model)
        model_scripted.save(f"/media/mangaldeep/HDD2/workspace/MotionControl_MasterThesis/models/{model._get_name()+'_optim.pt'}")
    return f1_val


study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(),study_name='EEGnet_OCIData')
        # pruner=optuna.pruners.MedianPruner(), study_name='EEGnet_trials')
study.optimize(objective, n_trials=100)

best_trial = study.best_trial
for key, value in best_trial.params.items():
    print("{}: {}".format(key, value))

fig = optuna.visualization.plot_contour(study)
fig.show()

fig = optuna.visualization.plot_param_importances(study)
fig.show()


#     pruned_trials = [t for t in study.trials if t.state == optuna.structs.TrialState.PRUNED]
#     complete_trials = [t for t in study.trials if t.state == optuna.structs.TrialState.COMPLETE]

#     print("Study statistics: ")
#     print("  Number of finished trials: ", len(study.trials))
#     print("  Number of pruned trials: ", len(pruned_trials))
#     print("  Number of complete trials: ", len(complete_trials))

#     print("Best trial:")
#     trial = study.best_trial

#     print("  Value: ", trial.value)

#     print("  Params: ")
#     for key, value in trial.params.items():
#         print("    {}: {}".format(key, value))

#                 trial.report(accuracy, epoch)

#         # Handle pruning based on the intermediate value.
#         if trial.should_prune():
#             raise optuna.exceptions.TrialPruned()


# best_trial = study.best_trial

# for key, value in best_trial.params.items():
#     print("{}: {}".format(key, value))