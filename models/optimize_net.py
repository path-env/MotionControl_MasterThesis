import sys


sys.path.append('/media/mangaldeep/HDD2/workspace/MotionControl_MasterThesis')

import optuna
import torch
import numpy as np
from torch import optim

from models.EEGNet_2018 import EEGnet
import torch.optim.lr_scheduler as lr_scheduler
from data.params import BCI3Params, EEGNetParams
from main.extraction.data_extractor import data_container 
from models.train_net import train_and_validate

device = 'cuda' if torch.cuda.is_available() else "cpu"
dtype = torch.float32 if device == 'cuda' else torch.float32

dCfg = BCI3Params()
nCfg = EEGNetParams()

filename = 'Train_locl_ssp_car_ica_TF_EEGnet_CNN_[3]_1.npz'
datafile = np.load(f'/media/mangaldeep/HDD2/workspace/MotionControl_MasterThesis/main/feature_extraction/{filename}')
train_data = datafile['arr_0']
if len(train_data.shape) <4:
    train_data = np.expand_dims(train_data, axis=1)
labels = datafile['arr_1']
# print(labels)
n_classes = np.unique(labels).shape[0]
data = data_container(train_data, labels, nCfg)

_,_,n_chan,_ = data.x.shape
model =  EEGnet(n_classes,n_chan,dCfg.sfreq)

def objective(trial):
    param = {
              'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-1),
              'optimizer': trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD","Adamax"])
            #   'n_unit': trial.suggest_int("n_unit", 4, 18)
              }    
    lr = param['learning_rate']
    optimizer = getattr(optim, param['optimizer'])(model.parameters(), lr= lr)
    LRscheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    loss = torch.nn.BCELoss()

    # Generate the model.
    # model = model(trial).to(device, dtype)
    tb_comment = f'bs:{nCfg.train_bs}|lr:{lr}|optim: {optimizer}|'
    tb = (f'log/{model._get_name()}/{dCfg.name}/optuna', f'{lr}')
    _, train_accuracy = train_and_validate(data, model, loss, optimizer,LRscheduler, tb, dCfg, nCfg, epochs=20, opt_trial=trial)

    return train_accuracy




if __name__ == "__main__":
    EPOCHS = 30
        
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(),
            pruner=optuna.pruners.SuccessiveHalvingPruner(), study_name='EEGnet_trials')
    study.optimize(objective, n_trials=10)

    best_trial = study.best_trial

    for key, value in best_trial.params.items():
        print("{}: {}".format(key, value))
    
    optuna.visualization.plot_contour(study)

    optuna.visualization.plot_param_importances(study)



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