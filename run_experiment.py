import mne
# import subprocess.Popen as Popen
from data.testing_and_making_data import SimEnv
from main.extraction.oci_MI import extractOCI
from main.bci_pipeline import BrainSignalAnalysis
from data.params import OCIParams
from utils import run
dCfg = OCIParams()

# 1, Get data
Expr_name = 'P2_Day11_125'
sim = SimEnv(Expr_name)
sim.train_getdata()
# 2, COnvert to raw

raw = extractOCI(Expr_name=Expr_name)

# 3, Run BCI Pipeline to extract features
methods = 'locl_car_RAW_LDA_ML'
bsa = BrainSignalAnalysis(raw)
bsa.artifact_removal(methods, save_epochs = False)
bsa.feature_extraction(methods, save_feat = True)
# eeg_TS = self.dataFormat(bsa.features[0])

extractf = dCfg.name+'_'+'_'.join(methods.split('_')[:3])+'.npz'
run.run(filename=extractf)

# 4, Run OpenBCI GUI


# 5, runcarla

# Popen(['./Carla.sh'], cwd = '/opt/carala-simulator')

# 6, 

# roslaunch carla_bci_control carla_ros_bridge_with_bci_car_vehicle.launch 