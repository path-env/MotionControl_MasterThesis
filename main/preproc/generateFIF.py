import sys
sys.path.append('/media/mangaldeep/HDD2/workspace/MotionControl_MasterThesis')

from main.extraction.physionet_MI import extractPhysionet, extractBCI3
from main.preproc.preproc import Preproc
from data.params import PhysionetParams, globalTrial

data_cfg = PhysionetParams()
analy_cfg = globalTrial()
# %%
#1, Data Extractor
runs = [3]
person_id = 3
raw = extractBCI3(runs , person_id)

# %%
#2, Generate FiFs
procs = ['raw', 'ica', 'ssp', 'car', 'car_ica', 'ssp_car', 'ssp_car_ica', 'ssp_ica']
preproc = Preproc(raw, data_cfg, analy_cfg)
for method in procs:
    print(f'###### {method} is processing ########')
    print(' ')
    preproc.run(method, data_cfg, str(runs), str(person_id))
    print(' ')
    print(f'###### {method} is completed ########')
    print(' ')
    print(' ')