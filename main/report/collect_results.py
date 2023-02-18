import time
import numpy as np

from data.params import OCIParams, BCI3Params, PhysionetParams
from data.params import EEGNetParams, ATTNnetParams, CasCnnRnnnetParams

from utils import run
from main.extraction.physionet_MI import extractPhysionet
from main.extraction.bci3_IVa import extractBCI3
from main.extraction.oci_MI import extractOCI
from main.bci_pipeline import BrainSignalAnalysis

from models.EEGNet_2018 import EEGnet
from models.DLSTM_MI_2019 import ATTNnet
from models.DL_LSTM_MI_2021 import CasCnnRnnnet


def ML_methods():
    try:
        Results = []
        # take a dataset 
        oci_cfg = OCIParams()
        bbci_cfg = BCI3Params()
        phy_cfg = PhysionetParams()
        overall = []
        ANS = {}
        dataset= [bbci_cfg,oci_cfg, phy_cfg]

        ANS['fit_time'] = []
        ANS['score_time'] = []
        ANS['test_f1'] = []
        ANS['test_acc'] = []
        ANS['test_roc'] = []
        ANS['cv_time'] = []
        ANS['preproc'] = []
        ANS['features'] = []
        ANS['clasfier'] = []
        ANS['Feat+Clf'] = []
        ANS['dataset'] = []

        # take an algo
        for ds in dataset:
            # algo = [EEGNetParams.name, None]
            
            if ds.name =='Physionet':
                runs = [5, 6, 9, 10, 13, 14] #[3, 4, 7, 8, 11, 12]
                person_id = 1
                raw = extractPhysionet(runs, person_id)
            elif ds.name == 'BCI3IVa':
                raw = extractBCI3( )
            elif ds.name == 'OCIParams':
                raw = extractOCI(Expr_name = 'P2_Day11_125')
            
            bsa = BrainSignalAnalysis(raw, data_cfg = ds)
            # take a preprocessing step
            procs = ['ica', 'ssp', 'car', 'car_ica', 'ssp_car', 'ssp_car_ica', 'ssp_ica']
            for proc in procs:

                # take a featuure extraction
                feats= ['_CSP','_WST']#,'_STAT','_RAW','_TF']
                for feat in feats:
                    classes = ['_SVC_ML', '_LDA_ML', '_DTC_ML']
                    for cl in classes:
                        methods = ds.name+'_locl_norm_'+ proc+ feat + cl
                        print(methods)
                        

                        ans = {'fit_time':[], 'score_time':[], 'test_f1':[], 
                        'test_acc':[], 'test_roc':[], 'cv_time':[]}
                        # repeat every experiment 10 times to get the variation
                        cv = 2
                        for i in range(cv):
                            start = time.time()
                            bsa = BrainSignalAnalysis(raw, data_cfg = ds)
                            bsa.artifact_removal(methods, save_epochs = False)
                            bsa.feature_extraction(methods, save_feat = False)
                            scores = bsa.classifier(methods, save_model = False)
                            scores['cv_time'] = time.time() - start

                            ans['fit_time'] = np.hstack(( ans['fit_time'] ,scores['fit_time']))
                            ans['score_time'] = np.hstack(( ans['score_time'] ,scores['score_time']))
                            ans['test_f1'] = np.hstack(( ans['test_f1'] ,scores['test_f1']))
                            ans['test_acc'] = np.hstack(( ans['test_acc'] ,scores['test_acc']))
                            ans['test_roc'] = np.hstack(( ans['test_roc'] ,scores['test_roc']))
                            ans['cv_time'] = np.hstack(( ans['cv_time'] ,scores['cv_time']))

                        ANS['fit_time'].extend(ans['fit_time'].tolist())

                        ANS['score_time'].extend(ans['score_time'].tolist())                    

                        ANS['test_f1'].extend(ans['test_f1'].tolist())

                        ANS['test_acc'].extend(ans['test_acc'].tolist())    

                        ANS['test_roc'].extend(ans['test_roc'].tolist())    

                        ANS['cv_time'].extend(ans['cv_time'].tolist())

                        ANS['preproc'].extend([proc]*cv*10)
                        ANS['features'].extend([feat[1:]]*cv*10)
                        ANS['clasfier'].extend([cl[1:]]*cv*10)
                        ANS['Feat+Clf'].extend([feat[1:] +'_'+cl[1:]]*cv*10)
                        ANS['dataset'].extend([ds.name]*cv*10)

                        # overall.extend(ANS)
                            # ans[methods+str(i)] = scores
    finally:
        np.savez('ML_results', [ANS])


def DL_methods():
    try:
        Results = []
        # take a dataset 
        oci_cfg = OCIParams()
        bbci_cfg = BCI3Params()
        phy_cfg = PhysionetParams()
        overall = []
        ANS = {}
        dataset= [bbci_cfg, phy_cfg,oci_cfg]

        ANS['fit_time'] = []
        ANS['score_time'] = []
        ANS['test_f1'] = []
        ANS['test_acc'] = []
        ANS['test_roc'] = []
        ANS['cv_time'] = []
        ANS['preproc'] = []
        ANS['features'] = []
        ANS['clasfier'] = []
        ANS['Feat+Clf'] = []
        ANS['dataset'] = []

        # take an algo
        for ds in dataset:
            # algo = [EEGNetParams.name, None]
            
            if ds.name =='Physionet':
                runs = [5, 6, 9, 10, 13, 14] #[3, 4, 7, 8, 11, 12]
                person_id = 1
                raw = extractPhysionet(runs, person_id)
            elif ds.name == 'BCI3IVa':
                raw = extractBCI3( ) 
            elif ds.name == 'OCIParams':
                raw = extractOCI(Expr_name = 'P2_Day11_125')
            
            bsa = BrainSignalAnalysis(raw, data_cfg = ds)
            # take a preprocessing step
            procs = ['ica', 'ssp_car_ica','ssp', 'car',  'car_ica', 'ssp_car', 'ssp_ica']
            for proc in procs:
                # take a featuure extraction
                classes = ['_STAT_ATTNnet', '_RAW_EEGnet_CNN', '_TF_EEGnet_CNN'] #'_IMG_CasCnnRnnnet']
                for cl in classes:
                    ans = {'fit_time':[], 'score_time':[], 'test_f1':[], 
                    'test_acc':[], 'test_roc':[], 'cv_time':[]}
                    # repeat every experiment 10 times to get the variation
                    cv = 2
                    for i in range(cv):
                        raw1 = raw.copy()
                        methods = ds.name+'_locl_norm_'+ proc+ cl
                        print(methods)
                        start = time.time()
                        bsa = BrainSignalAnalysis(raw1, data_cfg = ds)
                        bsa.artifact_removal(methods, save_epochs = False)
                        bsa.feature_extraction(methods, save_feat = True)
                        extractf = ds.name+'_'+ '_'.join(methods.split('_')[:-2])+'.npz'
                        del bsa, raw1
                        if cl.find('EEGnet')!=-1:
                            nCfg = EEGNetParams()
                        elif cl.find('ATTNnet')!=-1:
                            nCfg = ATTNnetParams()
                        elif cl.find('CasCnnRnnnet')!=-1:
                            nCfg =CasCnnRnnnetParams()

                        acc_test, f1_test, roc_test, cf_val = run.run(filename=extractf, dCfg= ds, nCfg = nCfg)
                        cv_time = time.time() - start

                        ans['fit_time'] = np.hstack(( ans['fit_time'] ,0))
                        ans['score_time'] = np.hstack(( ans['score_time'] ,0))
                        ans['test_f1'] = np.hstack(( ans['test_f1'] ,f1_test))
                        ans['test_acc'] = np.hstack(( ans['test_acc'] ,acc_test))
                        ans['test_roc'] = np.hstack(( ans['test_roc'] ,roc_test))
                        ans['cv_time'] = np.hstack(( ans['cv_time'] ,cv_time))

                    ANS['fit_time'].extend(ans['fit_time'].tolist())

                    ANS['score_time'].extend(ans['score_time'].tolist())                    

                    ANS['test_f1'].extend(ans['test_f1'].tolist())

                    ANS['test_acc'].extend(ans['test_acc'].tolist())    

                    ANS['test_roc'].extend(ans['test_roc'].tolist())    

                    ANS['cv_time'].extend(ans['cv_time'].tolist())

                    ANS['preproc'].extend([proc]*cv)
                    ANS['features'].extend(['_'.join(cl.split('_')[1:2])]*cv)
                    ANS['clasfier'].extend(['_'.join(cl.split('_')[2:])]*cv)
                    ANS['Feat+Clf'].extend([cl[1:]]*cv)
                    ANS['dataset'].extend([ds.name]*cv)

                    # overall.extend(ANS)
                        # ans[methods+str(i)] = scores
    finally:
        np.savez('DL_results', [ANS])    

if __name__ =='__main__':
    DL_methods()