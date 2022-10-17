from main.bci_pipeline import BrainSignalAnalysis
from data.params import OCIParams
import rospy
import time
import mne
import numpy as np
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
import brainflow as bf
from mne_realtime import LSLClient
from pylsl import StreamInlet, resolve_stream
import torch
from data import brain_atlas  as bm

class OpenBciInterface():
    def __init__(self, jit_model) -> None:
        self.role_name = 'bci_car'
        EPOCH_len = 9 #sec
        dCfg = OCIParams()
        self.sfreq = dCfg.sfreq
        self.n_samp = 1 + EPOCH_len * self.sfreq
        self.timeout = 10
        self.eeg_ch = list(range(16))
        # self._init_ros_control()
        # self._bf_comm()
        # self._lsl_comm()
        self._mne_stream()
        self.steer = 0
        self.model = torch.jit.load(jit_model)
        self.model = self.model.to('cuda').eval()
        
        # self.manual_ctrl_ovrrd = rospy.Publisher(f'/carla/{self.role_name}/vehicle_control_manual_override', Bool, queue_size=1)
        # self.auto_pilot = rospy.Publisher(f'/carla/{self.role_name}/enable_autopilot', Bool, queue_size=1)
        # self.egoCMD =  rospy.Publisher(f'/carla/{self.role_name}/vehicle_control_cmd_manual', CarlaEgoVehicleControl, queue_size=1)

    def __del__(self):
        print('Streaming stopped')
        # self.board.release_session()
        try:
            self.board.stop_stream()
        except Exception as e:
            self.client.stop()
            pass
    
    def _bf_comm(self):
        # 2, Using Brain FLow
        params = BrainFlowInputParams ()
        params.timeout = self.timeout
        params.serial_port = '/dev/ttyUSB0'
        self.eeg_ch = BoardShim.get_eeg_channels(BoardIds.CYTON_DAISY_BOARD.value)
        self.board = BoardShim(BoardIds.CYTON_DAISY_BOARD, params)
        try:
            self.board.prepare_session()
            # time.sleep(2)
        except Exception as e:
            print('board is off')
            exit(1)
        self.board.start_stream(num_samples= self.n_samp)
        self.s_freq = self.board.get_sampling_rate(BoardIds.CYTON_DAISY_BOARD)

    def _lsl_comm(self):
        # 1, Using LSL 
        # first resolve an EEG stream on the lab network
        print("looking for an EEG stream...")
        streams = resolve_stream('type', 'EEG')
        # create a new inlet to read from the stream
        self.inlet = StreamInlet(streams[0])

    def _mne_stream(self):
        host = 'openbcigui'
        self.client = LSLClient(info=None, host=host, wait_max=self.timeout)
        try:
            self.client.start()
        except:
            print("Connection refused")

    """    
    def _init_ros_control(self):
        self.control = CarlaEgoVehicleControl()
        # Header
        self.control.header.seq = np.uint32(1)
        self.control.header.stamp.secs = 0
        self.control.header.stamp.nsecs = 0
        self.control.header.frame_id = ''

        # body
        self.control.throttle = 0.5  #float32	Scalar value to cotrol the vehicle throttle: [0.0, 1.0]
        self.control.steer = 0   #float32	Scalar value to control the vehicle steering direction: [-1.0, 1.0]
                # to control the vehicle steering
        self.control.brake = 0 #float32	Scalar value to control the vehicle brakes: [0.0, 1.0]
        self.control.hand_brake = False # bool	If True, the hand brake is enabled.
        self.control.reverse = False  #bool	If True, the vehicle will move reverse.
        self.control.gear = 0 #	int32	Changes between the available gears in a vehicle.
        self.control.manual_gear_shift = False	#bool	If True, the gears will be shifted using gear.
    """

    def get_data_lsl(self):# ip stream can have filter, could be activated in OpenBCIGUI
        data = []
        t = []
        for i in range(self.n_samp-1):
            sample, timestamp = self.inlet.pull_sample()#(max_samples=n_samp)
            data.append(sample)
            t.append(timestamp)
        data = np.array(data).T
        eeg_TS = self.dataFormat(data.copy())
        return t,eeg_TS

    def get_data_bf(self, call): # ip stream has no filter, butterflow filter added manually
        if call ==1:
            data = self.board.get_current_board_data(num_samples= self.n_samp)
        elif call ==2:
            data= self.board.get_board_data() #only to remove from ring buffer
        eeg_TS = self.dataFormat(data.copy())
        return eeg_TS

    def get_data_mne(self, event_id = 2):
        data = self.client.get_data_as_raw(12*self.sfreq)
        methods = 'locl_car_RAW_LDA_ML'
        data.rename_channels(dict(zip(data.ch_names, bm.oci_Channels)))
        event_data = np.uint16([[8*self.sfreq, 0,event_id]])
        ann = mne.annotations_from_events(event_data,sfreq=self.sfreq)
        data = data.set_annotations(ann)
        bsa = BrainSignalAnalysis(data)
        bsa.artifact_removal(methods, save_epochs = False)
        bsa.feature_extraction(methods, save_feat = False)
        eeg_TS = self.dataFormat(bsa.features[0,:,:-1])
        # eeg_TS = self.dataFormat(data.get_data()[0])
        return 1,eeg_TS       

    def dataFormat(self, data):
        if data.shape[1] == 0:
            print('Board is probably off')
            exit(1)
        eeg_TS = data[self.eeg_ch, :] /1000000
        # eeg_TS = eeg_TS / 1000000 # BrainFlow returns uV, convert to V for MNE
        # print(eeg_TS.shape)
        # eeg_channels = BoardShim.get_eeg_channels(BoardIds.CYTON_DAISY_BOARD.value)
        if eeg_TS.shape[1] != self.n_samp-1:
            eeg_TS = np.zeros((16,self.n_samp-1))
            print('Data not in required size, padding zeros....')
        # else:
            # for i in range(len(self.eeg_ch)):
            #     bf.DataFilter.perform_bandpass(eeg_TS[i], self.freq, 1, 49,4, 1, ripple=0.5)

            # eeg_TS = (eeg_TS - eeg_TS.min()) / (eeg_TS.max() - eeg_TS.min())
        eeg_TS = torch.tensor(eeg_TS).unsqueeze(0).unsqueeze(0)
        eeg_TS = eeg_TS.to('cuda', torch.float32)
        return eeg_TS

    def getCmd(self):
        # self.manual_ctrl_ovrrd.publish((Bool(data=True)))
        # self.auto_pilot.publish((Bool(data=False)))

        eeg_TS = self._get_data(1) # pass input argument to use brainflow stream
        # eeg_TS = np.zeros((16,self.n_samp))
        # print(self.eeg_TS.shape)
        op = self.model(eeg_TS)
        # torch.argmax(torch.softmax(eeg_TS, dim =1),dim=1).tolist()
        prob = torch.softmax(op, dim =1)
        op = torch.argmax(prob)
        # print(op)
        steer_increment = 5e-4 #* milliseconds

        if op == 0:
            self.steer = 0.0
            action = 'No Steer'
        elif op == 1:
            self.steer-=(op+steer_increment)
            action = 'Steer left'
        elif op == 2:
            self.steer+=(op+steer_increment)
            action = 'Steer right'

        steer = min(0.7, max(-0.7,self.steer))
        # print(f'{action}, {steer}')

        # self.control.steer = steer
        # self.egoCMD.publish(self.control) 
        return prob, eeg_TS