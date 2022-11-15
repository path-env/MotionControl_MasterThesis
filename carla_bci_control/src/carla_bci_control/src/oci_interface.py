#!/usr/bin/python3
from main.bci_pipeline import BrainSignalAnalysis
from data.params import OCIParams
import rospy
from std_msgs.msg import Float32
import time
import mne
import numpy as np
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
import brainflow as bf
from mne_realtime import LSLClient, MockLSLStream
from pylsl import StreamInlet, resolve_stream
import torch
from data import brain_atlas  as bm

from ros_compatibility.node import CompatibleNode
import ros_compatibility as roscomp
from threading import Thread

class OpenBciInterface(CompatibleNode):
    def __init__(self, jit_model) -> None:
        self.role_name = 'ego_vehicle' #self.get_param("role_name", "ego_vehicle")
        self.raw_len = 12 #sec
        self.dCfg = OCIParams()
        self.sfreq = self.dCfg.sfreq
        self.f_samp = self.raw_len * self.sfreq
        epoch_len = 9 #sec
        self.epoch_samp = epoch_len * self.sfreq +1
        self.timeout = 10
        self.eeg_ch = list(range(16))
        # self._init_ros_control()
        # self._bf_comm()
        # self._lsl_comm()
        # self._mne_stream()
        self.steer = 0
        self.device = 'cuda'
        self.model = torch.load(jit_model)
        self.model = self.model.to(self.device).eval()

        self.steer_cmd = rospy.Publisher(f'/carla/{self.role_name}/bci_steer_cmd', Float32, queue_size=100)
        
        # self.manual_ctrl_ovrrd = rospy.Publisher(f'/carla/{self.role_name}/vehicle_control_manual_override', Bool, queue_size=1)
        # self.auto_pilot = rospy.Publisher(f'/carla/{self.role_name}/enable_autopilot', Bool, queue_size=1)
        # self.egoCMD =  rospy.Publisher(f'/carla/{self.role_name}/vehicle_control_cmd_manual', CarlaEgoVehicleControl, queue_size=1)

    # def __del__(self):
    #     print('Streaming stopped')
    #     # self.board.release_session()
    #     try:
    #         self.board.stop_stream()
    #     except Exception as e:
    #         self.client.stop()
    #         pass

    def _genLSL(self):
        filename = ['Raw_P2_Day3_125.npz','Raw_P2_Day4_125.npz']
        train_x, train_y,events = [],[], []
        for file in filename:
            train = np.load(f'/media/mangaldeep/HDD2/workspace/MotionControl_MasterThesis/data/{file}', allow_pickle=True)
            train_x.append(np.float64(train['arr_0']))
            events.append(train['arr_1'])
            train_y.append(train['arr_2'])

        train_x = np.hstack(train_x)
        train_y = np.hstack(train_y)
        event_t = np.hstack(events)

        sfreq = self.dCfg.sfreq
        ch_names = bm.oci_Channels
        ch_types = ['eeg'] * len(ch_names)
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types);  
        raw = mne.io.RawArray(train_x, info, verbose=False);
        self.stream = MockLSLStream(host = 'openbcigui', raw = raw , ch_type = 'eeg')
        self.stream.start()

    def _bf_comm(self):
        # 2, Using Brain FLow
        self.n_samp = self.f_samp
        params = BrainFlowInputParams()
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
        self.n_samp = self.f_samp
        # first resolve an EEG stream on the lab network
        print("looking for an EEG stream...")
        streams = resolve_stream('type', 'EEG')
        # create a new inlet to read from the stream
        self.inlet = StreamInlet(streams[0])
        # time.sleep(15)

    def _mne_stream(self):
        self.n_samp = self.epoch_samp
        self.client = LSLClient(info=None, host='openbcigui', wait_max=1)
        try:
            self.client.start()
        except:
            self._mne_stream()
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
        # data , t = self.inlet.pull_chunk(max_samples= self.n_samp, timeout=self.timeout)
        data, t = [], []
        for i in range(self.n_samp):
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
        data = self.client.get_data_as_raw(self.raw_len * self.sfreq)
        methods = 'locl_car_RAW_LDA_ML'
        data.rename_channels(dict(zip(data.ch_names, bm.oci_Channels)))
        event_data = np.uint16([[8*self.sfreq, 0,event_id]])
        ann = mne.annotations_from_events(event_data,sfreq=self.sfreq, verbose=False)
        data = data.set_annotations(ann, verbose=False)
        bsa = BrainSignalAnalysis(data)
        bsa.artifact_removal(methods, save_epochs = False)
        bsa.feature_extraction(methods, save_feat = False)
        eeg_TS = self.dataFormat(bsa.features[0])
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
        if eeg_TS.shape[1] != self.n_samp:
            eeg_TS = np.zeros((16,self.n_samp))
            print('Data not in required size, padding zeros....')
        # else:
            # for i in range(len(self.eeg_ch)):
            #     bf.DataFilter.perform_bandpass(eeg_TS[i], self.freq, 1, 49,4, 1, ripple=0.5)

            # eeg_TS = (eeg_TS - eeg_TS.min()) / (eeg_TS.max() - eeg_TS.min())
        eeg_TS = torch.tensor(eeg_TS).unsqueeze(0).unsqueeze(0)
        eeg_TS = eeg_TS.to(self.device, torch.float32)
        return eeg_TS

    def getCmd(self):
        # self.manual_ctrl_ovrrd.publish((Bool(data=True)))
        # self.auto_pilot.publish((Bool(data=False)))

        _,eeg_TS = self.get_data_mne() # pass input argument to use brainflow stream
        # eeg_TS = np.zeros((16,self.n_samp))
        # print(self.eeg_TS.shape)
        op = self.model(eeg_TS.to(self.device, dtype = torch.float32))
        # torch.argmax(torch.softmax(eeg_TS, dim =1),dim=1).tolist()
        prob = torch.softmax(op, dim =1)
        op = torch.argmax(prob)
        # print(op)
        # steer_increment = 5e-2 #* milliseconds

        # if op == self.dCfg.event_dict_rec['right']:
        #     self.steer+=(steer_increment)
        #     action = 'Steer right'
        # elif op == self.dCfg.event_dict_rec['left']:
        #     self.steer-=(steer_increment)
        #     action = 'Steer left'
        # elif op == self.dCfg.event_dict_rec['none']:
        #     self.steer+=0
        #     action = 'No Steer'

        # steer = min(0.7, max(-0.7,self.steer))
        # print(f'{action}, {steer}')
        self.steer_cmd.publish(op.item())
        # print(op.item())
        # self.control.steer = steer
        # self.egoCMD.publish(self.control) 
        self.prob = prob
        return self.prob, eeg_TS


if __name__ =="__main__":
    roscomp.init('bci_ros')
    # rospy.init_node('bci_ros')
    jit_model = '/media/mangaldeep/HDD2/workspace/MotionControl_MasterThesis/models/EEGnet.pt'
    try:
        oci_node = OpenBciInterface(jit_model)
        # oci_node._genLSL() # comment this for online
        oci_node._mne_stream()
        executor = roscomp.executors.MultiThreadedExecutor()
        executor.add_node(oci_node)
        spin_thread = Thread(target=oci_node.spin)
        spin_thread.start()
        rate = rospy.Rate(1)
        t = time.time()
        while roscomp.ok():
            oci_node.getCmd()
            rate.sleep()
            # print((time.time() -t))
    except KeyboardInterrupt:
        roscomp.loginfo("User requested shut down.")
    finally:
        roscomp.shutdown()
        spin_thread.join()
        oci_node.stream.stop()
