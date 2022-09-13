#!/usr/bin/python3
import rospy
import time

from pylsl import StreamInlet, resolve_stream
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
import brainflow as bf

from carla_msgs.msg import CarlaEgoVehicleControl
from std_msgs.msg import Bool

import numpy as np
import torch
        
class BCIcontrol():
    def __init__(self, jit_model) -> None:
        self.role_name = 'bci_car'
        self._init_ros_control()
        # self._bf_comm()
        # self._lsl_comm()

        self.steer = 0
        self.model = torch.jit.load(jit_model)
        self.model = self.model.to('cuda').eval()
        
        self.manual_ctrl_ovrrd = rospy.Publisher(f'/carla/{self.role_name}/vehicle_control_manual_override', Bool, queue_size=1)
        self.auto_pilot = rospy.Publisher(f'/carla/{self.role_name}/enable_autopilot', Bool, queue_size=1)
        self.egoCMD =  rospy.Publisher(f'/carla/{self.role_name}/vehicle_control_cmd_manual', CarlaEgoVehicleControl, queue_size=1)
        EPOCH_len = 3 #sec
        freq = 110
        self.n_samp = 1 + EPOCH_len * freq

    def _bf_comm(self):
        # 2, Using Brain FLow
        params = BrainFlowInputParams ()
        params.timeout = 10
        params.serial_port = '/dev/ttyUSB0'
        self.eeg_ch = BoardShim.get_eeg_channels (BoardIds.CYTON_DAISY_BOARD.value)
        self.board = BoardShim (BoardIds.CYTON_DAISY_BOARD, params)
        self.board.prepare_session()
        self.board.start_stream()
        self.s_freq = self.board.get_sampling_rate(BoardIds.CYTON_DAISY_BOARD)

    def _lsl_comm(self):
        # 1, Using LSL 
        # first resolve an EEG stream on the lab network
        print("looking for an EEG stream...")
        streams = resolve_stream('type', 'EEG')
        # create a new inlet to read from the stream
        self.inlet = StreamInlet(streams[0])

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

    def _get_data(self):# ip stream has filter, could be activated in OpenBCIGUI
        eeg_TS = []
        for i in range(self.n_samp):
            sample, timestamp = self.inlet.pull_sample()#(max_samples=n_samp)
            eeg_TS.append(sample)
        # print(len(eeg_TS))
        if eeg_TS == []:
            return np.zeros((16,self.n_samp))
        else:
            return np.transpose(eeg_TS)

    def _get_data(self, bfdummy): # ip stream has no filter, butterflow filter added manually
        data = self.board.get_current_board_data(num_samples= self.n_samp)
        # print(data.shape)
        eeg_TS = data[self.eeg_ch, :]
        # eeg_TS = eeg_TS / 1000000 # BrainFlow returns uV, convert to V for MNE
        # print(eeg_TS.shape)
        eeg_channels = BoardShim.get_eeg_channels(BoardIds.CYTON_DAISY_BOARD.value)
        for i in range(len(eeg_channels)):
            bf.DataFilter.perform_bandpass(eeg_TS[i], self.s_freq, 0.5, 50,4, 0, ripple=0.1)

        if eeg_TS.shape[1] != self.n_samp:
            return np.zeros((16,self.n_samp))
        else:
            return eeg_TS

    def getCmd(self):
        self.manual_ctrl_ovrrd.publish((Bool(data=True)))
        self.auto_pilot.publish((Bool(data=False)))

        # eeg_TS = self._get_data(1) # pass input argument to use brainflow stream
        eeg_TS = np.zeros((16,self.n_samp))
        self.eeg_TS = torch.tensor(eeg_TS).unsqueeze(0).unsqueeze(0)
        self.eeg_TS = self.eeg_TS.to('cuda', torch.float32)
        # print(self.eeg_TS.shape)
        op = self.model(self.eeg_TS)
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
        print(f'{action}, {steer}')

        self.control.steer = steer
        self.egoCMD.publish(self.control)    

def main(jit_model):
    obj = BCIcontrol(jit_model)
    rospy.init_node('bci', anonymous=True)
    rate = rospy.Rate(2) # 10hz
    try:
        while not rospy.is_shutdown():
            obj.getCmd()
            # obj._get_data(1)
    except KeyboardInterrupt:
        rospy.loginfo("User requested shut down.")

if __name__ == '__main__':
    jit_model = '/media/mangaldeep/HDD2/workspace/MotionControl_MasterThesis/models/EEGnet_md_script.pt'
    main(jit_model)
