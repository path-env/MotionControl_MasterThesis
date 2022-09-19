#!/usr/bin/python3
import rospy
import time

from pylsl import StreamInlet, resolve_stream
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
import brainflow as bf

from carla_msgs.msg import CarlaEgoVehicleControl, CarlaStatus
from std_msgs.msg import Bool

import ros_compatibility as roscomp
from ros_compatibility.node import CompatibleNode
from ros_compatibility.qos import QoSProfile, DurabilityPolicy

import numpy as np
import torch


try:
    import pygame
    from pygame.locals import KMOD_CTRL
    from pygame.locals import KMOD_SHIFT
    from pygame.locals import K_COMMA
    from pygame.locals import K_DOWN
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_F1
    from pygame.locals import K_LEFT
    from pygame.locals import K_PERIOD
    from pygame.locals import K_RIGHT
    from pygame.locals import K_SLASH
    from pygame.locals import K_SPACE
    from pygame.locals import K_UP
    from pygame.locals import K_a
    from pygame.locals import K_d
    from pygame.locals import K_h
    from pygame.locals import K_m
    from pygame.locals import K_p
    from pygame.locals import K_q
    from pygame.locals import K_s
    from pygame.locals import K_w
    from pygame.locals import K_b

except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')   
# ==============================================================================
# -- KeyboardControl -----------------------------------------------------------
# ==============================================================================


class BCIControl(object):
    """
    Handle BCI events
    """
    def __init__(self, role_name, hud, node):
        self.role_name = role_name
        self.hud = hud
        self.node = node
        jit_model = '/media/mangaldeep/HDD2/workspace/MotionControl_MasterThesis/models/EEGnet_md_script.pt'
        self.bci_interface = BCI(jit_model)
        self._autopilot_enabled = False
        self._control = CarlaEgoVehicleControl()
        self._steer_cache = 0.0

        fast_qos = QoSProfile(depth=10)
        fast_latched_qos = QoSProfile(depth=10, durability=DurabilityPolicy.TRANSIENT_LOCAL)

        self.vehicle_control_manual_override_publisher = self.node.new_publisher(
            Bool,f"/carla/{self.role_name}/vehicle_control_manual_override", qos_profile=fast_latched_qos)

        self.vehicle_control_manual_override = False

        self.auto_pilot_enable_publisher = self.node.new_publisher(
            Bool,f"/carla/{self.role_name}/enable_autopilot", qos_profile=fast_qos)

        self.vehicle_control_publisher = self.node.new_publisher(
            CarlaEgoVehicleControl,f"/carla/{self.role_name}/vehicle_control_cmd_manual", qos_profile=fast_qos)

        self.carla_status_subscriber = self.node.new_subscription(
            CarlaStatus,  "/carla/status", self._on_new_carla_frame, qos_profile=10)

        self.set_autopilot(self._autopilot_enabled)

        self.set_vehicle_control_manual_override(
            self.vehicle_control_manual_override)  # disable manual override

    def set_vehicle_control_manual_override(self, enable):
        """
        Set the manual control override
        """
        self.hud.notification('Set vehicle control manual override to: {}'.format(enable))
        self.vehicle_control_manual_override_publisher.publish((Bool(data=enable)))

    def set_autopilot(self, enable):
        """
        enable/disable the autopilot
        """
        self.auto_pilot_enable_publisher.publish(Bool(data=enable))

    # pylint: disable=too-many-branches
    def parse_events(self, clock):
        """
        parse an input event
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            elif event.type == pygame.KEYUP:
                if self._is_quit_shortcut(event.key):
                    return True
                elif event.key == K_F1:
                    self.hud.toggle_info()
                elif event.key == K_h or (event.key == K_SLASH and
                                          pygame.key.get_mods() & KMOD_SHIFT):
                    self.hud.help.toggle()
                elif event.key == K_b:
                    self.vehicle_control_manual_override = not self.vehicle_control_manual_override
                    self.set_vehicle_control_manual_override(self.vehicle_control_manual_override)
                if event.key == K_q:
                    self._control.gear = 1 if self._control.reverse else -1
                elif event.key == K_m:
                    self._control.manual_gear_shift = not self._control.manual_gear_shift
                    self.hud.notification(
                        '%s Transmission' %
                        ('Manual' if self._control.manual_gear_shift else 'Automatic'))
                elif self._control.manual_gear_shift and event.key == K_COMMA:
                    self._control.gear = max(-1, self._control.gear - 1)
                elif self._control.manual_gear_shift and event.key == K_PERIOD:
                    self._control.gear = self._control.gear + 1
                elif event.key == K_p:
                    self._autopilot_enabled = not self._autopilot_enabled
                    self.set_autopilot(self._autopilot_enabled)
                    self.hud.notification('Autopilot %s' %
                                          ('On' if self._autopilot_enabled else 'Off'))
        if not self._autopilot_enabled and self.vehicle_control_manual_override:
            self._parse_vehicle_keys(pygame.key.get_pressed(), clock.get_time())
            self._control.reverse = self._control.gear < 0

    def _on_new_carla_frame(self, data):
        """
        callback on new frame

        As CARLA only processes one vehicle control command per tick,
        send the current from within here (once per frame)
        """
        if not self._autopilot_enabled and self.vehicle_control_manual_override:
            try:
                self.vehicle_control_publisher.publish(self._control)
            except Exception as error:
                self.node.logwarn("Could not send vehicle control: {}".format(error))

    def _parse_vehicle_keys(self, keys, milliseconds):
        """
        parse key events
        """
        steer_pro,_ = self.bci_interface.getCmd()
        self._control.throttle = 1.0 if keys[K_UP] or keys[K_w] else 0.2
        steer_increment = 5e-4 * milliseconds

        steer_op = torch.argmax(steer_prob)
        if steer_op == 1:
            self.hud.notification('BCI - Steer Neutral')            
            self._steer_cache =0
        elif steer_op == 0:
            self.hud.notification('BCI - Steer Left')    
            self._steer_cache -= steer_increment
        elif steer_op == 2:
            self.hud.notification('BCI - Steer Right')    
            self._steer_cache += steer_increment
        else:
            self.hud.notification('BCI - Unknown command')   
            self._steer_cache = 0.0

        self._steer_cache = min(0.7, max(-0.7, self._steer_cache))
        self._control.steer = round(self._steer_cache, 1)
        self._control.brake = 1.0 if keys[K_DOWN] or keys[K_s] else 0.0
        self._control.hand_brake = bool(keys[K_SPACE])

    @staticmethod
    def _is_quit_shortcut(key):
        return (key == K_ESCAPE) or (key == K_q and pygame.key.get_mods() & KMOD_CTRL)



class BCI():
    def __init__(self, jit_model) -> None:
        self.role_name = 'bci_car'
        # self._init_ros_control()
        # self._bf_comm()
        # self._lsl_comm()

        self.steer = 0
        self.model = torch.jit.load(jit_model)
        self.model = self.model.to('cuda').eval()
        
        # self.manual_ctrl_ovrrd = rospy.Publisher(f'/carla/{self.role_name}/vehicle_control_manual_override', Bool, queue_size=1)
        # self.auto_pilot = rospy.Publisher(f'/carla/{self.role_name}/enable_autopilot', Bool, queue_size=1)
        # self.egoCMD =  rospy.Publisher(f'/carla/{self.role_name}/vehicle_control_cmd_manual', CarlaEgoVehicleControl, queue_size=1)
        EPOCH_len = 3 #sec
        freq = 110
        self.n_samp = 1 + EPOCH_len * freq

    def _bf_comm(self):
        # 2, Using Brain FLow
        params = BrainFlowInputParams ()
        params.timeout = 10
        params.serial_port = '/dev/ttyUSB0'
        self.eeg_ch = BoardShim.get_eeg_channels(BoardIds.CYTON_DAISY_BOARD.value)
        self.board = BoardShim (BoardIds.CYTON_DAISY_BOARD, params)
        try:
            self.board.prepare_session()
        except Exception as e:
            print('board is off')
            exit(1)
        self.board.start_stream()
        self.s_freq = self.board.get_sampling_rate(BoardIds.CYTON_DAISY_BOARD)

    def _lsl_comm(self):
        # 1, Using LSL 
        # first resolve an EEG stream on the lab network
        print("looking for an EEG stream...")
        streams = resolve_stream('type', 'EEG')
        # create a new inlet to read from the stream
        self.inlet = StreamInlet(streams[0])

    # def _init_ros_control(self):
    #     self.control = CarlaEgoVehicleControl()
    #     # Header
    #     self.control.header.seq = np.uint32(1)
    #     self.control.header.stamp.secs = 0
    #     self.control.header.stamp.nsecs = 0
    #     self.control.header.frame_id = ''

    #     # body
    #     self.control.throttle = 0.5  #float32	Scalar value to cotrol the vehicle throttle: [0.0, 1.0]
    #     self.control.steer = 0   #float32	Scalar value to control the vehicle steering direction: [-1.0, 1.0]
    #             # to control the vehicle steering
    #     self.control.brake = 0 #float32	Scalar value to control the vehicle brakes: [0.0, 1.0]
    #     self.control.hand_brake = False # bool	If True, the hand brake is enabled.
    #     self.control.reverse = False  #bool	If True, the vehicle will move reverse.
    #     self.control.gear = 0 #	int32	Changes between the available gears in a vehicle.
    #     self.control.manual_gear_shift = False	#bool	If True, the gears will be shifted using gear.

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
        # self.manual_ctrl_ovrrd.publish((Bool(data=True)))
        # self.auto_pilot.publish((Bool(data=False)))

        eeg_TS = self._get_data(1) # pass input argument to use brainflow stream
        # eeg_TS = np.zeros((16,self.n_samp))
        eeg_TS = (eeg_TS - eeg_TS.min()) / (eeg_TS.max() - eeg_TS.min())
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
        # print(f'{action}, {steer}')

        # self.control.steer = steer
        # self.egoCMD.publish(self.control) 
        return prob, eeg_TS

# def main(jit_model):
#     obj = BCIcontrol(jit_model)
#     rospy.init_node('bci', anonymous=True)
#     rate = rospy.Rate(2) # 10hz
#     try:
#         while not rospy.is_shutdown():
#             obj.getCmd()
#             # obj._get_data(1)
#     except KeyboardInterrupt:
#         rospy.loginfo("User requested shut down.")

# if __name__ == '__main__':
#     jit_model = '/media/mangaldeep/HDD2/workspace/MotionControl_MasterThesis/models/EEGnet_md_script.pt'
#     main(jit_model)
