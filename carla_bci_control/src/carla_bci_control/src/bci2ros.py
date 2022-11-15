#!/usr/bin/python3
from carla_msgs.msg import CarlaEgoVehicleControl, CarlaStatus
from std_msgs.msg import Bool, Float32

import ros_compatibility as roscomp
from ros_compatibility.node import CompatibleNode
from ros_compatibility.qos import QoSProfile, DurabilityPolicy

import numpy as np
import torch

from oci_interface import OpenBciInterface
from data.params import OCIParams
from threading import Thread

"""
# commands to start carla - BCI - ROS

OpenGUI to check if electrodes are not railed.

Run carla simulator: ./Carla.sh

roslaunch carla_bci_control carla_ros_bridge_with_bci_car_vehicle.launch 

Use keys P, B, M for toggling  manual control, autopilot off ....
"""

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
        self._autopilot_enabled = True
        self._control = CarlaEgoVehicleControl()
        self._steer_cache = 0.0
        self.dCfg = OCIParams()
        fast_qos = QoSProfile(depth=10)
        fast_latched_qos = QoSProfile(depth=10, durability=DurabilityPolicy.TRANSIENT_LOCAL)

        self.vehicle_control_manual_override_publisher = self.node.new_publisher(
            Bool,f"/carla/{self.role_name}/vehicle_control_manual_override", qos_profile=fast_latched_qos)

        self.vehicle_control_manual_override = True

        self.auto_pilot_enable_publisher = self.node.new_publisher(
            Bool,f"/carla/{self.role_name}/enable_autopilot", qos_profile=fast_latched_qos)

        self.vehicle_control_publisher = self.node.new_publisher(
            CarlaEgoVehicleControl,f"/carla/{self.role_name}/vehicle_control_cmd_manual", qos_profile=fast_qos)

        self.carla_status_subscriber = self.node.new_subscription(
            CarlaStatus,  "/carla/status", self._on_new_carla_frame, qos_profile=10)

        self.bci_steer_cmd_subscriber = self.node.new_subscription(
            Float32, f"/carla/{self.role_name}/bci_steer_cmd",self._steer_cmd , qos_profile=fast_latched_qos)

        self.set_autopilot(self._autopilot_enabled)

        self.set_vehicle_control_manual_override(
            self.vehicle_control_manual_override)  # disable manual override

        # jit_model = '/media/mangaldeep/HDD2/workspace/MotionControl_MasterThesis/models/EEGnet.pt'
        # self.bci_interface = OpenBciInterface(jit_model)
        # self.bci_interface._mne_stream()
        # self.bci_thread = Thread(target = self.bci_interface.getCmd)
        # self.bci_thread.start()

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
    
    def _steer_cmd(self, steerval):
        try:
            self.steer_op = steerval.data
        except Exception as error:
            self.node.logwarn("Could not read steer message: {}".format(error))

    def _parse_vehicle_keys(self, keys, milliseconds):
        """
        parse key events
        """
        # steer_prob,_ = self.bci_interface.getCmd()
        # steer_prob = self.bci_thread.prob
        # steer_prob = data
        self._control.throttle = 1.0 if keys[K_UP] or keys[K_w] else 0.2
        steer_increment = 5e-4 * milliseconds

        # steer_op = torch.argmax(steer_prob)
        if self.steer_op == self.dCfg.event_dict_rec['none']:
            self.hud.notification('BCI - Steer Neutral')            
            self._steer_cache =0
        elif self.steer_op == self.dCfg.event_dict_rec['left']:
            self.hud.notification('BCI - Steer Left')    
            self._steer_cache -= steer_increment
        elif self.steer_op == self.dCfg.event_dict_rec['right']:
            self.hud.notification('BCI - Steer Right')    
            self._steer_cache += steer_increment
        else:
            self.hud.notification('BCI - Unknown command')   
            self._steer_cache = 0.0

        self._steer_cache = min(0.5, max(-0.5, self._steer_cache))
        self._control.steer = round(self._steer_cache, 3)
        print(f'Steer Value: {self._control.steer}')
        self._control.brake = 1.0 if keys[K_DOWN] or keys[K_s] else 0.0
        self._control.hand_brake = bool(keys[K_SPACE])

    @staticmethod
    def _is_quit_shortcut(key):
        return (key == K_ESCAPE) or (key == K_q and pygame.key.get_mods() & KMOD_CTRL)

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
