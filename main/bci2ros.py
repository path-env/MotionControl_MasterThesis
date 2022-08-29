#%%
# #!/usr/bin/python3
import rospy
from carla_msgs.msg import CarlaEgoVehicleControl
from std_msgs.msg import Header
import numpy as np
import time

role_name = 'bci_car'
control = CarlaEgoVehicleControl()
# print(control)

# Header
control.header.seq = np.uint32(1)
control.header.stamp.secs = 0
control.header.stamp.nsecs = 0
control.header.frame_id = ''

# body
control.throttle = 0.5  #float32	Scalar value to cotrol the vehicle throttle: [0.0, 1.0]
control.steer = 0   #float32	Scalar value to control the vehicle steering direction: [-1.0, 1.0]
        # to control the vehicle steering
control.brake = 0 #float32	Scalar value to control the vehicle brakes: [0.0, 1.0]
control.hand_brake = False # bool	If True, the hand brake is enabled.
control.reverse = False  #bool	If True, the vehicle will move reverse.
control.gear = 0 #	int32	Changes between the available gears in a vehicle.
control.manual_gear_shift = False	#bool	If True, the gears will be shifted using gear.


# #%%
egoCMD =  rospy.Publisher(f'/carla/{role_name}/vehicle_control_cmd_manual', CarlaEgoVehicleControl, queue_size=1)


rospy.init_node('bci', anonymous=True)
rate = rospy.Rate(2) # 10hz
while not rospy.is_shutdown():
        egoCMD.publish(control)
