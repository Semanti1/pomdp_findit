from interbotix_sdk.robot_manipulation import InterbotixRobot
from interbotix_descriptions import interbotix_mr_descriptions as mrd
import numpy as np
import serial
import time
# run this command(roslaunch interbotix_sdk arm_run.launch robot_name:=vx300 use_time_based_profile:=true gripper_operating_mode:=pwm)
# run this file in a separete terminal

# documentation: https://github.com/Interbotix/interbotix_ros_arms/blob/master/interbotix_sdk/src/interbotix_sdk/robot_manipulation.py
# both the interbotix_sdk and interbotix_descriptions folders need to be added to the Python PATH to function properly
# coordinate system: given the robot arm's resting orientation, +x is forward, +y is left, and +z is up

# use set_ee_pose_components() to direct the end effector of the arm to an absolute (x, y, z) position
# use set_ee_cartesian_trajectory() to direct end effector of the arm using a straight line path using relative positions
# both functions will NOT move if the function does not find a valid inverse kinematics path to target

# open serial channel for Arduino communication
ser = serial.Serial('/dev/ttyACM0', 9600)
# declare arm type for movement commands
arm = InterbotixRobot(robot_name="vx300", mrd=mrd)

#z-axis height of object being observed
object_height = 0.09

# lowers arm to check if an object is present, returning True if the force sensor
# returns a value higher than 300 and False otherwise
def check_pose(x, y, z):
    arm.set_ee_pose_components(x, y, z=object_height, pitch=1.5)
    # takes 3 readings to alleviate random error
    for i in range(3):
        string = None
        while not string:
            # flush serial to ensure readings are live
            ser.flush()
            ser.flushInput()
            b = ser.readline()     # read a byte string
            string_n = b.decode()  # decode byte string into Unicode  
            string = string_n.rstrip() # remove \n and \r
            if string:
                flt = float(string)    # convert string to float
                print(flt)
                if flt > 300:
                    return True
            time.sleep(0.01)          
    arm.set_ee_pose_components(x, y, z, pitch=1.5)
    return False

def main():
    arm.set_ee_pose_components(x=0.3, z=0.2)
    arm.set_ee_cartesian_trajectory(pitch=1.5)


    # arbitrary locations, checks a 3x3 grid of locations to find object
    for i in range(3):
        for j in range(3):
            x_pos = 0.2+0.1*i
            y_pos = 0.1*j
            arm.set_ee_pose_components(x=x_pos, y=y_pos, z=0.15, pitch=1.5)
            if check_pose(x_pos, y_pos, 0.15):
                # hard-coded sequence of movements to drop object off at arbitrary location
                arm.set_ee_pose_components(x=x_pos, y=y_pos, z=0.15, pitch=1.5)
                arm.open_gripper(delay=1.5)
                arm.set_ee_pose_components(x=x_pos, y=y_pos, z=0.05, pitch=1.5)
                arm.close_gripper(delay=1.5)
                arm.set_ee_pose_components(x=0, y=0.3, z=0.3, pitch=1.5)
                arm.open_gripper(delay=1.5)
                arm.close_gripper(delay=2.0)
                arm.set_ee_pose_components(x=0.3, z=0.2)
                arm.go_to_sleep_pose()
                return
            
            print((i,j))

    arm.set_ee_pose_components(x=0.3, z=0.2)
    arm.go_to_sleep_pose()

if __name__=='__main__':
    main()