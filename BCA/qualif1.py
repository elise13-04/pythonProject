import rob1a_v04 as rob1a  # get the robot code
import control  # robot control functions
import filt # sensors filtering functions
import time

if __name__ == "__main__":
    rb = rob1a.Rob1A()   # create a robot (instance of Rob1A class)
    rb.set_pseudo ("Elise la best") # you can define your pseudo here
    ctrl = control.RobotControl() # create a robot controller
    flt_front_sonar = filt.Filter() # create a filter for front sonar

    print ("No log file")
    rb.set_log(0)   # do not log data (set to 1 for logging data)

    print ("Go !!!")
    speed_left = 1600 # set speed on 11 bits [0 2047]
    speed_right = speed_left # same speed on both wheels, straight line (perfect robot)
    dist_obstacle = 0.30 # stops at 30 cm of a front obstacle
    duration_max = 60.0 # set a max duration of 1 minute
    # note : to use the robot and the filter in test_move_until_obstacle(), you need to pass them as arguments
    ctrl.test_move_until_obstacle (rb,flt_front_sonar,speed_left,speed_right,dist_obstacle,duration_max)
    print ("End of motion")

    rb.full_end() # clean end of simulation
