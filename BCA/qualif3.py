import rob1a_v04 as rob1a  # get the robot code
import control  # robot control functions
import filt # sensors filtering functions
import numpy as np
import math
import time

if __name__ == "__main__":
    rb = rob1a.Rob1A()   # create a robot (instance of Rob1A class)
    rb.set_pseudo ("Elise la best") # you can define your pseudo here (if not already done)
    ctrl = control.RobotControl() # create a robot controller

    # put your mission code here
    print("No log file")
    rb.set_log(1)  # do log data

    print("go")
    durmax = math.inf
    distmin=0.26
    loop_iter_time = 0.1
    to = time.time()
    seuil=0.8
    while True:
        ctrl.suivi_de_mur(rb,durmax,distmin,seuil)
        print('tourner')
        ctrl.tourner(rb, distmin)
        print('avancer')
    rb.stop()

    

