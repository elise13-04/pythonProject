import time
import math
from numpy.distutils.system_info import sfftw_info


def filtre_detect_ligne( v, seuil):
    if v >= seuil:
        return True
    else:
        return False

class RobotControl:
    def __init__(self):
        # set some constants you may find useful
        self.distBetweenWheels = 0.12
        self.nTicksPerRevol = 1024
        self.wheelDiameter = 0.06

    def test_move_until_obstacle (self,rb,flt,speed_left,speed_right,dist_obstacle,duration_max):
        """
        Example of test function to check if the robot moves and stops on front obstacle

        input parameters :
        rb : robot object
        flt : filter object for filtering front sonar
        speed_left : speed command of left wheel
        speed_right : speed command of right wheel
        dist_obstacle : minimu distance to front obstacle
        duration_max : maximum duration of the move (to avoid infinite loop)

        output paremeters :
        None
        """
        loop_iter_time = 0.1 # control at 10 Hz (10 commands/second)
        t_start = time.time()
        while True:
            # start control loop
            rb.start_control_loop(loop_iter_time) 
            if (time.time() - t_start) > duration_max:
                break # max time reached , escape the loop ...
            df = rb.get_sonar("front") # get distance from front sonar
            dff = flt.doNothingFilter(df) # filter the distance (does nothing, juste for test)
            if dff > 0.0:
                print ("front sonar distance = %.2f m"%(df))  # debug : print front distance on terminal
            if dff>0.0 and df<dist_obstacle:
                break # obstacle at less than 25 cm , escape the loop ...
            # forward motion
            rb.set_speed(speed_left, speed_right)
            # end of loop
            rb.end_control_loop(warning=True) # warning=True to print warning if loop is too long
        # stop the robot 
        rb.stop()

    def suivre_ligne(self,rb,durmax,seuil):
        loop_iter_time = 0.05
        to = time.time()
        while True:
            # start control loop
            rb.start_control_loop(loop_iter_time)
            if (time.time()-to)>durmax:
                break
            if 0.6>rb.get_sonar("front")>0:
                print(rb.get_sonar("front"))
                break
            if rb.lineSensorMiddle>seuil:
                rb.set_speed(900,900)
            else:
                if rb.lineSensorRight>seuil:
                    rb.set_speed(500,400)
                elif rb.lineSensorLeft>seuil:
                    rb.set_speed(400,500)
                else:
                    print("baisse la freq")
            rb.end_control_loop(warning=True)  # warning=True to print warning if loop is too long
        rb.stop()

    def avancer_pour_eviter_obst(self,rb,distmin,durmax):
        loop_iter_time=0.1
        to=time.time()
        while True:
            rb.start_control_loop(loop_iter_time)
            if(time.time()-to)>durmax:
                break
            if 0 < rb.get_sonar(name="front") < distmin:
                break
            rb.set_speed(350,350)
            rb.end_control_loop(warning=True)
        rb.stop()


    def tourner(self,rb,distmin):
        loop_iter_time = 0.1
        deltatick = 503
        rb.stop()
        time.sleep(0.1)
        tickfr = rb.get_odometers()[1]

        sf = rb.get_sonar(name="front")
        sr = rb.get_sonar(name="right")
        sl = rb.get_sonar(name="left")
        print('sl=',sl,' ','sr=',sr)
        speed = 50
        distcotes=0.4

        if 0 < sf < distmin:
            if 0 < sl < distcotes:
                tickfr -= deltatick
            elif 0 < sr < distcotes:
                tickfr += deltatick
            elif 0 < sl < distcotes and 0 < sr < distcotes:
                rb.stop()
                return
            else:
                rb.stop()
                return
        else:
            rb.stop()
            return

        while True:
            rb.start_control_loop(loop_iter_time)

            tick = rb.get_odometers()[1]

            if tick < tickfr and 0 < sr < distmin:

                rb.set_speed(-speed, speed)
                print(tick, tickfr,"tourne à gauche")

            elif tick > tickfr and 0 < sl < distmin:

                rb.set_speed(speed, -speed)
                print(tick,tickfr,"tourne à droite")
            else:
                break
            rb.end_control_loop(warning=True)
        rb.stop()

    def suivi_de_mur(self, rb,durmax,distmin,seuil):
        loop_iter_time = 0.1
        to = time.time()
        while True:
            rb.start_control_loop(loop_iter_time)
            if (time.time() - to) > durmax:
                break
            if 0 < rb.get_sonar(name="front") < distmin:
                break
            if rb.lineSensorMiddle > seuil and rb.get_sonar("front")<=0 :
                break
            else:
                if rb.get_sonar("right") > rb.get_sonar("left"):
                    rb.set_speed(410, 400)
                elif rb.get_sonar("left") > rb.get_sonar("right"):
                    rb.set_speed(400, 410)
                else:
                    rb.set_speed(400, 400)
            rb.end_control_loop(warning=True)
        rb.stop()















