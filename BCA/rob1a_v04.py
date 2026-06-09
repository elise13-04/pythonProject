import threading
import sys
import struct
import time
import signal
#import numpy as np
import zmq
import binascii
import os

"""
UE 2.2 sensor actuator loop
Simple robot for 1A , 2 Wheels drive with castor wheel
Actuators : differencial command with two motors
Sensors : 4 ultrasonic telemeters, 2 encoders, magnetic compass
V03 : main changes , communication socket with VREP is replaced by publishers/subscribers (using zmq)
V04 : main change : use simTime as reference for time in the python control programs
Do not change this program - Ne modifiez pas ce programme
"""

class Rob1A:
    # interrupt handler
    def interrupt_handler (self, signal, frame):
        print ('You pressed Ctrl+C! Rob1A will stop in the next 2 seconds ')
        try:
            vrep_alive = self.vrep.isAlive()
        except:
            try:
                vrep_alive = self.vrep.is_alive()
            except:
                pass
        if vrep_alive:
            self.set_speed(0,0)
            self.full_end()
        sys.exit(0)

    def __init__ (self, eval=0):
        self.subs_ok = False
        self.kill_sent = False # use for timeout stop

        self.__pseudo = "pseudoHasNotBeenDefinedYet"
        self.eval = eval
        self.status_eval = eval
        self.do_log = 0
        self.status_log = 0
        self.robot_alive = 0

        self.do_save_log = 0
        self.status_save_log = 0
        self.do_save_eval = 0
        self.status_save_eval = 0    
        self.ctrl_py = 0    

        self.simulation_alive = False

        self.simTimeLoopStart = None
        self.cpuTimeLoopStart = None
        self.dt_loop = 0.05
        self.eps_dt_loop = 0.005
        self.simTimeLoopEnd = None
        self.cpuTimeLoopEnd = None
        self.simTimeLoopDuration = None
        self.cpuTimeLoopDuration = None

        self.speed_left = 0
        self.speed_right = 0

        self.distFront = 0.0
        self.distLeft = 0.0
        self.distRight = 0.0
        self.distBack = 0.0

        self.magneticSensorX = 0.0 # new 2022 (magx,magy components as on IMU)
        self.magneticSensorY = 0.0 # 

        self.lineSensorLeft = 0.0 # new 2022 , painted center line sensors
        self.lineSensorMiddle = 0.0
        self.lineSensorRight = 0.0        

        self.encoderLeft = 0
        self.encoderRight = 0

        self.radar_beacons_dist = [0.,0,0,0]
        self.radar_beacons_azim = [0.,0,0,0]

        self.magneticSensorCalibration = [0,0,0,0]
        
        self.port_cmds = "17272"
        self.port_sensors = "17271"
        self.zmq_context = zmq.Context()
        self.socket_pub_cmds = self.zmq_context.socket(zmq.PUB)
        self.socket_pub_cmds.bind("tcp://*:%s" % self.port_cmds)
        self.socket_sub_sensors = self.zmq_context.socket(zmq.SUB)
        self.socket_sub_sensors.connect ("tcp://localhost:%s" % self.port_sensors)
        #self.socket_sub_sensors.subscribe("1111")
        self.socket_sub_sensors.subscribe("") # subscribe to all topics
        
        self.rob1a_ready = threading.Event()
        self.rob1a_ready.clear()
        self.vrep = threading.Thread(target=self.subscribe_sensors,args=(self.socket_sub_sensors,self.rob1a_ready,))
        self.vrep.start()
        # wait for robot to be ready
        self.rob1a_ready.wait()
        
        time.sleep(0.2) # wait for socket to start , empirical !!
        # trap hit of ctrl-x to stop robot and exit (emergency stop!)
        signal.signal(signal.SIGINT, self.interrupt_handler)
        print ("Rob1A ready ...")
        # tell the sim we are ready to control the bot
        self.ctrl_py = 1
        self.set_cmds()
        # store start time
        self.rob_start_time = time.time()

    def set_pseudo(self,pseudo):
        self.__pseudo = pseudo

    def get_pseudo(self):
        return self.__pseudo

    def full_end (self):
        """
        Fully stop the simulation of the robot , set motors speed to 0
        and close the connection with the simulator vrep
        sleep for 1 second to end cleanly the log file
        """
        self.set_speed(0,0)
        self.rob_stop_time = time.time()
        mission_full_duration = self.rob_stop_time - self.rob_start_time

        print ("mission full duration : %.2f"%(mission_full_duration))

        print ("clean stop of  Rob1A")
        print (self.subs_ok)
        # clean end of com with simulator , only if simulator is connected ...
        if self.subs_ok:  
            self.save_log_n_eval()
            self.bye()
        
        # stop subscribe_sensors thread 
        self.simulation_alive = False
        # print (self.vrep)
        # print (self.vrep.is_alive())
        # print (dir(self.vrep))
        self.vrep.join()
        time.sleep(0.25)
        print ("out!")

    def set_save (self,item, do_save): 
        # 1 ask , 2 ack
        # print ("set_save",item,do_save)
        send_cmds = False
        if item == "log":
            self.do_save_log = do_save
            send_cmds = True
        elif item == "eval":
            self.do_save_eval = do_save
            send_cmds = True
        if send_cmds:
            self.set_cmds ()

    def bye(self):
        print ("bye ...")
        self.ctrl_py = 0
        self.set_cmds ()

    def set_eval (self,eval):
        self.eval = eval
        self.set_cmds ()

    def save_log_n_eval (self):
        dt_wait = 0.25
        # save log to file (wait for completion)
        if self.do_log == 1:
            print ("save log data ...")
            self.set_save("log",1) # ask for saving log in file
            # print ("status save log",self.status_save_log)
            while self.status_save_log != 2:
                time.sleep(dt_wait)
                # print ("status save log",self.status_save_log)
            self.set_save("log",2) # acknowledge log file saved
            time.sleep(dt_wait)

        # save eval to file (wait for completion)
        if self.eval == 1:
            print ("save eval data ...")
            fps = open("pseudo.txt","w")
            fps.write ("%s\n"%(self.get_pseudo()))
            fps.close()
            self.set_save("eval",1) # ask for saving log in file
            while self.status_save_eval != 2:
                time.sleep(dt_wait)
            self.set_save("eval",2) # acknowledge log file saved
            time.sleep(dt_wait)

    def subscribe_sensors (self, socket, ev):
        poller = zmq.Poller()
        poller.register(socket, zmq.POLLIN)
        t0 = None
        dt_sleep = 0.0 # 0.025
        ev.set()    
        self.simulation_alive = True
        print ("starting sub loop in subscribe_sensors()")
        while True:
            # print ("Alive",self.simulation_alive)
            socks = dict(poller.poll(1000)) # 1 s timeout
            if socket in socks and socks[socket] == zmq.POLLIN:
                binmsg = socket.recv()
                fmt = "<4sd4f2i3f4i8f4i"
                topic,simTime,distFront,distLeft,distRight,distBack,magX,magY,lineLeft,lineMiddle,lineRight,odoLeft,odoRight,pub_cnt,status_all,d1,a1,d2,a2,d3,a3,d4,a4,magx0,magx1,magy0,magy1 = struct.unpack(fmt,binmsg)
                self.distFront = distFront
                self.distLeft = distLeft
                self.distRight = distRight
                self.distBack = distBack
                self.magneticSensorX = magX
                self.magneticSensorY = magY
                self.lineSensorLeft = lineLeft
                self.lineSensorMiddle = lineMiddle
                self.lineSensorRight = lineRight
                self.encoderLeft = odoLeft
                self.encoderRight = odoRight
                self.radar_beacons_dist = [d1,d2,d3,d4]
                self.radar_beacons_azim = [a1,a2,a3,a4]
                self.magneticSensorCalibration = [magx0,magx1,magy0,magy1]
                self.status_log = status_all & 0x0F
                self.status_eval =      (status_all & 0x0F0) >> 4   
                self.robot_alive =      (status_all & 0x0F00) >> 8   
                self.status_save_log =  (status_all & 0x0F000) >> 12
                self.status_save_eval = (status_all & 0x0F0000) >> 16
                self.subs_ok = True
                #print ("status : %d : %d %d %d"%(status_all, self.status_log, self.status_eval, self.robot_alive))  
                #print ("Received pub message [", topic,simTime,distFront, "]")
                if t0 is None:
                    t0 = time.time()
                    st0 = simTime
                    self.simTime = simTime
                    self.cpuTime = t0
                else:
                    t1 = time.time()
                    dt = t1 - t0
                    dst = simTime - st0
                    t0 = t1
                    st0 = simTime
                    self.realTimeFact = dst/dt
                    self.simTimeLast = self.simTime
                    self.simTime = simTime
                    self.cpuTimeLast = self.cpuTime
                    self.cpuTime = t1
                    #print (self.simTimeLast,self.simTime,self.cpuTimeLast,self.cpuTime)
                    #print ("real time fact %.2f, cnt=%4d, st=%.6f"%(dst/dt,pub_cnt,simTime))
                    # print ("real time fact %.2f, cnt=%4d, st=%.6f, d=[%.2f,%.2f,%.2f,%.2f], mag=[%d,%d], line=[%.2f,%.2f,%.2f], odo=[%d,%d]"%(dst/dt,pub_cnt,simTime,
                    #         self.distFront, self.distLeft, self.distRight, self.distBack, self.magneticSensorX, self.magneticSensorY,
                    #         self.lineSensorLeft, self.lineSensorMiddle, self.lineSensorRight,self.encoderLeft,self.encoderRight))
            else:
                print ("Simulation is not started :")
                print ("   - stop python program (Ctrl-C)")
                print ("   - click on the start icon in CoppeliasSim")
                print ("   - start the python ptogram")
                self.subs_ok = False
            if self.robot_alive == 2:
                if not self.kill_sent:
                    print ("Robot is stalled , timeout on no new command , SIGINT signal (Ctrl-C) is sent")
                    self.kill_sent = True
                    self.bye()
                    self.simulation_alive = False
                    print ("main pid",os.getpid())
                    os.kill(os.getpid(), signal.SIGINT) # send Ctrl-C to stop on time out
                else:
                    print ("Robot is stalled , SIGINT signal (Ctrl-C) has already been sent")
            if dt_sleep>0.0:
                time.sleep(dt_sleep)
            if not self.simulation_alive:
                break 
        print ("bye from subscribe_sensors()")

    def get_sim_time (self):
        return self.simTime, self.simTimeLast, self.cpuTime, self.cpuTimeLast, self.realTimeFact

    def start_control_loop (self,dt_loop=0.05,eps_dt_loop=0.005,debug=False):
        self.simTimeLoopStart = self.simTime
        self.cpuTimeLoopStart = time.time()
        self.eps_dt_loop = eps_dt_loop
        self.dt_loop = dt_loop - self.eps_dt_loop
        self.simTimeLoopEnd = self.simTimeLoopStart + self.dt_loop
        if debug:
            print ("dt_loop",self.dt_loop)
            print ("start control loop at %.3f"%(self.simTimeLoopStart))
            print ("end control loop at %.3f"%(self.simTimeLoopEnd))
    
    def end_control_loop (self, warning=True, debug=False):
        if warning:
            if self.simTime > self.simTimeLoopEnd:
                print ("Warning : the control loop time exceeded %d ms"%(1000.0*(self.simTime-self.simTimeLoopEnd)))

        while True:
            simTime = self.simTime
            self.cpuTimeLoopEnd = time.time()
            if simTime >= self.simTimeLoopEnd:
                break
            time.sleep(0.001)
        if debug:
            print ("end (real) control loop at %.3f"%(simTime))
        self.simTimeLoopEnd = simTime
        self.simTimeLoopDuration = self.simTimeLoopEnd - self.simTimeLoopStart
        self.cpuTimeLoopDuration = self.cpuTimeLoopEnd - self.cpuTimeLoopStart
        return self.simTimeLoopEnd, self.cpuTimeLoopEnd, self.simTimeLoopDuration, self.cpuTimeLoopDuration

    def set_cmds (self):
        # publish on topic 1212
        pub_id = "1212".encode()
        fmt = "<4siii"
        cmd_log_eval = (self.do_log & 0x0F)
        cmd_log_eval += ((self.eval & 0x0F) << 4)
        cmd_log_eval += ((self.do_save_log & 0x0F) << 8)
        cmd_log_eval += ((self.do_save_eval & 0x0F) << 12)
        cmd_log_eval += ((self.ctrl_py & 0x0F) << 16)
        # print ("cmd_log_eval",hex(cmd_log_eval),self.do_save_log ,self.do_save_eval)
        bst = struct.pack(fmt,pub_id,self.speed_left,self.speed_right,cmd_log_eval)
        # print (bst)
        # print ("pub:",binascii.hexlify(bytearray(bst)))
        self.socket_pub_cmds.send (bst)  

    def set_log (self,do_log):
        """
           do_log = 0 : no log
           do_log = 1 : log
        """
        self.do_log = do_log
        # print ("set log",do_log)
        self.set_cmds ()

    def set_speed (self, spd_left, spd_right):
        """
        Robot motion is achieved by left and right wheel speeds.
        Speed is defined by a 11 bits PWM value. (instead of 10 bits in V03)
        The minimum value is 0 (wheel stopped)
        The maximum value is 2^11 - 1  = 2047 (wheel at max speed)

        Inputs :
           spd_left : PWM value in [0 2047]
           spd_right : PWM value in [0 2047]
        """
        # force integers
        self.speed_left = int(round(spd_left))
        self.speed_right = int(round(spd_right))
        self.set_cmds ()

    def stop (self):
        """
        Stop the robot by setting the speed of the motors to 0
        """
        self.set_speed(0,0)

    def get_sonar(self, name):
        """
        return the measurement of the selected sonar (ultrasonic sensor)
        if obstacle between 0.01 m to 1.5 m return the distance to 
        the nearest obstacle, otherwise return 0.0
        distance is returned in meters
        
        Inputs : 
           name : string that can be "front","left","right" or "back"
        Outputs :
           val : distance measured by the sonar 
        """
        val = 0.0
        if name == "front":
            val = self.distFront
        elif name == "left":
            val = self.distLeft           
        elif name == "right":
            val = self.distRight
        elif name == "back":
            val = self.distBack
        return val

    def get_multiple_sonars (self,names):
        """
        return the measurement of selected sonars (ultrasonic sensors)
        if obstacle between 0.01 m to 1.5 m return the distance to 
        the nearest obstacle, otherwise return 0.0
        distance are returned in meters

        Inputs :
           names : array of strings with sonar name 
                   sonar name can be "front","left","right", "back" or "all" 
                   if "all" the 4 distances are returned in the order "front","left","right", "back"
        Outputs :
           vals : array of floats, measured distance(s)
        """
        vals = []
        for name in names:
            if name == "front":
                vals.append(self.distFront)
            elif name == "left":
                vals.append(self.distLeft)           
            elif name == "right":
                vals.append(self.distRight)
            elif name == "back":
                vals.append(self.distBack)
            elif name == "all":
                vals.append(self.distFront)
                vals.append(self.distLeft)
                vals.append(self.distRight)
                vals.append(self.distBack)
        return vals
 

    def get_magnetic_sensor (self):
        """
        Get the horizontal components (x,y) of the magnetic sensor.
        With the assumption wa are working in the horizontal plane, we do not need the z component.
        A calibration process is required before computing the heading.
        Components are expressed as 32 bits integers.
        Note that we do not need the value in nano Tesla to compute the heading)

        Inputs:
            None
        Outputs:
            xmagneticSensorX : float : x component of the magnetic field
            ymagneticSensorY : float : y component of the magnetic field
        """
        return (self.magneticSensorX, self.magneticSensorY)

    def get_centerline_sensors(self):
        """
        Get the reflectivity values from the 3 infrared reflective sensors in front of the robot.
        The output is between 0.0 (black surface, no reflectivity) to 1.0 (white surface, full reflectivity)

        Inputs:
            None
        Outputs:
            lineSensorLeft : float : reflectivity from left sensor
            lineSensorMiddle : float : reflectivity from middle sensor
            lineSensorRight : float : reflectivity from right sensor                   
        """
        return self.lineSensorLeft, self.lineSensorMiddle, self.lineSensorRight

    def get_odometers (self):
        """
        return the values of ticks counters for the two wheels

        Inputs:
            None
        Outputs:
            encoderLeft : long int : number of ticks on left wheel odometer
            encoderRight : long int : number of ticks on right wheel odometer
        """
        return self.encoderLeft,self.encoderRight

    def get_radar_beacons(self):
        """
        return the values of the processed 4 radar beacons

        Inputs:
            None
        Outputs:
            distance[4] : float array with distances to the 4 beacons 
            encoderRight : float array with the azimuts o the 4 beacons 
        """        
        return self.radar_beacons_dist, self.radar_beacons_azim

    def get_magnetic_sensors_calibration_data (self):
        """
        return the values of the calibration process

        Inputs:
            None
        Outputs:
            mag_x_min : minimum magnetic sensor value along x
            mag_x_max : maximum magnetic sensor value along x
            mag_y_min : minimum magnetic sensor value along y
            mag_y_max : maximum magnetic sensor value along y
        """
        mag_x_min = self.magneticSensorCalibration[0]
        mag_x_max = self.magneticSensorCalibration[1]
        mag_y_min = self.magneticSensorCalibration[2]
        mag_y_max = self.magneticSensorCalibration[3]
        return mag_x_min, mag_x_max, mag_y_min, mag_y_max