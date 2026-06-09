import time

class Filter ():
    def __init__(self):  # set some constants you may find useful
        self.distBetweenWheels = 0.12
        self.nTicksPerRevol = 256
        self.wheelDiameter = 0.06
        self.nomsp = 900
        self.previous_error = 0
        self.P = 0
        self.I = 0
        self.D = 0

        self.a = 1.0  # here you define some useful variables
        pass
    
    def doNothingFilter(self,v):
        """
        The filter does nothing and returns the input value
        This is just for testing purpose
        """
        vf = self.a*v
        return vf

    def PID(self, rb, error):
        Kp = 100
        Kd = 400
        Ki = 0
        self.P = error
        self.I += error
        self.D = error - self.previous_error
        PID = Kp * self.P + Ki * self.I + Kd * self.D
        self.previous_error = error
        lsp = self.nomsp - PID
        rsp = self.nomsp + PID
        if lsp > 1023: lsp = 1023
        if lsp < 0: lsp = 0
        if rsp > 1023: rsp = 1023
        if rsp < 0: rsp = 0
        rb.set_speed(lsp, rsp)

