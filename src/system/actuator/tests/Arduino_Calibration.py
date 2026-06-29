import serial
import time
from pynput import keyboard


class ServoController:

    def __init__(self, port="/dev/ttyUSB1", baud=115200):
        self.ser = serial.Serial(port, baud, timeout=1)
        time.sleep(2)

    def send(self, cmd):
        msg = cmd.strip() + "\r\n"
        self.ser.write(msg.encode("utf-8"))
        self.ser.flush()

    def set_mode(self, mode):
        self.send(f"MODE,{mode}")

    def set_center(self, a_deg, b_deg):
        self.send(f"CENTER,{a_deg},{b_deg}")

    def jog(self, servo_id, delta_deg):
        self.send(f"JOG,{servo_id},{delta_deg}")

    def experiment_cmd(self, a_deg, b_deg):
        self.send(f"CMD,{a_deg},{b_deg}")

    def save(self):
        self.send("SAVE")
       
    def next_cal(self):
        self.send("T")

    def prev_cal(self):
        self.send("R")
       


ctrl = ServoController("/dev/ttyUSB0")

ctrl.set_mode("CAL")

STEP = 5.0


def on_press(key):

    try:

        if key.char == 'q':
            print("servo1 -1")
            ctrl.jog(1, -STEP)

        elif key.char == 'e':
            print("servo1 +1")
            ctrl.jog(1, STEP)

        elif key.char == 'a':
            print("servo2 -1")
            ctrl.jog(2, -STEP)

        elif key.char == 'd':
            print("servo2 +1")
            ctrl.jog(2, STEP)

        elif key.char == 's':
            print("Saving calibration")
            ctrl.save()

        elif key.char == 'r':
            print("previous cal point")
            ctrl.prev_cal()

        elif key.char == 't':
            print("next cal point")
            ctrl.next_cal()

    except AttributeError:
        pass


listener = keyboard.Listener(on_press=on_press)
listener.start()

print("Calibration controls:")
print("Q/E servo1")
print("A/D servo2")
print("R/T previous/next cal point")
print("S save calibration")

# Angles (deg): [155.04013922  65.97390473]
# for center 110, 110

while True:
    time.sleep(1)
