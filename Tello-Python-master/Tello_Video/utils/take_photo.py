import tello
import cv2
from tello_control_ui import TelloUI
import time
import math
import numpy as np
drone = tello.Tello('', 8889)

def main():
    global drone
    time.sleep(5)
    print(drone.get_battery())
    i = 0
    while(1):
        frame = drone.read() 
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imshow("frame",frame)
        key = cv2.waitKey(1) & 0xFF
        print(key)
        if key == ord('q'):
            pic = 'img/opencv_frame_' + str(i) + '.jpg'
            i += 1
            cv2.imwrite(pic, frame)



if __name__ == "__main__":
    main()
