import tello
import sys
sys.path.insert(1, 'utils/')
import util
import HOG
import time
import cv2
def main():
    drone = tello.Tello('', 8889)  
    time.sleep(5)
    while (True):
        frame = drone.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        try: 
            frame = HOG.HOG(frame)
        except:
            pass
        cv2.imshow("frame",frame) 
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == "__main__":
    main()
