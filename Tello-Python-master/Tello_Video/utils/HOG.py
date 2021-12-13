import numpy as np
import time
import cv2
import util
def HOG(frame):
    # HOG
    cameraMatrix, distCoeffs =  util.camera_matrix("C:\\Users\\Chieh-Ming Jiang\\Desktop\\VVS_final\\Tello-Python-master\\Tello_Video\\utils\\tello.txt")
    objp, imgp = util.create_points(0.5, 1.65)
    hog = cv2.HOGDescriptor() 
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    rects, weights = hog.detectMultiScale(
        frame, winStride = (4, 4), scale = 1.05, useMeanshiftGrouping= False, padding= (8,8))
    for i in range(len(rects)):
        (x, y, w, h) = rects[i]
        if h < 400:
            continue
        imgp[0] = [x, y]
        imgp[1] = [x+w, y]
        imgp[2] = [x, y+h]
        imgp[3] = [x+w, y+h]
        cv2.rectangle(frame, tuple(imgp[0]) , tuple(imgp[3]),  (0, 255, 255), 2)
    retval, rvec, tvec = cv2.solvePnP(objp, imgp, cameraMatrix, distCoeffs)
    if retval:
        cv2.putText(frame, str(tvec[2]) , (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA )
    return frame
