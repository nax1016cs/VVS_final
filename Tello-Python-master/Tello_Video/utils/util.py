import numpy as np
import cv2

def create_points(width, height):
    objp = np.zeros((2*2,3), np.float32)
    objp[:,:2] = np.mgrid[0:2,0:2].T.reshape(-1,2)
    imgp = np.zeros((2*2,2), np.float32)
    imgp[:,:2] = np.mgrid[0:2,0:2].T.reshape(-1,2)
    objp[1][0] = objp[3][0] = width
    objp[2][1] = objp[3][1] = height
    return objp, imgp

def camera_matrix(path):
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    cameraMatrix = fs.getNode("intrinsic")
    distCoeffs = fs.getNode("distortion")
    cameraMatrix = cameraMatrix.mat()
    distCoeffs = distCoeffs.mat()
    return cameraMatrix, distCoeffs

