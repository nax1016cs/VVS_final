import tello
import sys
sys.path.insert(1, 'utils/')
import util
from HOG import estimate_distance
from HOG import HOG
import time
import cv2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import serial
from yolo import YOLO

social_dis = 1.5

def detect_and_predict_mask(frame, faceNet, maskNet):
    # grab the dimensions of the frame and then construct a blob
    # from it
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
        (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the face detections
    faceNet.setInput(blob)
    detections = faceNet.forward()

    # initialize our list of faces, their corresponding locations,
    # and the list of predictions from our face mask network
    faces = []
    locs = []
    preds = []

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with
        # the detection
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the confidence is
        # greater than the minimum confidence
        if confidence > 0.5:
            # compute the (x, y)-coordinates of the bounding box for
            # the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # ensure the bounding boxes fall within the dimensions of
            # the frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            # extract the face ROI, convert it from BGR to RGB channel
            # ordering, resize it to 224x224, and preprocess it
            face = frame[startY:endY, startX:endX]
            if face.any():
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face = cv2.resize(face, (224, 224))
                face = img_to_array(face)
                face = preprocess_input(face)

                # add the face and bounding boxes to their respective
                # lists
                faces.append(face)
                locs.append((startX, startY, endX, endY))

    # only make a predictions if at least one face was detected
    if len(faces) > 0:
        # for faster inference we'll make batch predictions on *all*
        # faces at the same time rather than one-by-one predictions
        # in the above `for` loop
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32)

    # return a 2-tuple of the face locations and their corresponding
    # locations
    return (locs, preds)

def main():
    COM_PORT = 'COM3'
    BAUD_RATES = 9600
    ser = serial.Serial(COM_PORT, BAUD_RATES)
    print("[INFO] loading face detector model...")
    prototxtPath = "C:\\Users\\Chieh-Ming Jiang\\Desktop\\VVS_final\\Face-Mask-Detection\\face_detector\\deploy.prototxt"
    weightsPath = "C:\\Users\\Chieh-Ming Jiang\\Desktop\\VVS_final\\Face-Mask-Detection\\face_detector\\res10_300x300_ssd_iter_140000.caffemodel"
    faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
    # load the face mask detector model from disk
    print("[INFO] loading face mask detector model...")
    mask_path = "C:\\Users\\Chieh-Ming Jiang\\Desktop\\VVS_final\\Face-Mask-Detection\\mask_detector.model"
    maskNet = load_model(mask_path)
    yolo = YOLO("models/cross-hands.cfg", "models/cross-hands.weights", ["hand"])
    drone = tello.Tello('', 8889)  
    time.sleep(5)
    takeoff = False
    ser.write(b'2\n')
    ser.write(b'4\n')
    while (True):
        detectMask = False
        detectHand = False
        distance = -1
        frame = drone.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        # Hands
        width, height, inference_time, results = yolo.inference(frame)
        # sort by confidence
        results.sort(key=lambda x: x[2])
        # how many hands should be shown
        hand_count = len(results)

        # display hands
        for detection in results[:hand_count]:
            id, name, confidence, x, y, w, h = detection
            if confidence > 0.6:
                detectHand = True
            print(id, name)
            cx = x + (w / 2)
            cy = y + (h / 2)
            # draw a bounding box rectangle and label on the image
            color = (0, 255, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            text = "%s (%s)" % (name, round(confidence, 2))
            cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 2)
        # Mask
        (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

        for (box, pred) in zip(locs, preds):
            # unpack the bounding box and predictions
            (startX, startY, endX, endY) = box
            distance = estimate_distance(startX, startY, endX, endY)
            (mask, withoutMask) = pred
            label = "Mask" if mask > withoutMask else "No Mask"
            if label == "Mask" :
                detectMask = True
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
            cv2.putText(frame, label, (startX, startY - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
            cv2.putText(frame, str(distance[0]), (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255 , 255), 2)
        key = cv2.waitKey(1)
        if key != -1:
            if key & 0xFF == ord('1'):
                takeoff = True
            drone.keyboard(key)
        cv2.imshow("frame",frame) 
        if key & 0xFF == ord('q'):
            break
        if distance > social_dis and detectHand and detectMask:
            print("Flipping")
            drone.flip('r')
            ser.write(b'2\n')
            ser.write(b'4\n')
            time.sleep(3)
            drone.move_left(0.3)
        elif distance > social_dis and detectHand and not detectMask:
            print("no mask greeting!!!")
            drone.rotate_cw(30)
            time.sleep(5)
            drone.rotate_ccw(30)
            time.sleep(5)
        elif distance < social_dis and not detectMask and distance > 0:
            ser.write(b'1\n')
            ser.write(b'3\n')
            print("Cannot get in!!!")
        elif distance < social_dis and detectMask:
            print("Please get in!!!")
            ser.write(b'2\n')
            ser.write(b'4\n')
            drone.move_right(0.8)
            time.sleep(3)
            drone.move_left(0.8)
            time.sleep(3)
        else: 
            ser.write(b'2\n')
            ser.write(b'4\n')

if __name__ == "__main__":
    main()
