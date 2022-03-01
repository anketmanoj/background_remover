import cv2
import cvzone
from cvzone.SelfiSegmentationModule import SelfiSegmentation
import os
import numpy as np

# url = "https://192.168.0.178:8080/video"
cap = cv2.VideoCapture("./vitamins.mp4")

cap.set(3, 640)
cap.set(4, 480)


fpsReader = cvzone.FPS()


while True:
    ret, img = cap.read()
    # threshold on white
    # Define lower and uppper limits
    lower = np.array([100, 170, 180])
    upper = np.array([255, 255, 255])

    # imgOut = segmentor.removeBG(img, (0, 255, 0), threshold=0.8)
    # Create mask to only select black
    thresh = cv2.inRange(img, lower, upper)

    # apply morphology
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # invert morp image
    mask = 255 - morph

    # apply mask to image
    result = cv2.bitwise_and(img, img, mask=mask)

    # imageStacked = cvzone.stackImages([img, imgOut], 2, 1)
    # _, imageStacked = fpsReader.update(imageStacked, color=(0, 0, 255))

    cv2.imshow('image result', result)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
