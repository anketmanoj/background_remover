import cv2
import cvzone
from cvzone.SelfiSegmentationModule import SelfiSegmentation
import os

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
cap.set(cv2.CAP_PROP_FPS, 60)
segmentor = SelfiSegmentation(0)
fpsReader = cvzone.FPS()


while True:
    ret, img = cap.read()
    imgOut = segmentor.removeBG(img, (0, 255, 0), threshold=0.4)

    imageStacked = cvzone.stackImages([img, imgOut], 2, 1)
    _, imageStacked = fpsReader.update(imageStacked, color=(0, 0, 255))

    cv2.imshow('image stack', imageStacked)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
