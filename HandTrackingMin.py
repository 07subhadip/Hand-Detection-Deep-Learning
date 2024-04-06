import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

while True:
    success , img = cap.read()
    cv2.imshow("Image",img)
    cv2.waitKey(1)
# this is to access the camera of device or any cam module like webcam
