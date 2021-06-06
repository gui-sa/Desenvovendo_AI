
import os
import numpy as np
import cv2

cap = cv2.VideoCapture(0)

fgbg = cv2.createBackgroundSubtractorMOG2(history = 10, varThreshold = 100, detectShadows = False)

while(1):
    ret, frame = cap.read()

    fgmask = fgbg.apply(frame)
    #fgmask = np.multiply(fgmask,frame)
    cv2.imshow('frame',fgmask)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()