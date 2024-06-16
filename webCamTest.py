import cv2 as cv
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


cv.namedWindow("CamWindow", cv.WINDOW_NORMAL)
capture = cv.VideoCapture(0)
if not capture.isOpened():
    capture = cv.VideoCapture(1)

while (k := cv.waitKey(1)) != ord('e'):
    returned, frame = capture.read()
    if returned:
        cv.imshow("CamWindow", frame)
    k = cv.waitKey(1)
    
capture.release()
cv.destroyAllWindows()