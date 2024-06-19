import cv2 as cv

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