import cv2 as cv

cv.namedWindow("OpenCV-main", cv.WINDOW_NORMAL)
capture = cv.VideoCapture(0)
if not capture.isOpened():
    capture = cv.VideoCapture(1)

frame = None
returned = False
while not returned:
    returned, frame = capture.read()
capture.release()
    
cv.circle(frame, (50, 50), 3, 0, 6)
cv.circle(frame, (0, 0), 3, 0, 6)
cv.circle(frame, (100, 100), 3, 0, 6)
cv.imshow("OpenCV-main", frame)
cv.waitKey(0)