import cv2 as cv
import mediapipe as mp
import sys
#import numpy.core.multiarray

img = cv.imread(cv.samples.findFile("/home/meteoroetem/Pictures/ענחנו ארבים.JPG"))
if img is None:
    sys.exit("couldn't find the picture")

cv.namedWindow("openCV-Window", cv.WINDOW_NORMAL)
cv.imshow("openCV-Window", img)

k = cv.waitKey(0)

if k == ord('e'):
    sys.exit()