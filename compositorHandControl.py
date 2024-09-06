import time
import subprocess
import numpy as np
import cv2 as cv
import mediapipe as mp
from mediapipe.tasks import python as tasks
from mediapipe.tasks.python.components.containers import NormalizedLandmark
from mediapipe.tasks.python import vision as visionTasks

from handTrackingHandlers import HandLandmarksHandle

modelPath = "./hand_landmarker.task"

detectionResult = None
lastPosCheckTime = time.time()
lastListHandLandmarks = [[NormalizedLandmark(0,0,0) for i in range(21)]]

def FrameResultHandle(frameResult: visionTasks.HandLandmarkerResult): # type: ignore
    if frameResult == None:
        return
    listHandedness = frameResult.handedness
    listHandLandmarks = frameResult.hand_landmarks
    rightHandsIndexes = []
    
    for handedness in listHandedness:
        rightHandsIndexes = [categories.index for categories in handedness if categories.display_name == "Right"]
    
    global lastPosCheckTime
    global lastListHandLandmarks
    
    if len(rightHandsIndexes) > 0:
        if (deltaTime := time.time() - lastPosCheckTime) > 0.05:
            for rightHandIndex in rightHandsIndexes:
                HandLandmarksHandle(listHandLandmarks[rightHandIndex], lastListHandLandmarks[rightHandIndex])
            lastListHandLandmarks = listHandLandmarks
            lastPosCheckTime = time.time()
    return 


def ResultCallback(result: visionTasks.HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int): # type: ignore
    global detectionResult
    #print(f"Hand landmarker callback: {result}\nTimestamp: {timestamp_ms}\n\n")
    detectionResult = result


options = visionTasks.HandLandmarkerOptions (
    base_options=tasks.BaseOptions(model_asset_path=modelPath),
    running_mode=visionTasks.RunningMode.LIVE_STREAM,
    num_hands=2,
    result_callback=ResultCallback
)
with visionTasks.HandLandmarker.create_from_options(options) as landmarker:
    capture = cv.VideoCapture(1)
    if not capture.isOpened():
        cpature = cv.VideoCapture(0)
    while (k := cv.waitKey(1)) != ord('e'):
        returned, frame = capture.read()
        if returned:
            mpImage = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            landmarker.detect_async(mpImage, int(time.time()*1000))
            FrameResultHandle(detectionResult)
        k = cv.waitKey(1)
    capture.release()
cv.destroyAllWindows()