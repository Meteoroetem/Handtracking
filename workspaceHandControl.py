import time
import subprocess
import numpy as np
import cv2 as cv
import mediapipe as mp
from mediapipe.tasks import python as tasks
from mediapipe.tasks.python import vision as visionTasks

modelPath = "./hand_landmarker.task"

detectionResult = None
rightHandThumbPoses = [(0,0)]
rightHandIndexFingPoses = [(0.9,0.9)]
rightHandThumbLastPoses = [(0,0)]
lastPosCheckTime = time.time()
rightHandThumbDXs = [0]


def HandlandmarkerHandle(frameResult: visionTasks.HandLandmarkerResult):
    if frameResult == None:
        return
    listHandedness = frameResult.handedness
    listHandLandmarks = frameResult.hand_landmarks
    rightHandsIndexes = []
    
    for handedness in listHandedness:
        rightHandsIndexes = [categories.index for categories in handedness if categories.display_name == "Right"]
    
    global rightHandThumbLastPoses
    global lastPosCheckTime
    global rightHandThumbPoses
    global rightHandIndexFingPoses
    global rightHandThumbDXs
    
    if len(rightHandsIndexes) > 0:
        if (deltaTime := time.time() - lastPosCheckTime) > 0.05:
            for rightHandIndex in rightHandsIndexes:
                rightHandThumbPoses[rightHandIndex] = (listHandLandmarks[rightHandIndex][4].x, listHandLandmarks[rightHandIndex][4].y)
                rightHandIndexFingPoses[rightHandIndex] = (listHandLandmarks[rightHandIndex][8].x, listHandLandmarks[rightHandIndex][8].y)

                rightHandThumbDXs[rightHandIndex] = rightHandThumbPoses[rightHandIndex][0] - rightHandThumbLastPoses[rightHandIndex][0]
                
                rightHandThumbLastPoses[rightHandIndex] = rightHandThumbPoses[rightHandIndex]
                if abs(rightHandIndexFingPoses[rightHandIndex][0] - rightHandThumbPoses[rightHandIndex][0]) < 0.1 and abs(rightHandIndexFingPoses[rightHandIndex][1] - rightHandThumbPoses[rightHandIndex][1]) < 0.1:
                    if rightHandThumbDXs[rightHandIndex] > 0.1:
                        print("Workspace up")
                        subprocess.call(['hyprctl', 'dispatch', 'workspace', 'r+1'])
                    elif rightHandThumbDXs[rightHandIndex] < -0.1:
                        print("Workspace down")
                        subprocess.call(['hyprctl', 'dispatch', 'workspace', 'r-1'])
                                    
            lastPosCheckTime = time.time()
    return


def ResultCallback(result: visionTasks.HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
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
    #cv.namedWindow("cvMain", cv.WINDOW_NORMAL)
    capture = cv.VideoCapture(1)
    if not capture.isOpened():
        cpature = cv.VideoCapture(0)
    while (k := cv.waitKey(1)) != ord('e'):
        returned, frame = capture.read()
        if returned:
            mpImage = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            landmarker.detect_async(mpImage, int(time.time()*1000))
            #cv.imshow("cvMain", AnnotateFrame(frame, detectionResult))
            HandlandmarkerHandle(detectionResult)
        k = cv.waitKey(1)
    capture.release()
cv.destroyAllWindows()