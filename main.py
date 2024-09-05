import time
import subprocess
import numpy as np
import cv2 as cv
import mediapipe as mp
from mediapipe.tasks import python as tasks
from mediapipe.tasks.python import vision as visionTasks

modelPath = "./hand_landmarker.task"

detectionResult = None
rightHandThumbPoses = [(0,0,0)]
rightHandIndexFingPoses = [(0.9,0.9,0.9)]
rightHandThumbLastPoses = [(0,0,0)]
lastPosCheckTime = time.time()
rightHandThumbDXs = [0]

def GetBGR(hue : int, saturation : float, value : float):
    c = value * saturation
    x = c * (1 - abs(((hue/60) % 2) - 1))
    m = value - c
    if hue < 0:
        return 0
    if hue < 60:
        return ((m+0)*255,(m+x)*255,(m+c)*255)
    if hue < 120:
        return ((m+0)*255,(m+c)*255,(m+x)*255)
    if hue < 180:
        return ((m+x)*255,(m+c)*255,(m+0)*255)
    if hue < 240:
        return ((m+c)*255,(m+x)*255,(m+0)*255)
    if hue < 300:
        return ((m+c)*255,(m+0)*255,(m+x)*255)
    if hue < 360:
        return ((m+x)*255,(m+0)*255,(m+c)*255)
    return 0


def AnnotateFrame(frame, frameResult: visionTasks.HandLandmarkerResult):
    if frameResult == None:
        return frame
    listHandedness = frameResult.handedness
    listHandLandmarks = frameResult.hand_landmarks
    annotatedFrame = np.copy(frame)
    rightHandIndexes = []
    
    for landmarks in listHandLandmarks:
        #for landmark in landmarks:
        for i in range(length := len(landmarks)):
            cv.circle(annotatedFrame, (int(landmarks[i].x * annotatedFrame.shape[1]), int(landmarks[i].y * annotatedFrame.shape[0])), (r := int((0.3 - landmarks[i].z)*10)), GetBGR((360//length) * i, 0.6, 1), 2 * r)    

    annotatedFrame = cv.flip(annotatedFrame, 1)
    
    for handedness in listHandedness:
        for categories in handedness:
            cv.putText(annotatedFrame, (displayName := categories.display_name), (50 if displayName == "Left" else annotatedFrame.shape[1] - 200, 50), cv.FONT_HERSHEY_SIMPLEX, 1, 1, 1)
            if categories.display_name == "Right":
                rightHandIndexes.append(categories.index)
    
    global rightHandThumbLastPoses
    global lastPosCheckTime
    global rightHandThumbPoses
    global rightHandIndexFingPoses
    global rightHandThumbDXs
    
    if len(rightHandIndexes) > 0:
        if (deltaTime := time.time() - lastPosCheckTime) > 0.05:
            for rightHandIndex in rightHandIndexes:
                rightHandThumbPoses[rightHandIndex] = (listHandLandmarks[rightHandIndex][4].x, listHandLandmarks[rightHandIndex][4].y, listHandLandmarks[rightHandIndex][4].z)
                rightHandIndexFingPoses[rightHandIndex] = (listHandLandmarks[rightHandIndex][8].x, listHandLandmarks[rightHandIndex][8].y, listHandLandmarks[rightHandIndex][8].z)

                rightHandThumbDXs[rightHandIndex] = rightHandThumbPoses[rightHandIndex][0] - rightHandThumbLastPoses[rightHandIndex][0]
                
                rightHandThumbLastPoses[rightHandIndex] = rightHandThumbPoses[rightHandIndex]
                if abs(rightHandIndexFingPoses[rightHandIndex][0] - rightHandThumbPoses[rightHandIndex][0]) < 0.1 and abs(rightHandIndexFingPoses[rightHandIndex][1] - rightHandThumbPoses[rightHandIndex][1]) < 0.1 and abs(rightHandIndexFingPoses[rightHandIndex][2] - rightHandThumbPoses[rightHandIndex][2]) < 0.1:
                    if rightHandThumbDXs[rightHandIndex] > 0.1:
                        #cv.putText(annotatedFrame, "Workspace up", (50, 400), cv.FONT_HERSHEY_SIMPLEX, 1, 1, 2)
                        print("Workspace up")
                        subprocess.call(['hyprctl', 'dispatch', 'workspace', 'r+1'])
                    elif rightHandThumbDXs[rightHandIndex] < -0.1:
                        #cv.putText(annotatedFrame, "Workspace down", (50, 400), cv.FONT_HERSHEY_SIMPLEX, 1, 1, 2)
                        print("Workspace down")
                        subprocess.call(['hyprctl', 'dispatch', 'workspace', 'r-1'])
                
            lastPosCheckTime = time.time()
            

        #cv.putText(annotatedFrame, str(rightHandThumbDX), (50, 400), cv.FONT_HERSHEY_SIMPLEX, 1, 1, 2)
        #cv.putText(annotatedFrame, str(rightHandThumbPos), (50, 300), cv.FONT_HERSHEY_SIMPLEX, 1, 1, 2)
    
    return annotatedFrame
    

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
    cv.namedWindow("cvMain", cv.WINDOW_NORMAL)
    capture = cv.VideoCapture(1)
    if not capture.isOpened():
        cpature = cv.VideoCapture(0)
    while (k := cv.waitKey(1)) != ord('e'):
        returned, frame = capture.read()
        if returned:
            mpImage = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            landmarker.detect_async(mpImage, int(time.time()*1000))
            cv.imshow("cvMain", AnnotateFrame(frame, detectionResult))
        k = cv.waitKey(1)
    capture.release()
cv.destroyAllWindows()