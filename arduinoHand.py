import time
import serial
import numpy as np
import cv2 as cv
import mediapipe as mp
from mediapipe.tasks import python as tasks
from mediapipe.tasks.python import vision as visionTasks

modelPath = "./hand_landmarker.task"

detectionResult = None


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
    
    for landmarks in listHandLandmarks:
        #for landmark in landmarks:
        for i in range(length := len(landmarks)):
            cv.circle(annotatedFrame, (int(landmarks[i].x * annotatedFrame.shape[1]), int(landmarks[i].y * annotatedFrame.shape[0])), (r := int((0.3 - landmarks[i].z)*10)), GetBGR((360//length) * i, 0.6, 1), 2 * r)
    
    annotatedFrame = cv.flip(annotatedFrame, 1)
    
    for handedness in listHandedness:
        for categories in handedness:
            cv.putText(annotatedFrame, (displayName := categories.display_name), (50 if displayName == "Left" else annotatedFrame.shape[1] - 200, 50), cv.FONT_HERSHEY_SIMPLEX, 1, 1, 1)
    
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
    cv.namedWindow("Main", cv.WINDOW_NORMAL)
    capture = cv.VideoCapture(1)
    if not capture.isOpened():
        cpature = cv.VideoCapture(0)
    while (k := cv.waitKey(1)) != ord('e'):
        returned, frame = capture.read()
        if returned:
            mpImage = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            landmarker.detect_async(mpImage, int(time.time()*1000))
            cv.imshow("Main", AnnotateFrame(frame, detectionResult))
        k = cv.waitKey(1)
    capture.release()
cv.destroyAllWindows()