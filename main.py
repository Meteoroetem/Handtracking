import time
import numpy as np
import cv2 as cv
import mediapipe as mp
from mediapipe.tasks import python as tasks
from mediapipe.tasks.python import vision as visionTasks

modelPath = "./hand_landmarker.task"

detectionResult = None

def AnnotateFrame(frame, frameResult: visionTasks.HandLandmarkerResult):
    if frameResult == None:
        return frame
    listHandednss = frameResult.handedness
    listHandLandmarks = frameResult.hand_landmarks
    annotatedFrame = np.copy(frame)
    
    for landmarkList in listHandLandmarks:
        for landmark in landmarkList:
            cv.circle(annotatedFrame, (int(landmark.x * annotatedFrame.shape[1]), int(landmark.y * annotatedFrame.shape[0])), (r := int((0.3 - landmark.z)*10)), 0, 2 * r)
    
    annotatedFrame = cv.flip(annotatedFrame, 1)
    
    if len(listHandednss) > 0:
        cv.putText(annotatedFrame, (displayName := listHandednss[0][0].display_name), (50 if displayName == "Left" else annotatedFrame.shape[1] - 200, 50), cv.FONT_HERSHEY_SIMPLEX, 1, 1, 1)
    
    return annotatedFrame
    

def ResultCallback(result: visionTasks.HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global detectionResult
    #print(f"Hand landmarker callback: {result}\nTimestamp: {timestamp_ms}\n\n")
    detectionResult = result


options = visionTasks.HandLandmarkerOptions (
    base_options=tasks.BaseOptions(model_asset_path=modelPath),
    running_mode=visionTasks.RunningMode.LIVE_STREAM,
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