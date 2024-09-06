import time
import subprocess
import mediapipe as mp
from mediapipe.tasks import python as tasks
from mediapipe.tasks.python.components.containers import NormalizedLandmark
from mediapipe.tasks.python import vision as visionTasks
from pymouse import PyMouse

def HandLandmarksHandle(handLandmarks : list[NormalizedLandmark], lastHandLandmarks : list[NormalizedLandmark]): # type: ignore
    thumbPos = handLandmarks[4]
    lastThumbPos = lastHandLandmarks[4]
    thumbPosDX = thumbPos.x - lastThumbPos.x
    indexFingPos = handLandmarks[8]
    if PosesTouch(thumbPos, indexFingPos):
        print(f"Thumb and finger touching!\nThumb Delta X: {thumbPosDX}")
        if thumbPosDX > 0.1:
            print("Workspace up")
            subprocess.call(['hyprctl', 'dispatch', 'workspace', 'r+1'])
        elif thumbPosDX < -0.1:
            print("Workspace down")
            subprocess.call(['hyprctl', 'dispatch', 'workspace', 'r-1'])
    ringFingPos = handLandmarks[16]
    lastRingFingPos = lastHandLandmarks[16]
    m = PyMouse()
    if PosesTouch(thumbPos, ringFingPos) and not PosesTouch(thumbPos, lastRingFingPos):
        print("Click!")
        m.click((mPos := m.position())[0], mPos[1])

def PosesTouch(pos1 : NormalizedLandmark, pos2 : NormalizedLandmark): # type: ignore
    return abs(pos1.x - pos2.x) < 0.1 and abs(pos1.y - pos2.y) < 0.1