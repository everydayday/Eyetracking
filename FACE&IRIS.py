import cv2
import mediapipe as mp
import numpy as np
import time

cap = cv2.VideoCapture(0)
pTime = 0

mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=2)
drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=2)

imgWidth = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
imgHeight = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

while cap.isOpened():
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceMesh.process(imgRGB)


    if results.multi_face_landmarks is not None:
        for faceLms in results.multi_face_landmarks:
            joint = np.zeros((500, 3))
            for j,lm in enumerate(faceLms.landmark):
                joint[j] = [lm.x, lm.y, lm.z]
