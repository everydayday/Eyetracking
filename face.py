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



            pixel474 = mpDraw._normalized_to_pixel_coordinates(joint[474,0], joint[474,1], imgWidth, imgHeight)
            pixel475 = mpDraw._normalized_to_pixel_coordinates(joint[475,0], joint[475,1], imgWidth, imgHeight)
            pixel476 = mpDraw._normalized_to_pixel_coordinates(joint[476,0], joint[476,1], imgWidth, imgHeight)
            pixel477 = mpDraw._normalized_to_pixel_coordinates(joint[477,0], joint[477,1], imgWidth, imgHeight)
            pixel469 = mpDraw._normalized_to_pixel_coordinates(joint[469,0], joint[469,1], imgWidth, imgHeight)
            pixel470 = mpDraw._normalized_to_pixel_coordinates(joint[470,0], joint[470,1], imgWidth, imgHeight)
            pixel471 = mpDraw._normalized_to_pixel_coordinates(joint[471,0], joint[471,1], imgWidth, imgHeight)
            pixel472 = mpDraw._normalized_to_pixel_coordinates(joint[472,0], joint[472,1], imgWidth, imgHeight)
            
            cv2.circle(img, pixel474, 3, (0, 255, 0), -1)
            cv2.circle(img, pixel475, 3, (0, 255, 0), -1)
            cv2.circle(img, pixel476, 3, (0, 255, 0), -1)
            cv2.circle(img, pixel477, 3, (0, 255, 0), -1)
            cv2.circle(img, pixel469, 3, (0, 255, 0), -1)
            cv2.circle(img, pixel470, 3, (0, 255, 0), -1)
            cv2.circle(img, pixel471, 3, (0, 255, 0), -1)
            cv2.circle(img, pixel472, 3, (0, 255, 0), -1)
            #cv2.circle(img, pixel473, 2, (0, 255, 0), -1)


    cv2.imshow("Image", img)
    if cv2.waitKey(1) == ord('q'):
        break
