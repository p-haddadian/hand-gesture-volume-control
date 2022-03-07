import numpy as np
import cv2
import time
import hand_tracking as ht
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# volume changer initialization
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
# volume.GetMute()
# volume.GetMasterVolumeLevel()

# Parameters:
width_cam, height_cam = 640, 480
volume_range = volume.GetVolumeRange()
min_volume = volume_range[0]
max_volume = volume_range[1]

cap = cv2.VideoCapture(1)
cap.set(3, width_cam)
cap.set(4, height_cam)
p_time = 0

detector = ht.HandDetector(detection_confidence= 0.7)

while True:
    success, img = cap.read()
    img = detector.find_hands(img)
    lm_list = detector.find_position(img)
    if len(lm_list) != 0:
        # print(lm_list[4], lm_list[8])

        x1, y1 = lm_list[4][1], lm_list[4][2]
        x2, y2 = lm_list[8][1], lm_list[8][2]
        cx, cy = (x2 + x1) // 2, (y2 + y1) // 2

        cv2.circle(img, (x1, y1), 10, (255, 255, 0), cv2.FILLED)
        cv2.circle(img, (x2, y2), 10, (255, 255, 0), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), (255, 255, 0), 2)
        cv2.circle(img, (cx, cy), 10, (255, 255, 0), cv2.FILLED)

        # euclidean distance of the two fingers
        length = math.hypot(x2 - x1, y2 - y1)

        # range of the fingers: 30 --> 300
        # volume range -75 --> 0.0
        vol = np.interp(length, [30, 300], [min_volume, max_volume])
        volume.SetMasterVolumeLevel(vol, None)

        if length < 30:
            cv2.circle(img, (cx, cy), 10, (0, 255, 0), cv2.FILLED)

    c_time = time.time()
    fps = 1 / (c_time - p_time)
    p_time = c_time

    cv2.putText(img, 'FPS: {}'.format(int(fps)), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Volume Hand Control', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):   # press button 'q' for exit
        break