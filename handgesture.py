import cv2
import mediapipe as mp
import numpy as np
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# Initialize MediaPipe Hand model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1)

# Initialize PyCaw for system audio control
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
vol_range = volume.GetVolumeRange()
min_vol, max_vol = vol_range[0], vol_range[1]

# Start video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip and convert image
    frame = cv2.flip(frame, 1)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    lmList = []
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            for id, lm in enumerate(hand_landmarks.landmark):
                h, w, _ = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append((id, cx, cy))

        if lmList:
            # Get positions of thumb tip (id 4) and index finger tip (id 8)
            x1, y1 = lmList[4][1], lmList[4][2]
            x2, y2 = lmList[8][1], lmList[8][2]

            # Draw line and circles
            cv2.circle(frame, (x1, y1), 8, (255, 0, 0), -1)
            cv2.circle(frame, (x2, y2), 8, (255, 0, 0), -1)
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Calculate distance between fingers
            length = math.hypot(x2 - x1, y2 - y1)

            # Map the length range to volume range
            vol = np.interp(length, [30, 200], [min_vol, max_vol])
            volume.SetMasterVolumeLevel(vol, None)

            # Display volume bar
            vol_bar = np.interp(length, [30, 200], [400, 150])
            cv2.rectangle(frame, (50, 150), (85, 400), (0, 255, 0), 2)
            cv2.rectangle(frame, (50, int(vol_bar)), (85, 400), (0, 255, 0), -1)

            # Display volume percentage
            vol_percent = np.interp(length, [30, 200], [0, 100])
            cv2.putText(frame, f'{int(vol_percent)} %', (40, 430), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 255, 0), 3)

    cv2.imshow("Hand Volume Control", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
