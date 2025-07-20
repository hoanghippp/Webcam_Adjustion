import cv2
import mediapipe as mp
import time
from datetime import datetime, timedelta
import os
import numpy as np
from filter import *

output_dir = 'assets/captures'
os.makedirs(output_dir, exist_ok=True)
TIME_DISPLAY_SAVE_IMAGE = 3 # second
test_display = False
text_start_time = None
message = ''
filter_list = ["none", "warm_orange", "cool_blue", "white_pink", "vintage"]
filter_index = 0
current_filter = filter_list[filter_index]

mp_drawing = mp.solutions.drawing_utils
mp_drawing_style = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

# Khởi tạo FaceMesh một lần
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=2,
    refine_landmarks=True,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

prev_time = 0
stream = cv2.VideoCapture(0)

def mesh_face(frame, face_landmarks):
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)

    # Lấy toàn bộ các điểm landmark trên mặt
    points = [(int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0]))
              for lm in face_landmarks.landmark]

    # Tạo đường bao quanh khuôn mặt
    hull = cv2.convexHull(np.array(points))
    cv2.fillConvexPoly(mask, hull, 255)

    return mask

def Bilateral_filter(frame, mask):
    smoothed = cv2.bilateralFilter(frame, d=9, sigmaColor=75, sigmaSpace=75)

    face_smoothed = cv2.bitwise_and(smoothed, smoothed, mask=mask)
    background = cv2.bitwise_and(frame, frame, mask=cv2.bitwise_not(mask))

    res = cv2.add(background, face_smoothed)
    return res


while stream.isOpened():
    ret, frame = stream.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))  # giảm kích thước cho nhanh
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = face_mesh.process(rgb_frame)

    display_frame = frame.copy()

    if res.multi_face_landmarks:
        for face_landmarks in res.multi_face_landmarks:
            mask = mesh_face(frame, face_landmarks)
            frame = Bilateral_filter(frame, mask)
            filtered_frame = apply_filter(frame, current_filter)

            display_frame = filtered_frame.copy()
            cv2.putText(display_frame, f'filter: {current_filter}', (400, 30),
                        cv2.FONT_HERSHEY_DUPLEX, 0.6, (200, 255, 200), 1)
            # mp_drawing.draw_landmarks(
            #     image=frame,
            #     landmark_list=face_landmarks,
            #     connections=mp_face_mesh.FACEMESH_CONTOURS,
            #     landmark_drawing_spec=None,
            #     connection_drawing_spec=mp_drawing_style.get_default_face_mesh_contours_style()
            # )
    else:
        filtered_frame = apply_filter(frame, current_filter)
        display_frame = filtered_frame.copy()
        cv2.putText(display_frame, f'FILTER: {current_filter}', (400, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 20, 200), 1)

    cur_time = time.time()
    fps = int(1 / (cur_time - prev_time)) if (cur_time - prev_time) != 0 else 0
    prev_time = cur_time

    cv2.putText(display_frame, f'FPS: {fps}', (20, 70), cv2.FONT_HERSHEY_PLAIN,
                2, (0, 196, 255), 2)

    if test_display:
        elapsed = (datetime.now() - text_start_time).total_seconds()
        if  elapsed < TIME_DISPLAY_SAVE_IMAGE:
            alpha = 1.0 - (1.2 * (elapsed - 0.2) / TIME_DISPLAY_SAVE_IMAGE)

            #Bản sao
            overlay = display_frame.copy()
            cv2.putText(overlay, message, (20, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (240, 200, 255), 1, cv2.LINE_AA)
            cv2.addWeighted(overlay, alpha, display_frame, 1 - alpha, 0, display_frame)
        else:
            test_display = False

    cv2.imshow("Webcam!", display_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('c'):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'{output_dir}/capture_{timestamp}.jpg'
        cv2.imwrite(filename, display_frame)
        
        print(f"Save completely capture {filename}")
        message = f"Save: {filename.split('/')[-1]}"
        test_display = True
        text_start_time = datetime.now()

    elif key == ord('q'):
        break
    
    elif key == ord('a'):
        filter_index = (filter_index - 1) % len(filter_list)
        current_filter = filter_list[filter_index]
        print(f"Filter change to: {current_filter}")
    elif key == ord('d'):
        filter_index = (filter_index + 1) % len(filter_list)
        current_filter = filter_list[filter_index]
        print(f"Filter change to: {current_filter}")

stream.release()
cv2.destroyAllWindows()