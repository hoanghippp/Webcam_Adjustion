import cv2
import numpy as np
import mediapipe as mp
import time
import os
from datetime import datetime
from filter import apply_filter
from sticker import load_sticker, overlay_sticker
from background import load_backgrounds, resize_bg

# Setup paths & output

output_dir = 'assets/captures'
os.makedirs(output_dir, exist_ok=True)

# === Load Resources ===
filter_list = ["none", "warm_orange", "cool_blue", "white_pink", "vintage"]
filter_index = 0
current_filter = filter_list[filter_index]

stickers = load_sticker()
sticker_idx = 0

bg_list = load_backgrounds()
bg_idx = 0

# === Mediapipe setup ===
mp_selfie_segmentation = mp.solutions.selfie_segmentation
segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=2,
    refine_landmarks=True,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# === Webcam capture ===
cap = cv2.VideoCapture(0)
prev_time = 0

def main():
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 480))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb)
        display_frame = frame.copy()

        # === Face detected: apply filter + sticker ===
        if result.multi_face_landmarks:
            for landmarks in result.multi_face_landmarks:
                mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                points = [(int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])) for lm in landmarks.landmark]
                hull = cv2.convexHull(np.array(points))
                cv2.fillConvexPoly(mask, hull, 255)
                smooth = cv2.bilateralFilter(frame, d=9, sigmaColor=75, sigmaSpace=75)
                frame = cv2.bitwise_and(smooth, smooth, mask=mask) + \
                        cv2.bitwise_and(frame, frame, mask=cv2.bitwise_not(mask))

                frame = apply_filter(frame, current_filter)

                # === Overlay sticker ===
                _, sticker_img = stickers[sticker_idx]
                if sticker_img is not None:
                    h, w, _ = frame.shape
                    left_eye = landmarks.landmark[33]
                    right_eye = landmarks.landmark[263]
                    x1, y1 = int(left_eye.x * w), int(left_eye.y * h)
                    x2, y2 = int(right_eye.x * w), int(right_eye.y * h)
                    cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                    width = abs(x2 - x1) * 2
                    height = int(width * 0.5)
                    x = max(0, cx - int(width / 2))
                    y = max(0, cy - int(height / 2))
                    frame = overlay_sticker(frame, sticker_img, x, y, int(width), int(height))
        else:
            frame = apply_filter(frame, current_filter)

        # === Background replacement ===
        res = segmentation.process(rgb)
        h, w, _ = frame.shape
        cur_bg_name, cur_bg = bg_list[bg_idx]

        if cur_bg is None:
            output_frame = frame.copy()
        else:
            background_image = resize_bg(cur_bg, w, h)
            mask = res.segmentation_mask
            condition = mask > 0.6
            output_frame = np.where(condition[:, :, None], frame, background_image)

        # === Display info ===
        cur_time = time.time()
        fps = int(1 / (cur_time - prev_time)) if (cur_time - prev_time) != 0 else 0
        prev_time = cur_time

        cv2.putText(output_frame, f"Filter: {current_filter}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 150, 150), 1)
        cv2.putText(output_frame, f"Background: {cur_bg_name}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 255, 150), 1)
        cv2.putText(output_frame, f"FPS: {fps}", (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 255, 255), 1)

        cv2.imshow("Webcam App", output_frame)

        # === Key control ===
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('a'):
            filter_index = (filter_index - 1) % len(filter_list)
            current_filter = filter_list[filter_index]
        elif key == ord('d'):
            filter_index = (filter_index + 1) % len(filter_list)
            current_filter = filter_list[filter_index]
        elif key == ord('n'):
            sticker_idx = (sticker_idx + 1) % len(stickers)
        elif key == ord('p'):
            sticker_idx = (sticker_idx - 1) % len(stickers)
        elif key == ord('1'):
            bg_idx = (bg_idx + 1) % len(bg_list)
        elif key == ord('0'):
            bg_idx = (bg_idx - 1) % len(bg_list)
        elif key == ord('c'):
            filename = os.path.join(output_dir, f"capture_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
            cv2.imwrite(filename, output_frame)
            print(f"Captured: {filename}")

    cap.release()
    cv2.destroyAllWindows()

def process_frame(frame, filter_idx, sticker_idx, bg_idx):
    # === Chuẩn bị ===
    filter_list = ["none", "warm_orange", "cool_blue", "white_pink", "vintage"]
    current_filter = filter_list[filter_idx]

    stickers = load_sticker()
    bg_list = load_backgrounds()
    _, cur_bg = bg_list[bg_idx]

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)
    display_frame = frame.copy()

    # === Nếu có khuôn mặt: filter + sticker ===
    if result.multi_face_landmarks:
        for landmarks in result.multi_face_landmarks:
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            points = [(int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])) for lm in landmarks.landmark]
            hull = cv2.convexHull(np.array(points))
            cv2.fillConvexPoly(mask, hull, 255)
            smooth = cv2.bilateralFilter(frame, d=9, sigmaColor=75, sigmaSpace=75)
            frame = cv2.bitwise_and(smooth, smooth, mask=mask) + \
                    cv2.bitwise_and(frame, frame, mask=cv2.bitwise_not(mask))

            frame = apply_filter(frame, current_filter)

            # === Gắn sticker ===
            if sticker_idx < len(stickers):
                _, sticker_img = stickers[sticker_idx]
                if sticker_img is not None:
                    h, w, _ = frame.shape
                    left_eye = landmarks.landmark[33]
                    right_eye = landmarks.landmark[263]
                    x1, y1 = int(left_eye.x * w), int(left_eye.y * h)
                    x2, y2 = int(right_eye.x * w), int(right_eye.y * h)
                    cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                    width = abs(x2 - x1) * 2
                    height = int(width * 0.5)
                    x = max(0, cx - int(width / 2))
                    y = max(0, cy - int(height / 2))
                    frame = overlay_sticker(frame, sticker_img, x, y, int(width), int(height))
    else:
        frame = apply_filter(frame, current_filter)

    # === Thay nền ===
    res = segmentation.process(rgb)
    h, w, _ = frame.shape

    if cur_bg is None:
        output_frame = frame.copy()
    else:
        background_image = resize_bg(cur_bg, w, h)
        mask = res.segmentation_mask
        condition = mask > 0.6
        output_frame = np.where(condition[:, :, None], frame, background_image)

    return output_frame