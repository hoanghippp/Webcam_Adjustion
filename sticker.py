import mediapipe as mp
import os
import cv2
import numpy as np


mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.4,
    min_tracking_confidence=0.4
)

def load_sticker(sticker_dir='Stickers'):
    stickers = []
    stickers.append(('Normal', None))
    for file in os.listdir("Stickers"):
        if file.endswith(".png"):
            key = file.split('.')[0]
            img = cv2.imread(os.path.join("Stickers", file), cv2.IMREAD_UNCHANGED)  # Giữ alpha channel
            if img is not None and img.shape[2] == 4:
                stickers.append((key, img))
            else:
                print(f"Ảnh {file} không có kênh alpha (4 kênh), bỏ qua.")
    return stickers

def overlay_sticker(frame, sticker, x, y, w, h):
    sticker_resized = cv2.resize(sticker, (w, h))
    alpha_s = sticker_resized[:, :, 3] / 255.0
    alpha_l = 1.0 - alpha_s

    for c in range(3):
        frame[y:y+h, x:x+w, c] = (
            alpha_s * sticker_resized[:, :, c] + alpha_l * frame[y:y+h, x:x+w, c]
        )
    return frame

def main():
    cap = cv2.VideoCapture(0)
    sticker_dict = load_sticker()
    sticker_idx = 0


    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 480))
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = face_mesh.process(rgb_frame)

        cur_sticker_name, cur_sticker = sticker_dict[sticker_idx]

        if res.multi_face_landmarks:
            for face_landmark in res.multi_face_landmarks:
                h, w, _ = frame.shape
                left_eye = face_landmark.landmark[33]
                right_eye = face_landmark.landmark[263]

                x1 = int(left_eye.x * w)
                y1 = int(left_eye.y * h)
                x2 = int(right_eye.x * w)
                y2 = int(right_eye.y * h)

                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                width = abs(x2 - x1) * 2
                height = int(width * 0.5)

                if cur_sticker is not None:
                    frame = overlay_sticker(frame, cur_sticker, cx - int(width / 2),
                                            cy - int(height / 2), int(width), height)

        cv2.imshow("Sticker", frame)
        key = cv2.waitKey(1)

        if key == ord('q'):
            break
        elif key == ord('a'):
            sticker_idx = (sticker_idx + 1) % len(sticker_dict)
        elif key == ord('p'):
            sticker_idx = (sticker_idx - 1) % len(sticker_dict)
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
