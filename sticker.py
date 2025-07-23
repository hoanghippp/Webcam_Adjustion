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

def overlay_transparent(background, overlay, x, y, overlay_size=None):
    """Overlay PNG trong suốt lên ảnh gốc tại (x, y)"""
    if overlay_size:
        overlay = cv2.resize(overlay, overlay_size)

    b, g, r, a = cv2.split(overlay)
    overlay_rgb = cv2.merge((b, g, r))
    mask = cv2.merge((a, a, a)) / 255.0

    h, w = overlay.shape[:2]
    roi = background[y:y+h, x:x+w]

    if roi.shape[0] != h or roi.shape[1] != w:
        return background  # tránh lỗi khi overlay vượt ra ngoài

    blended = (1.0 - mask) * roi + mask * overlay_rgb
    background[y:y+h, x:x+w] = blended.astype(np.uint8)

    return background

def apply_sticker(frame, sticker_img):
    """
    Gắn sticker PNG trong suốt (ví dụ: kính, nón) lên mặt người trong frame.
    """
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if not results.multi_face_landmarks:
        return frame

    face = results.multi_face_landmarks[0]
    ih, iw = h, w

    # Lấy điểm mắt trái (33) và mắt phải (263)
    left_eye = face.landmark[33]
    right_eye = face.landmark[263]

    x1, y1 = int(left_eye.x * iw), int(left_eye.y * ih)
    x2, y2 = int(right_eye.x * iw), int(right_eye.y * ih)

    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2  # tâm giữa 2 mắt
    w_sticker = int(1.3 * abs(x2 - x1))     # chiều rộng dựa trên khoảng cách giữa mắt
    h_sticker = int(sticker_img.shape[0] * w_sticker / sticker_img.shape[1])  # theo tỉ lệ PNG gốc

    x_offset = cx - w_sticker // 2
    y_offset = cy - h_sticker // 2

    # Giới hạn vùng không bị out-of-bounds
    x_offset = max(0, x_offset)
    y_offset = max(0, y_offset)

    frame = overlay_transparent(frame, sticker_img, x_offset, y_offset, (w_sticker, h_sticker))
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
