import cv2
import mediapipe as mp
import numpy as np
import os

mp_selfie_segmentation = mp.solutions.selfie_segmentation
segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

def load_backgrounds(bg_dir='Backgrounds'):
    backgrounds = []
    backgrounds.append(("Original", None))

    for filename in os.listdir(bg_dir):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            key = filename.split('.')[0]
            path = os.path.join(bg_dir, filename)
            img = cv2.imread(path)
            if img is not None:
                backgrounds.append((key, img))

    return backgrounds

def resize_bg(cur_bg, w, h):
    return cv2.resize(cur_bg, (w, h))

def apply_background(frame, bg_img):
    # Chuyển ảnh sang RGB cho MediaPipe
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Dự đoán segmentation mask
    results = segmentation.process(rgb)

    if results.segmentation_mask is None:
        return frame  # fallback nếu không detect được người

    # Tạo mask nhị phân
    mask = results.segmentation_mask
    condition = mask > 0.6  # ngưỡng càng cao thì tách càng sát người

    # Resize bg_img nếu cần (đã resize ở ngoài)
    bg_img = cv2.resize(bg_img, (frame.shape[1], frame.shape[0]))

    # Kết hợp ảnh
    output_image = np.where(condition[..., None], frame, bg_img)
    return output_image

def main():
    cap = cv2.VideoCapture(0)

    bg_lst = load_backgrounds()
    cur_bg_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 480))
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = segmentation.process(rgb_frame)

        h, w, _ = frame.shape
        cur_bg_name, cur_bg = bg_lst[cur_bg_idx]
        if cur_bg is None:
            output_frame = frame.copy()
        else:

            background_image = resize_bg(cur_bg, w, h)
            mask = res.segmentation_mask
            condition = mask > 0.6
            output_frame = np.where(condition[:, :, None], frame, background_image)

        cv2.putText(output_frame, f"Background: {cur_bg_name}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Virtual Background", output_frame)
        key = cv2.waitKey(1)

        if key == ord("q"):
            break
        elif key == ord('n'):
            cur_bg_idx = (cur_bg_idx + 1) % len(bg_lst)
        elif key == ord('p'):
            cur_bg_idx = (cur_bg_idx - 1) % len(bg_lst)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()