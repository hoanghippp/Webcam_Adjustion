import cv2
import time
import os
import numpy as np # Import numpy


from filter import apply_filter
from sticker import apply_sticker, load_sticker
from background import load_backgrounds, resize_bg, apply_background



FILTERS = ['original', 'gray', 'sepia', 'cartoon', 'negative', 'blur']

# Biến thời gian để đo FPS
prev_time = time.time()

def process_frame(frame, filter_idx, sticker_idx, bg_idx, filter_list, stickers, backgrounds):
    global prev_time

    if frame is None or not isinstance(frame, np.ndarray):
        raise ValueError("Khung hình không hợp lệ.")

    try:
        frame = cv2.resize(frame, (640, 480))
    except Exception as e:
        raise ValueError(f"Lỗi resize frame: {e}")

    # Áp dụng filter
    if filter_list and 0 <= filter_idx < len(filter_list):
        current_filter_name = filter_list[filter_idx]
        if current_filter_name != 'none':
            frame = apply_filter(frame, current_filter_name)


    if stickers and 0 <= sticker_idx < len(stickers):
        _, sticker_img = stickers[sticker_idx]
        if sticker_img is not None:
            frame = apply_sticker(frame, sticker_img)


    if backgrounds and 0 <= bg_idx < len(backgrounds):
        _, bg_img = backgrounds[bg_idx]
        if bg_img is not None:
            bg_resized = resize_bg(bg_img, frame.shape[1], frame.shape[0])
            frame = apply_background(frame, bg_resized)


    curr_time = time.time()
    fps = 1 / (curr_time - prev_time + 1e-6)
    prev_time = curr_time
    cv2.putText(frame, f'FPS: {int(fps)}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return frame
