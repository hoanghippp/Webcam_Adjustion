import cv2
import time
import os
import numpy as np # Import numpy

# Import các hàm từ các file khác
from filter import apply_filter
from sticker import apply_sticker, load_sticker
from background import load_backgrounds, resize_bg, apply_background


# Khởi tạo các danh sách tài nguyên (được tải trong WebcamApp)
FILTERS = ['original', 'gray', 'sepia', 'cartoon', 'negative', 'blur'] # Thêm 'blur'
# STICKERS và BACKGROUNDS sẽ được tải trong WebcamApp và truyền vào

# Biến thời gian để đo FPS
prev_time = time.time()

def process_frame(frame, filter_idx, sticker_idx, bg_idx, filter_list, stickers, backgrounds):
    """
    Xử lý một khung hình: resize, áp dụng bộ lọc, nhãn dán, và nền.
    """
    global prev_time

    # **KIỂM TRA QUAN TRỌNG NHẤT Ở ĐÂY:**
    # Đảm bảo 'frame' là một ảnh hợp lệ (NumPy array) trước khi xử lý
    if frame is None or not isinstance(frame, np.ndarray):
        raise ValueError("Khung hình nhận được trong process_frame không hợp lệ (None hoặc không phải NumPy array).")

    try:
        # Luôn resize khung hình đầu tiên để đảm bảo kích thước chuẩn
        frame = cv2.resize(frame, (640, 480))
    except Exception as e:
        print(f"Lỗi khi thay đổi kích thước khung hình trong process_frame: {e}.")
        # Ném lại lỗi để hàm gọi biết có vấn đề
        raise ValueError("Lỗi resize khung hình, có thể do khung hình không hợp lệ.")


    # Áp dụng filter
    if filter_list and 0 <= filter_idx < len(filter_list):
        current_filter_name = filter_list[filter_idx]
        if current_filter_name == 'none': # Nếu là filter 'none', không làm gì cả
            pass
        else:
            frame = apply_filter(frame, current_filter_name)
    else:
        # Nếu filter_list rỗng hoặc chỉ số không hợp lệ, không áp dụng filter
        pass


    # Áp dụng sticker
    if stickers and 0 <= sticker_idx < len(stickers):
        frame = apply_sticker(frame, stickers[sticker_idx])
    else:
        # print("Không có nhãn dán để áp dụng hoặc chỉ số nhãn dán không hợp lệ.")
        pass


    # Áp dụng background
    if backgrounds and 0 <= bg_idx < len(backgrounds):
        # Đảm bảo bg_img được resize đúng với kích thước của frame
        # (frame.shape[1] là chiều rộng, frame.shape[0] là chiều cao)
        bg_img = resize_bg(backgrounds[bg_idx], frame.shape[1], frame.shape[0])
        if bg_img is not None:
            frame = apply_background(frame, bg_img)
    else:
        # print("Không có nền để áp dụng hoặc chỉ số nền không hợp lệ.")
        pass

    # Tính và hiển thị FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time + 1e-6) # Tránh chia cho 0
    prev_time = curr_time
    cv2.putText(frame, f'FPS: {int(fps)}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return frame