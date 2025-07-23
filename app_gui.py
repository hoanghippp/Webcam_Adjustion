import sys
import os
import cv2
import numpy as np
import time

from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer, Qt

from main import process_frame
from filter import apply_filter
from sticker import load_sticker
from background import load_backgrounds

class WebcamApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Ứng dụng Webcam - PyQt5")

        self.image_label = QLabel("Camera sẽ hiển thị ở đây")
        self.image_label.setFixedSize(640, 480)
        self.image_label.setAlignment(Qt.AlignCenter)

        self.btn_start = QPushButton("Bật Camera")
        self.btn_next_filter = QPushButton("Bộ Lọc Kế Tiếp")
        self.btn_next_bg = QPushButton("Nền Kế Tiếp")
        self.btn_next_sticker = QPushButton("Nhãn Dán Kế Tiếp")

        hbox_controls = QHBoxLayout()
        hbox_controls.addWidget(self.btn_next_filter)
        hbox_controls.addWidget(self.btn_next_sticker)
        hbox_controls.addWidget(self.btn_next_bg)

        vbox_main = QVBoxLayout()
        vbox_main.addWidget(self.image_label)
        vbox_main.addLayout(hbox_controls)
        vbox_main.addWidget(self.btn_start)
        self.setLayout(vbox_main)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.cap = None

        self.btn_start.clicked.connect(self.toggle_camera)
        self.btn_next_filter.clicked.connect(self.next_filter)
        self.btn_next_sticker.clicked.connect(self.next_sticker)
        self.btn_next_bg.clicked.connect(self.next_bg)

        self.filter_list = ["none", "gray", "sepia", "blur", "negative"]

        sticker_folder = "Stickers"
        if not os.path.exists(sticker_folder):
            print(f"Cảnh báo: Thư mục '{sticker_folder}' không tồn tại. Nhãn dán sẽ không được tải.")
            self.stickers = []
        else:
            self.stickers = load_sticker(sticker_folder)
            if not self.stickers:
                print(f"Không có nhãn dán nào được tìm thấy trong '{sticker_folder}'.")

        background_folder = "Backgrounds"
        if not os.path.exists(background_folder):
            print(f"Cảnh báo: Thư mục '{background_folder}' không tồn tại. Nền sẽ không được tải.")
            self.backgrounds = []
        else:
            self.backgrounds = load_backgrounds(background_folder)
            if not self.backgrounds:
                print(f"Không có nền nào được tìm thấy trong '{background_folder}'.")

        self.filter_idx = 0
        self.bg_idx = 0
        self.sticker_idx = 0

    def start_camera(self):
        if self.cap and self.cap.isOpened():
            print("Camera đã mở rồi.")
            return

        # --- THAY ĐỔI Ở ĐÂY ---
        # Thử các API backend khác nhau cùng với chỉ số camera
        # DirectShow (MSMF) thường là mặc định và tốt nhất trên Windows
        # CAP_DSHOW: Sử dụng DirectShow API (thường tốt trên Windows)
        # CAP_MSMF: Sử dụng Media Foundation (thay thế DirectShow, cũng trên Windows)
        # CAP_V4L2: Dành cho Linux
        # CAP_ANY: OpenCV tự động chọn
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
        camera_indices = [0, 1] # Thử camera 0 trước, sau đó 1

        found_camera = False
        for backend in backends:
            for index in camera_indices:
                print(f"Đang thử mở camera chỉ số {index} với backend {backend}...")
                self.cap = cv2.VideoCapture(index, backend)
                if self.cap.isOpened():
                    # Đặt độ phân giải nếu cần thiết (có thể giúp ổn định)
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    # Lấy lại độ phân giải thực tế để kiểm tra
                    width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    print(f"Đã mở camera thành công với chỉ số: {index}, backend: {backend}. Độ phân giải: {width}x{height}")
                    found_camera = True
                    break # Thoát vòng lặp camera_indices
            if found_camera:
                break # Thoát vòng lặp backends
        # --- KẾT THÚC THAY ĐỔI ---

        if not found_camera:
            print("--------------------------------------------------")
            print("LỖI: Không thể mở webcam với bất kỳ chỉ số hoặc backend nào.")
            print("Vui lòng kiểm tra các điều sau:")
            print("  1. Webcam của bạn có đang bị ứng dụng khác sử dụng không?")
            print("  2. Bạn đã cấp quyền truy cập camera cho ứng dụng này trên Windows/macOS chưa?")
            print("  3. Webcam có được kết nối đúng cách và hoạt động bình thường không?")
            print("--------------------------------------------------")
            self.btn_start.setText("Bật Camera (Thất bại)")
            return

        self.timer.start(30)
        self.btn_start.setText("Tắt Camera")
        print("Camera đã bật thành công.")

    def stop_camera(self):
        if self.cap and self.cap.isOpened():
            self.cap.release()
            self.cap = None
            self.timer.stop()
            self.image_label.clear()
            self.image_label.setText("Camera đã tắt")
            self.btn_start.setText("Bật Camera")
            print("Camera đã tắt.")

    def toggle_camera(self):
        if self.cap and self.cap.isOpened():
            self.stop_camera()
        else:
            self.start_camera()

    def update_frame(self):
        if self.cap is None or not self.cap.isOpened():
            print("Camera chưa mở hoặc bị mất kết nối trong update_frame.")
            self.stop_camera()
            return

        ret, frame = self.cap.read()
        if not ret or frame is None:
            print("Không thể lấy được khung hình từ camera (ret=False hoặc frame=None). Đang dừng camera.")
            self.stop_camera()
            return

        try:
            processed_frame = process_frame(
                frame,
                self.filter_idx,
                self.sticker_idx,
                self.bg_idx,
                self.filter_list,
                self.stickers,
                self.backgrounds
            )
        except ValueError as ve:
            print(f"Lỗi xử lý khung hình (ValueError): {ve}. Đang dừng camera.")
            self.stop_camera()
            return
        except Exception as e:
            print(f"Lỗi không xác định trong process_frame: {e}. Đang dừng camera.")
            self.stop_camera()
            return

        try:
            rgb_image = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.image_label.setPixmap(QPixmap.fromImage(qt_img))
        except Exception as e:
            print(f"Lỗi khi chuyển đổi hoặc hiển thị khung hình sang QImage: {e}. Đang dừng camera.")
            self.stop_camera()

    def next_filter(self):
        if self.filter_list:
            self.filter_idx = (self.filter_idx + 1) % len(self.filter_list)
            print(f"Chuyển sang bộ lọc: {self.filter_list[self.filter_idx]}")
        else:
            print("Không có bộ lọc nào để chọn.")

    def next_sticker(self):
        if self.stickers:
            self.sticker_idx = (self.sticker_idx + 1) % len(self.stickers)
            print(f"Chuyển sang nhãn dán số: {self.sticker_idx}")
        else:
            print("Không có nhãn dán nào để chọn.")

    def next_bg(self):
        if self.backgrounds:
            self.bg_idx = (self.bg_idx + 1) % len(self.backgrounds)
            print(f"Chuyển sang nền số: {self.bg_idx}")
        else:
            print("Không có nền nào để chọn.")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = WebcamApp()
    win.show()
    sys.exit(app.exec_())