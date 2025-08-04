import sys
import os
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QFileDialog
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer, Qt



try:
    from main import process_frame
except ImportError:
    def process_frame(frame, filter_idx, sticker_idx, bg_idx, filter_list, stickers, backgrounds):
        return frame

try:
    from sticker import load_sticker
except:
    def load_sticker(folder): return []

try:
    from background import load_backgrounds
except:
    def load_backgrounds(folder): return []



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
        self.btn_add_bg =QPushButton("Thêm nền mới")
        self.btn_add_sticker = QPushButton("Thêm Sticky mới")

        hbox_controls = QHBoxLayout()
        hbox_controls.addWidget(self.btn_next_filter)
        hbox_controls.addWidget(self.btn_next_sticker)
        hbox_controls.addWidget(self.btn_next_bg)
        hbox_controls.addWidget(self.btn_add_bg)
        hbox_controls.addWidget(self.btn_add_sticker)

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
        self.btn_add_bg.clicked.connect(self.add_background)
        self.btn_add_sticker.clicked.connect(self.add_sticker)

        self.filter_list = ["none", "gray", "sepia", "blur", "negative"]

        sticker_folder = "Stickers"
        self.stickers = load_sticker(sticker_folder) if os.path.exists(sticker_folder) else []

        background_folder = "Backgrounds"
        self.backgrounds = load_backgrounds(background_folder) if os.path.exists(background_folder) else []

        self.filter_idx = 0
        self.sticker_idx = 0
        self.bg_idx = 0

    def start_camera(self):
        if self.cap and self.cap.isOpened():
            print("Camera đã mở rồi.")
            return

        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
        camera_indices = [0, 1]
        found_camera = False

        for backend in backends:
            for index in camera_indices:
                print(f"Thử camera {index} với backend {backend}")
                self.cap = cv2.VideoCapture(index, backend)
                if self.cap.isOpened():
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    print("Đã mở camera thành công.")
                    found_camera = True
                    break
            if found_camera:
                break

        if not found_camera:
            print("Không thể mở webcam.")
            self.btn_start.setText("Bật Camera (Thất bại)")
            return

        self.timer.start(30)
        self.btn_start.setText("Tắt Camera")

    def stop_camera(self):
        if self.cap and self.cap.isOpened():
            self.cap.release()
            self.cap = None
            self.timer.stop()
            self.image_label.clear()
            self.image_label.setText("Camera đã tắt")
            self.btn_start.setText("Bật Camera")

    def toggle_camera(self):
        if self.cap and self.cap.isOpened():
            self.stop_camera()
        else:
            self.start_camera()

    def update_frame(self):
        if self.cap is None or not self.cap.isOpened():
            self.stop_camera()
            return

        ret, frame = self.cap.read()
        if not ret or frame is None:
            print("Không lấy được khung hình.")
            self.stop_camera()
            return

        try:

            filter_idx = self.filter_idx % len(self.filter_list) if self.filter_list else 0
            sticker_idx = self.sticker_idx % len(self.stickers) if self.stickers else 0
            bg_idx = self.bg_idx % len(self.backgrounds) if self.backgrounds else 0

            processed_frame = process_frame(
                frame,
                self.filter_idx % len(self.filter_list),
                self.sticker_idx % len(self.stickers) if self.stickers else 0,
                self.bg_idx % len(self.backgrounds) if self.backgrounds else 0,
                self.filter_list,
                self.stickers,
                self.backgrounds
            )

            rgb_image = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.image_label.setPixmap(QPixmap.fromImage(qt_img))

        except Exception as e:
            print(f"Lỗi khi xử lý/hiển thị khung hình: {e}")
            self.stop_camera()

    def next_filter(self):
        if self.filter_list:
            self.filter_idx = (self.filter_idx + 1) % len(self.filter_list)
            print(f"Bộ lọc hiện tại: {self.filter_list[self.filter_idx]}")

    def next_sticker(self):
        if self.stickers:
            self.sticker_idx = (self.sticker_idx + 1) % len(self.stickers)
            print(f"Nhãn dán hiện tại: {self.sticker_idx}")

    def next_bg(self):
        if self.backgrounds:
            self.bg_idx = (self.bg_idx + 1) % len(self.backgrounds)
            print(f"Nền hiện tại: {self.bg_idx}")

    def add_background(self):
        fname, _ = QFileDialog.getOpenFileName(self, 'Chọn ảnh nền', '.', "Images (*.png *.jpg *.jpeg)")
        if fname:
            img = cv2.imread(fname)
            if img is None:
                print("Không đọc được ảnh nền.")
                return
            self.backgrounds.append((os.path.basename(fname), img))
            print(f"Đã thêm nền mới: {fname} (tổng {len(self.backgrounds)})")

    def add_sticker(self):
        fname, _ = QFileDialog.getOpenFileName(self, 'Chọn ảnh sticker PNG', '.', "Images (*.png)")
        if fname:
            img = cv2.imread(fname, cv2.IMREAD_UNCHANGED)  # Đọc luôn alpha channel
            if img is None or img.shape[2] != 4:
                print("Sticker phải là ảnh PNG có nền trong suốt (4 kênh).")
                return
            self.stickers.append((os.path.basename(fname), img))
            print(f"Đã thêm sticker mới: {fname} (tổng {len(self.stickers)})")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = WebcamApp()
    win.show()
    sys.exit(app.exec_())
