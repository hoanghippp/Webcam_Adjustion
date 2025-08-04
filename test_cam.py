import sys
import cv2
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer, Qt

# Nếu chưa có các module bên dưới, bạn có thể comment chúng lại khi test thử cam
# from main import process_frame
# from filter import apply_filter
# from sticker import load_sticker
# from background import load_backgrounds

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

    def start_camera(self):
        if self.cap and self.cap.isOpened():
            print("Camera đã mở rồi.")
            return

        self.cap = cv2.VideoCapture(0)  # <== MỞ CAMERA Ở ĐÂY
        if not self.cap.isOpened():
            print("Không mở được camera.")
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
            return

        ret, frame = self.cap.read()
        if not ret:
            return

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(qt_image))

    def closeEvent(self, event):
        self.stop_camera()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = WebcamApp()
    win.show()
    sys.exit(app.exec_())
