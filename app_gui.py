import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer
import cv2
from main import process_frame

class WebcamApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Webcam App - PyQt5")

        self.image_label = QLabel("Camera will appear here")
        self.image_label.setFixedSize(640, 480)

        self.btn_start = QPushButton("Start Camera")
        self.btn_next_filter = QPushButton("Next Filter")
        self.btn_next_bg = QPushButton("Next Background")
        self.btn_next_sticker = QPushButton("Next Stcker")

        hbox = QHBoxLayout()
        hbox.addWidget(self.btn_next_filter)
        hbox.addWidget(self.btn_next_sticker)
        hbox.addWidget(self.btn_next_bg)

        vbox = QHBoxLayout()
        vbox.addWidget(self.image_label)
        vbox.addLayout(hbox)
        vbox.addWidget(self.btn_start)
        self.setLayout(vbox)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        self.cap = None

        self.btn_start.clicked.connect(self.start_camera)
        self.btn_next_filter.clicked.connect(self.next_filter)
        self.btn_next_sticker.clicked.connect(self.next_sticker)
        self.btn_next_bg.clicked.connect(self.next_bg)

        self.filter_idx = 0
        self.bg_idx = 0
        self.sticker_idx = 0

    def start_camera(self):
        self.cap = cv2.VideoCapture(0)
        self.timer.start(30)

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        frame = cv2.resize(frame, (640, 480))
        frame = process_frame(frame, self.filter_idx, self.sticker_idx, self.bg_idx)

        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_img = QImage(rgb_image.data, w, h, bytes_per_line, QTime.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(qt_img))


    def next_filter(self):
        self.filter_idx += 1

    def next_bg(self):
        self.bg_idx += 1

    def next_sticker(self):
        self.sticker_idx += 1

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = WebcamApp()
    win.show()
    sys.exit(app.exec_())

