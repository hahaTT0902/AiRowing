import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QWidget, QHBoxLayout, QVBoxLayout
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QImage, QPixmap
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt

class VideoWidget(QLabel):
    def __init__(self):
        super().__init__()
        self.setScaledContents(True)

    def update_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.setPixmap(QPixmap.fromImage(qimg))

class PlotWidget(FigureCanvas):
    def __init__(self, title, xlabel, ylabel):
        self.fig, self.ax = plt.subplots()
        super().__init__(self.fig)
        self.ax.set_title(title)
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)

    def update_plot(self, x, y, label, color):
        self.ax.cla()
        self.ax.plot(x, y, label=label, color=color)
        self.ax.legend()
        self.draw()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AiRowing 多视图GUI")
        self.video_widget = VideoWidget()
        self.plot1 = PlotWidget("Real-Time Movement", "Time (s)", "Movement (px)")
        self.plot2 = PlotWidget("Angle at Phase Switch", "Time (s)", "Angle (°)")

        # 布局
        central = QWidget()
        hbox = QHBoxLayout()
        vbox = QVBoxLayout()
        vbox.addWidget(self.plot1)
        vbox.addWidget(self.plot2)
        hbox.addWidget(self.video_widget, 2)
        hbox.addLayout(vbox, 1)
        central.setLayout(hbox)
        self.setCentralWidget(central)

        # 定时器
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_all)
        self.timer.start(30)

        # 视频流
        self.cap = cv2.VideoCapture(0)  # 或者用你的视频文件
        self.t = 0
        self.x = []
        self.y1 = []
        self.y2 = []

    def update_all(self):
        ret, frame = self.cap.read()
        if not ret:
            return
        self.video_widget.update_frame(frame)
        # 这里用假数据，实际请用你的 time_series、leg_series、angle_series 等
        self.t += 0.03
        self.x.append(self.t)
        self.y1.append(np.sin(self.t))
        self.y2.append(np.cos(self.t))
        self.plot1.update_plot(self.x, self.y1, "Buttocks", "green")
        self.plot2.update_plot(self.x, self.y2, "Angle", "blue")

    def closeEvent(self, event):
        self.cap.release()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())