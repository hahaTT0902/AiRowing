import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QWidget, QHBoxLayout, QVBoxLayout
from PyQt5.QtCore import QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt

# ---- 视频显示控件 ----
class VideoWidget(QLabel):
    def __init__(self):
        super().__init__()
        self.setScaledContents(True)
        self.setMaximumWidth(1000)    # 设置最大宽度
        self.setMaximumHeight(800)   # 设置最大高度（可根据实际需求调整）
        self.setMinimumHeight(240)   # 可选，防止太小

    def update_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.setPixmap(QPixmap.fromImage(qimg))

# ---- Matplotlib 曲线控件 ----
class PlotWidget(FigureCanvas):
    def __init__(self, title, xlabel, ylabel, lines_info):
        self.fig, self.ax = plt.subplots()
        super().__init__(self.fig)
        self.ax.set_title(title)
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        self.lines = []
        for color, label in lines_info:
            line, = self.ax.plot([], [], color=color, label=label)
            self.lines.append(line)
        self.ax.legend()
        self.fig.tight_layout()

    def update_plot(self, x, ys_list):
        for line, y in zip(self.lines, ys_list):
            line.set_data(x, y)
        self.ax.relim()
        self.ax.autoscale_view()
        self.draw()

# ---- 后台线程：运行 main.py 的主循环 ----
class WorkerThread(QThread):
    data_signal = pyqtSignal(dict)
    def __init__(self):
        super().__init__()
        self._running = True

    def run(self):
        from main import main
        main(data_callback=self.data_signal.emit, running_flag=lambda: self._running)

    def stop(self):
        self._running = False

# ---- 主窗口 ----
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AiRowing 多视图GUI")
        self.video_widget = VideoWidget()
        # 上方：动作幅度
        lines_info1 = [
            ('green', 'Buttocks'),
            ('blue', 'Back'),
            ('magenta', 'Arms')
        ]
        self.plot1 = PlotWidget("Real-Time Movement", "Time (s)", "Movement (px)", lines_info1)
        # 下方：角度
        lines_info2 = [
            ('lime', 'leg_drive_angle'),
            ('cyan', 'back_angle'),
            ('orange', 'arm_angle')
        ]
        self.plot2 = PlotWidget("Angle at Phase Switch", "Time (s)", "Angle (°)", lines_info2)
        central = QWidget()
        main_vbox = QVBoxLayout()
        main_vbox.addWidget(self.video_widget)  
        plots_hbox = QHBoxLayout()
        plots_hbox.addWidget(self.plot1)
        plots_hbox.addWidget(self.plot2)
        main_vbox.addLayout(plots_hbox)   
        central.setLayout(main_vbox)
        self.setCentralWidget(central)

        # 启动后台线程
        self.worker = WorkerThread()
        self.worker.data_signal.connect(self.update_all)
        self.worker.start()

        self._latest_data = None
        self.timer = QTimer(self)
        self.timer.setInterval(100)
        self.timer.timeout.connect(self._refresh_plots)
        self.timer.start()

    def update_all(self, data):
        self.video_widget.update_frame(data['frame'])
        self._latest_data = data

    def _refresh_plots(self):
        data = self._latest_data
        if data is None:
            return
        x = data['time_series']
        if x:
            t_now = x[-1]
            t_min = max(x[0], t_now - 10)
            indices = [i for i, t in enumerate(x) if t >= t_min]
            x10 = [x[i] for i in indices]
            leg10 = [data['leg_series'][i] for i in indices]
            back10 = [data['back_series'][i] for i in indices]
            arm10 = [data['arm_series'][i] for i in indices]
        else:
            x10, leg10, back10, arm10 = [], [], [], []
        self.plot1.update_plot(x10, [leg10, back10, arm10])

        # 角度数据
        if data['toggle_angles']:
            filtered = [a for a in data['toggle_angles'] if a[0] >= t_min]
            if filtered:
                times = [a[0] for a in filtered]
                leg_angle = [a[2].get('leg_drive_angle', 0) for a in filtered]
                back_angle = [a[2].get('back_angle', 0) for a in filtered]
                arm_angle = [a[2].get('arm_angle', 0) for a in filtered]
                self.plot2.update_plot(times, [leg_angle, back_angle, arm_angle])
            else:
                self.plot2.update_plot([], [[], [], []])
        else:
            self.plot2.update_plot([], [[], [], []])

    def keyPressEvent(self, event):
        if event.text().lower() == 'q':
            self.worker.stop()
            self.worker.wait(2000)
            self.close()

    def closeEvent(self, event):
        self.worker.quit()
        self.worker.wait()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
