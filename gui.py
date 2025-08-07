import sys
import cv2
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QWidget, QHBoxLayout, QVBoxLayout
from PyQt5.QtCore import QTimer, QThread, pyqtSignal, Qt
from PyQt5.QtGui import QImage, QPixmap
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt

# ---- 视频显示控件 ----
class VideoWidget(QLabel):
    def __init__(self):
        super().__init__()
        self.setScaledContents(False)  # 保持比例，不自动拉伸
        self.setMaximumWidth(1000)
        self.setMaximumHeight(800)
        self.setMinimumHeight(240)

    def update_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        # 按控件大小缩放，保持比例
        pixmap = pixmap.scaled(self.size(), aspectRatioMode=Qt.KeepAspectRatio, transformMode=Qt.SmoothTransformation)
        self.setPixmap(pixmap)

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
        # 检测屏幕分辨率，设置缩放因子
        screen = QApplication.primaryScreen()
        size = screen.size()
        # 以1920x1080为基准
        base_w, base_h = 1920, 1080
        scale_w = size.width() / base_w
        scale_h = size.height() / base_h
        self._ui_scale = min(scale_w, scale_h)

        self.video_widget = VideoWidget()
        self.metrics_widget = MetricsWidget()
        self.suggestion_label = QLabel("")
        # 字体和控件高度自适应缩放
        sug_font_size = int(15 * self._ui_scale)
        self.suggestion_label.setStyleSheet(
            f"font-size: {sug_font_size}px; color: #2a7c2a; background: #f7f9fa; border-radius: 8px; padding: 8px;"
        )
        self.suggestion_label.setMinimumHeight(int(48 * self._ui_scale))

        # 左右布局
        main_hbox = QHBoxLayout()
        left_vbox = QVBoxLayout()
        left_vbox.addWidget(self.video_widget, 2)
        lines_info1 = [
            ('green', 'Buttocks'),
            ('blue', 'Back'),
            ('magenta', 'Arms')
        ]
        self.plot1 = PlotWidget("Real-Time Movement", "Time (s)", "Movement (px)", lines_info1)
        lines_info2 = [
            ('lime', 'leg_drive_angle'),
            ('cyan', 'back_angle'),
            ('orange', 'arm_angle')
        ]
        self.plot2 = PlotWidget("Angle at Phase Switch", "Time (s)", "Angle (°)", lines_info2)
        plots_hbox = QHBoxLayout()
        plots_hbox.addWidget(self.plot1)
        plots_hbox.addWidget(self.plot2)
        left_vbox.addLayout(plots_hbox)
        main_hbox.addLayout(left_vbox, 2)
        right_vbox = QVBoxLayout()
        right_vbox.addWidget(self.metrics_widget, 1)
        right_vbox.addWidget(self.suggestion_label)
        main_hbox.addLayout(right_vbox, 1)

        central = QWidget()
        central.setLayout(main_hbox)
        self.setCentralWidget(central)

        # 设置窗口初始大小为基准的0.8倍缩放
        self.resize(int(base_w * self._ui_scale * 0.8), int(base_h * self._ui_scale * 0.8))

        # 启动后台线程
        self.worker = WorkerThread()
        self.worker.data_signal.connect(self.update_all)
        self.worker.start()

        self._latest_data = None
        self._last_metrics = {'finish': [], 'catch': []}
        self._last_suggestions = "暂无建议"
        self.timer = QTimer(self)
        self.timer.setInterval(100)
        self.timer.timeout.connect(self._refresh_plots)
        self.timer.start()

    def update_all(self, data):
        self.video_widget.update_frame(data['frame'])
        self._latest_data = data
        self._update_metrics_and_suggestion()

    def _refresh_plots(self):
        # 曲线区块刷新
        data = self._latest_data
        # metrics区块只在没有任何历史有效数据时才显示暂无指标数据
        has_metrics = bool(self._last_metrics['finish'] or self._last_metrics['catch'])
        if data is None:
            if not has_metrics:
                self.metrics_widget.show_nodata()
                self.suggestion_label.setText("暂无建议")
            else:
                # 保持上一次内容
                self.metrics_widget.update_metrics(self._last_metrics['finish'], self._last_metrics['catch'])
                self.suggestion_label.setText(self._last_suggestions)
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
        # 指标条和建议区块刷新
        self._update_metrics_and_suggestion()

    def _update_metrics_and_suggestion(self):
        data = self._latest_data
        # 默认用上一次有效数据
        finish_metrics = self._last_metrics['finish']
        catch_metrics = self._last_metrics['catch']
        suggestions = self._last_suggestions
        if data is not None:
            toggle_angles = data.get('toggle_angles', [])
            new_finish = []
            new_catch = []
            new_suggestions = None
            if toggle_angles:
                finish = [a for a in toggle_angles if a[1] == 'Drive→Recovery']
                catch = [a for a in toggle_angles if a[1] == 'Recovery→Drive']
                if finish:
                    last = finish[-1]
                    angles = last[2]
                    new_finish = [
                        ("腿驱动角度", angles.get('leg_drive_angle', 0), 190, 220, "°"),
                        ("背部角度", angles.get('back_angle', 0), 105, 135, "°"),
                        ("手臂角度", angles.get('arm_angle', 0), 80, 110, "°")
                    ]
                if catch:
                    last = catch[-1]
                    angles = last[2]
                    new_catch = [
                        ("腿驱动角度", angles.get('leg_drive_angle', 0), 275, 300, "°"),
                        ("背部角度", angles.get('back_angle', 0), 20, 45, "°"),
                        ("手臂角度", angles.get('arm_angle', 0), 160, 180, "°")
                    ]
            # 只有有新数据时才刷新
            if new_finish or new_catch:
                finish_metrics = new_finish
                catch_metrics = new_catch
                self._last_metrics['finish'] = finish_metrics
                self._last_metrics['catch'] = catch_metrics
                # 生成新建议
                all_metrics = []
                if finish_metrics:
                    all_metrics += [("出水", *m) for m in finish_metrics]
                if catch_metrics:
                    all_metrics += [("入水", *m) for m in catch_metrics]
                sug_list = []
                for phase, name, value, low, high, unit in all_metrics:
                    if value < low:
                        if "腿" in name:
                            sug_list.append(f"{phase}：腿驱动角度偏小，建议加大腿部发力")
                        elif "背" in name:
                            sug_list.append(f"{phase}：背部角度偏小，建议后倾更多")
                        elif "手臂" in name:
                            sug_list.append(f"{phase}：手臂角度偏小，建议手臂再靠近身体")
                    elif value > high:
                        if "腿" in name:
                            sug_list.append(f"{phase}：腿驱动角度偏大，注意避免过度伸展")
                        elif "背" in name:
                            sug_list.append(f"{phase}：背部角度偏大，注意避免过度后仰")
                        elif "手臂" in name:
                            sug_list.append(f"{phase}：手臂角度偏大，注意手臂不要过度打开")
                if not sug_list:
                    new_suggestions = "动作良好"
                else:
                    new_suggestions = "\n".join(sug_list)
                self._last_suggestions = new_suggestions
                suggestions = new_suggestions
        # 刷新显示（无新数据时保持上一次内容）
        if finish_metrics or catch_metrics:
            self.metrics_widget.update_metrics(finish_metrics, catch_metrics)
        else:
            self.metrics_widget.show_nodata()
        self.suggestion_label.setText(suggestions)
class MetricsWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(10, 10, 10, 10)
        self.layout.setSpacing(12)
        self.groups = {}
        self.setStyleSheet("background: #f7f9fa;")

    def clear(self):
        for i in reversed(range(self.layout.count())):
            widget = self.layout.itemAt(i).widget()
            if widget:
                widget.deleteLater()
        self.groups = {}

    def add_group(self, title):
        label = QLabel(f"<b>{title}</b>")
        label.setStyleSheet("font-size: 20px; margin-bottom: 0px; padding-bottom: 0px;")
        self.layout.addWidget(label)
        group_widget = QWidget()
        group_layout = QVBoxLayout(group_widget)
        group_layout.setContentsMargins(0, 0, 0, 0)
        group_layout.setSpacing(6)
        self.layout.addWidget(group_widget)
        self.groups[title] = group_layout
        return group_layout

    def add_metric(self, group, name, value, low, high, unit, min_val=None, max_val=None):
        hbox = QHBoxLayout()
        label = QLabel(name)
        label.setFixedWidth(180)
        label.setStyleSheet("font-size: 14px;")
        hbox.addWidget(label)
        if min_val is None:
            min_val = low - 20
        if max_val is None:
            max_val = high + 20
        bar = MetricBar(low, high, float(value), unit, min_val=min_val, max_val=max_val)
        hbox.addWidget(bar, 1)
        group.addLayout(hbox)

    def update_metrics(self, finish_metrics, catch_metrics):
        # 只有有内容且数据变化时才刷新，否则保持原有内容
        if not hasattr(self, '_last_finish'):
            self._last_finish = []
            self._last_catch = []
        if not finish_metrics and not catch_metrics:
            return
        # 判断数据是否变化
        if finish_metrics == self._last_finish and catch_metrics == self._last_catch:
            return
        self._last_finish = finish_metrics.copy()
        self._last_catch = catch_metrics.copy()
        self.clear()
        finish_group = self.add_group("出水")
        for m in finish_metrics:
            self.add_metric(finish_group, *m)
        catch_group = self.add_group("入水")
        for m in catch_metrics:
            self.add_metric(catch_group, *m)

    def show_nodata(self):
        self.clear()
        label = QLabel("暂无指标数据")
        label.setStyleSheet("font-size: 16px; color: #888;")
        self.layout.addWidget(label)

class MetricBar(QWidget):
    def __init__(self, low, high, value, unit, min_val=0, max_val=100, parent=None):
        super().__init__(parent)
        self.low = low
        self.high = high
        self.value = value
        self.unit = unit
        self.min_val = min_val
        self.max_val = max_val
        self.setFixedHeight(60)
        self.setMinimumWidth(300)

    def set_value(self, value):
        self.value = value
        self.update()

    def paintEvent(self, event):
        from PyQt5 import QtGui, QtCore
        painter = QtGui.QPainter(self)
        rect = self.rect()
        margin = 32
        bar_rect = QtCore.QRect(margin, rect.height()//2-8, rect.width()-2*margin, 16)
        painter.setBrush(QtGui.QColor("#e0e5ea"))
        painter.setPen(QtCore.Qt.NoPen)
        painter.drawRoundedRect(bar_rect, 8, 8)
        if self.max_val > self.min_val:
            ratio = (self.value - self.min_val) / (self.max_val - self.min_val)
            ratio = max(0, min(1, ratio))
            green_rect = QtCore.QRect(bar_rect.left(), bar_rect.top(), int(bar_rect.width()*ratio), bar_rect.height())
            painter.setBrush(QtGui.QColor("#8fd18e"))
            painter.drawRoundedRect(green_rect, 8, 8)
        font = painter.font()
        font.setPointSize(11)
        painter.setFont(font)
        painter.setPen(QtGui.QColor("#4a7c4a"))
        left_ratio = (self.low - self.min_val) / (self.max_val - self.min_val)
        right_ratio = (self.high - self.min_val) / (self.max_val - self.min_val)
        left_x = bar_rect.left() + int(left_ratio * bar_rect.width())
        right_x = bar_rect.left() + int(right_ratio * bar_rect.width())
        left_label = f"{self.low}{self.unit}>"
        right_label = f"<{self.high}{self.unit}"
        left_label_width = painter.fontMetrics().width(left_label)
        painter.drawText(left_x - left_label_width + 2, bar_rect.top()-12, left_label)
        painter.drawText(right_x + 2, bar_rect.top()-12, right_label)
        if self.max_val > self.min_val:
            x = bar_rect.left() + int(ratio * bar_rect.width())
            painter.setPen(QtGui.QColor("#444"))
            painter.drawLine(x, bar_rect.top()-2, x, bar_rect.bottom()+2)
            painter.setPen(QtGui.QColor("#222"))
            font.setPointSize(13)
            painter.setFont(font)
            value_label = f"{self.value:.0f}{self.unit}"
            value_width = painter.fontMetrics().width(value_label)
            painter.drawText(x-value_width//2, bar_rect.bottom()+16, value_label)

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