import csv
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5 import QtWidgets, QtCore, QtGui
from scipy.signal import savgol_filter

joint_indices = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
bones = [
    (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 12), (23, 24), (11, 23), (12, 24),
    (23, 25), (25, 27), (24, 26), (26, 28)
]

# 读取 log.csv
frames = []
times = []
leg = []
back = []
arm = []
spm = []
phase = []

switch_time = []
switch_type = []
leg_angle = []
back_angle = []
arm_angle = []

with open('log.csv') as f:
    reader = csv.DictReader(f)
    last_phase = None
    for row in reader:
        try:
            joint_pos = {}
            joint_pos[23] = np.array([0.0, 0.0])
            joint_pos[24] = joint_pos[23] + np.array([
                float(row.get("vec_23_24_dx", 0)),
                float(row.get("vec_23_24_dy", 0))
            ])
            joint_pos[11] = joint_pos[23] + np.array([
                float(row.get("vec_11_23_dx", 0)),
                float(row.get("vec_11_23_dy", 0))
            ])
            joint_pos[12] = joint_pos[24] + np.array([
                float(row.get("vec_12_24_dx", 0)),
                float(row.get("vec_12_24_dy", 0))
            ])
            joint_pos[13] = joint_pos[11] - np.array([
                float(row.get("vec_11_13_dx", 0)),
                float(row.get("vec_11_13_dy", 0))
            ])
            joint_pos[15] = joint_pos[13] - np.array([
                float(row.get("vec_13_15_dx", 0)),
                float(row.get("vec_13_15_dy", 0))
            ])
            joint_pos[14] = joint_pos[12] - np.array([
                float(row.get("vec_12_14_dx", 0)),
                float(row.get("vec_12_14_dy", 0))
            ])
            joint_pos[16] = joint_pos[14] - np.array([
                float(row.get("vec_14_16_dx", 0)),
                float(row.get("vec_14_16_dy", 0))
            ])
            joint_pos[25] = joint_pos[23] - np.array([
                float(row.get("vec_23_25_dx", 0)),
                float(row.get("vec_23_25_dy", 0))
            ])
            joint_pos[27] = joint_pos[25] - np.array([
                float(row.get("vec_25_27_dx", 0)),
                float(row.get("vec_25_27_dy", 0))
            ])
            joint_pos[26] = joint_pos[24] - np.array([
                float(row.get("vec_24_26_dx", 0)),
                float(row.get("vec_24_26_dy", 0))
            ])
            joint_pos[28] = joint_pos[26] - np.array([
                float(row.get("vec_26_28_dx", 0)),
                float(row.get("vec_26_28_dy", 0))
            ])
            xs = [joint_pos[idx][0] for idx in joint_indices]
            ys = [joint_pos[idx][1] for idx in joint_indices]
            frames.append((xs, ys))
            times.append(float(row['Time']))
            leg.append(float(row['Leg Movement']))
            back.append(float(row['Back Movement']))
            arm.append(float(row['Arm Movement']))
            spm.append(float(row['SPM']) if row['SPM'] else 0)
            phase_val = row['Phase']
            phase.append(phase_val)
            # 自动识别切换点
            if last_phase is not None:
                if last_phase.lower().startswith('recovery') and phase_val.lower().startswith('drive'):
                    switch_time.append(float(row['Time']))
                    switch_type.append('catch')
                    leg_angle.append(float(row['leg_drive_angle']) if row['leg_drive_angle'] else None)
                    back_angle.append(float(row['back_angle']) if row['back_angle'] else None)
                    arm_angle.append(float(row['arm_angle']) if row['arm_angle'] else None)
                elif last_phase.lower().startswith('drive') and phase_val.lower().startswith('recovery'):
                    switch_time.append(float(row['Time']))
                    switch_type.append('finish')
                    leg_angle.append(float(row['leg_drive_angle']) if row['leg_drive_angle'] else None)
                    back_angle.append(float(row['back_angle']) if row['back_angle'] else None)
                    arm_angle.append(float(row['arm_angle']) if row['arm_angle'] else None)
            last_phase = phase_val
        except Exception as e:
            print(f"[Warn] 跳过一帧，数据异常: {e}")
            continue

def normalize(xs, ys):
    xs = np.array(xs)
    ys = np.array(ys)
    center_x = (xs[joint_indices.index(23)] + xs[joint_indices.index(24)]) / 2
    center_y = (ys[joint_indices.index(23)] + ys[joint_indices.index(24)]) / 2
    xs = xs - center_x
    ys = ys - center_y
    scale = max(np.max(np.abs(xs)), np.max(np.max(ys)), 1e-6)
    xs = xs / scale
    ys = ys / scale
    return xs, ys


# ----------- MetricsWidget -------------
class MetricsWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QtWidgets.QVBoxLayout(self)
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
        label = QtWidgets.QLabel(f"<b>{title}</b>")
        label.setStyleSheet("font-size: 20px; margin-bottom: 0px; padding-bottom: 0px;")
        self.layout.addWidget(label)
        group_widget = QtWidgets.QWidget()
        group_layout = QtWidgets.QVBoxLayout(group_widget)
        group_layout.setContentsMargins(0, 0, 0, 0)
        group_layout.setSpacing(6)
        self.layout.addWidget(group_widget)
        self.groups[title] = group_layout
        return group_layout

    def add_metric(self, group, name, value, low, high, unit, good_range=None):
        hbox = QtWidgets.QHBoxLayout()
        label = QtWidgets.QLabel(name)
        label.setFixedWidth(180)
        label.setStyleSheet("font-size: 14px;")
        hbox.addWidget(label)
        # 自动设置进度条范围
        if name.lower().startswith("handle height"):
            min_val, max_val = 0, 100
        else:
            min_val, max_val = 0, 360
        bar = MetricBar(low, high, float(value), unit, min_val=min_val, max_val=max_val)
        hbox.addWidget(bar, 1)
        group.addLayout(hbox)

    def update_metrics(self, finish_metrics, catch_metrics):
        self.clear()
        finish_group = self.add_group("出水")
        for m in finish_metrics:
            self.add_metric(finish_group, *m)
        catch_group = self.add_group("入水")
        for m in catch_metrics:
            self.add_metric(catch_group, *m)

    def show_nodata(self):
        self.clear()
        label = QtWidgets.QLabel("暂无指标数据")
        label.setStyleSheet("font-size: 16px; color: #888;")
        self.layout.addWidget(label)

class MetricBar(QtWidgets.QWidget):
    def __init__(self, low, high, value, unit, min_val=0, max_val=100, parent=None):
        super().__init__(parent)
        self.low = low
        self.high = high
        self.value = value
        self.unit = unit
        self.min_val = min_val
        self.max_val = max_val
        self.setFixedHeight(60)      # 原来是44，改为60
        self.setMinimumWidth(300)    # 原来是220，改为300

    def set_value(self, value):
        self.value = value
        self.update()

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        rect = self.rect()
        margin = 32
        bar_rect = QtCore.QRect(margin, rect.height()//2-8, rect.width()-2*margin, 16)
        # 灰色底条
        painter.setBrush(QtGui.QColor("#e0e5ea"))
        painter.setPen(QtCore.Qt.NoPen)
        painter.drawRoundedRect(bar_rect, 8, 8)
        # 绿色进度条（从左到当前值）
        if self.max_val > self.min_val:
            ratio = (self.value - self.min_val) / (self.max_val - self.min_val)
            ratio = max(0, min(1, ratio))
            green_rect = QtCore.QRect(bar_rect.left(), bar_rect.top(), int(bar_rect.width()*ratio), bar_rect.height())
            painter.setBrush(QtGui.QColor("#8fd18e"))
            painter.drawRoundedRect(green_rect, 8, 8)
        # 区间标签悬浮在绿色区间两端（动态位置，右对齐和左对齐）
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
        # 右对齐区间左端
        left_label_width = painter.fontMetrics().width(left_label)
        painter.drawText(left_x - left_label_width + 2, bar_rect.top()-12, left_label)
        # 左对齐区间右端
        painter.drawText(right_x + 2, bar_rect.top()-12, right_label)
        # 当前值指示线和数值
        if self.max_val > self.min_val:
            x = bar_rect.left() + int(ratio * bar_rect.width())
            painter.setPen(QtGui.QColor("#444"))
            painter.drawLine(x, bar_rect.top()-2, x, bar_rect.bottom()+2)
            # 当前值标签（居中显示在竖线下方）
            painter.setPen(QtGui.QColor("#222"))
            font.setPointSize(13)
            painter.setFont(font)
            value_label = f"{self.value:.0f}{self.unit}"
            value_width = painter.fontMetrics().width(value_label)
            painter.drawText(x-value_width//2, bar_rect.bottom()+16, value_label)

# ----------- MainWindow -------------
class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("赛后分析")
        self.resize(1200, 700)
        main_layout = QtWidgets.QHBoxLayout(self)
        # 左侧 matplotlib
        left_layout = QtWidgets.QVBoxLayout()
        self.fig = Figure(figsize=(8, 7))
        self.canvas = FigureCanvas(self.fig)
        left_layout.addWidget(self.canvas, 2)
        # 初始化三个子图
        self.gs = self.fig.add_gridspec(2, 2, width_ratios=[1, 1.2], height_ratios=[1, 1])
        self.ax_stick = self.fig.add_subplot(self.gs[:, 0])
        self.ax_curve = self.fig.add_subplot(self.gs[0, 1])
        self.ax_angle = self.fig.add_subplot(self.gs[1, 1])
        # 建议区块
        self.suggestion_label = QtWidgets.QLabel("")
        self.suggestion_label.setStyleSheet(
            "font-size: 15px; color: #2a7c2a; background: #f7f9fa; border-radius: 8px; padding: 8px;"
        )
        self.suggestion_label.setMinimumHeight(48)
        left_layout.addWidget(self.suggestion_label)
        main_layout.addLayout(left_layout, 2)
        # 右侧指标条和控制区（原有代码不变）
        right_layout = QtWidgets.QVBoxLayout()
        self.metrics = MetricsWidget()
        right_layout.addWidget(self.metrics, 1)
        # 播放控制区（放在右下角）
        ctrl_layout = QtWidgets.QHBoxLayout()
        self.play_btn = QtWidgets.QPushButton("播放")
        self.play_btn.setCheckable(True)
        self.play_btn.clicked.connect(self.toggle_play)
        ctrl_layout.addWidget(self.play_btn)
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(len(frames)-1)
        self.slider.setValue(0)
        self.slider.valueChanged.connect(self.on_slider)
        ctrl_layout.addWidget(self.slider, 1)
        right_layout.addLayout(ctrl_layout)
        main_layout.addLayout(right_layout, 1)
        # 初始化数据
        self.idx = 0
        self.is_playing = False
        self.init_plot_objects()
        self.draw_all(self.idx)
        # 定时器
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.refresh)
        self.timer.start(100)

    def init_plot_objects(self):
        self.bone_idx = [(joint_indices.index(a), joint_indices.index(b)) for a, b in bones]
        self.stick_lines = []
        for a, b in self.bone_idx:
            line, = self.ax_stick.plot([], [], 'k-', lw=3)
            self.stick_lines.append(line)
        self.joints_scatter = self.ax_stick.scatter([], [], c='r')
        self.head_patch = matplotlib.patches.Circle((0,0), 0.18, color='orange', fill=False, lw=3)
        self.ax_stick.add_patch(self.head_patch)
        self.info_text = self.ax_stick.text(
            0.98, 0.98, "", fontsize=12, color='purple',
            bbox=dict(facecolor='white', alpha=0.7),
            ha='right', va='top', transform=self.ax_stick.transAxes
        )
        self.leg_line, = self.ax_curve.plot([], [], label='Leg', color='green')
        self.back_line, = self.ax_curve.plot([], [], label='Back', color='blue')
        self.arm_line, = self.ax_curve.plot([], [], label='Arm', color='magenta')
        self.vline_curve = self.ax_curve.axvline(0, color='red', linestyle='--', lw=2, alpha=0.7)
        self.ax_curve.legend()
        self.leg_angle_line, = self.ax_angle.plot([], [], 'o-', label='Leg Drive Angle', color='green')
        self.back_angle_line, = self.ax_angle.plot([], [], 'o-', label='Back Angle', color='blue')
        self.arm_angle_line, = self.ax_angle.plot([], [], 'o-', label='Arm Angle', color='magenta')
        self.vline_angle = self.ax_angle.axvline(0, color='red', linestyle='--', lw=2, alpha=0.7)
        self.ax_angle.legend()
        self.leg_angle_text = self.ax_stick.text(0, 0, "", color='green', fontsize=12, ha='center', va='bottom', zorder=11)
        self.back_angle_text = self.ax_stick.text(0, 0, "", color='blue', fontsize=12, ha='center', va='bottom', zorder=11)
        self.arm_angle_text = self.ax_stick.text(0, 0, "", color='magenta', fontsize=12, ha='center', va='bottom', zorder=11)

    def toggle_play(self):
        self.is_playing = self.play_btn.isChecked()
        self.play_btn.setText("暂停" if self.is_playing else "播放")

    def on_slider(self, val):
        self.idx = val
        self.draw_all(self.idx)

    def draw_all(self, idx):
        xs, ys = frames[idx]
        xs_n, ys_n = normalize(xs, ys)
        for i, (a, b) in enumerate(self.bone_idx):
            self.stick_lines[i].set_data([xs_n[a], xs_n[b]], [ys_n[a], ys_n[b]])
        self.joints_scatter.set_offsets(np.c_[xs_n, ys_n])
        left_shoulder = np.array([xs_n[joint_indices.index(11)], ys_n[joint_indices.index(11)]])
        right_shoulder = np.array([xs_n[joint_indices.index(12)], ys_n[joint_indices.index(12)]])
        neck = (left_shoulder + right_shoulder) / 2
        head_radius = 0.18
        head_center = neck + np.array([0, head_radius * 1.5])
        self.head_patch.center = head_center
        self.info_text.set_text(f"Time: {times[idx]:.2f}s\nPhase: {phase[idx]}\nSPM: {spm[idx]:.1f}")
        highlight_joints = {
            'Leg': joint_indices.index(23),
            'Back': joint_indices.index(11),
            'Arm': joint_indices.index(13)
        }
        def get_nearest_angle(tlist, alist):
            if not tlist or not alist: return None
            # 只考虑距离当前时间最近且非 None 的 angle
            valid = [(t, a) for t, a in zip(tlist, alist) if a is not None]
            if not valid: return None
            diffs = [abs(times[idx] - t) for t, a in valid]
            min_idx = np.argmin(diffs)
            return valid[min_idx][1]
        leg_val = get_nearest_angle(switch_time, leg_angle)
        back_val = get_nearest_angle(switch_time, back_angle)
        arm_val = get_nearest_angle(switch_time, arm_angle)
        if leg_val is not None:
            self.leg_angle_text.set_text(f"{leg_val:.1f}°")
            self.leg_angle_text.set_position((xs_n[highlight_joints['Leg']], ys_n[highlight_joints['Leg']] + 0.05))
        else:
            self.leg_angle_text.set_text("")
        if back_val is not None:
            self.back_angle_text.set_text(f"{back_val:.1f}°")
            self.back_angle_text.set_position((xs_n[highlight_joints['Back']], ys_n[highlight_joints['Back']] + 0.05))
        else:
            self.back_angle_text.set_text("")
        if arm_val is not None:
            self.arm_angle_text.set_text(f"{arm_val:.1f}°")
            self.arm_angle_text.set_position((xs_n[highlight_joints['Arm']], ys_n[highlight_joints['Arm']] + 0.05))
        else:
            self.arm_angle_text.set_text("")
        # 只显示指针附近 10 秒
        t_center = times[idx]
        t_min = t_center - 5
        t_max = t_center + 5
        # 曲线图
        mask_curve = [(t >= t_min and t <= t_max) for t in times]
        times_curve = np.array(times)[mask_curve]
        leg_curve = np.array(leg)[mask_curve]
        back_curve = np.array(back)[mask_curve]
        arm_curve = np.array(arm)[mask_curve]
        self.leg_line.set_data(times_curve, leg_curve)
        self.back_line.set_data(times_curve, back_curve)
        self.arm_line.set_data(times_curve, arm_curve)
        self.ax_curve.set_xlim(t_min, t_max)
        self.vline_curve.set_xdata([t_center])
        # 角度图
        mask_angle = [(t >= t_min and t <= t_max) for t in switch_time]
        switch_time_win = np.array(switch_time)[mask_angle]
        leg_angle_win = np.array(leg_angle)[mask_angle]
        back_angle_win = np.array(back_angle)[mask_angle]
        arm_angle_win = np.array(arm_angle)[mask_angle]
        self.leg_angle_line.set_data(switch_time_win, leg_angle_win)
        self.back_angle_line.set_data(switch_time_win, back_angle_win)
        self.arm_angle_line.set_data(switch_time_win, arm_angle_win)
        self.ax_angle.set_xlim(t_min, t_max)
        self.vline_angle.set_xdata([t_center])
        # 指标区块
        # 找到最近的 finish/catch 切换点（只用 <= 当前时间的最后一个）
        t_now = times[idx]
        finish_angles = [(t, a_leg, a_back, a_arm)
            for t, s, a_leg, a_back, a_arm 
            in zip(switch_time, switch_type, leg_angle, back_angle, arm_angle)
            if s and s.lower() == 'finish' and t <= t_now]
        catch_angles = [(t, a_leg, a_back, a_arm)
            for t, s, a_leg, a_back, a_arm 
            in zip(switch_time, switch_type, leg_angle, back_angle, arm_angle)
            if s and s.lower() == 'catch' and t <= t_now]

        finish_metrics = []
        catch_metrics = []
        if finish_angles:
            # 找到切换点在 frames 里的最近索引
            t_fin, leg_fin, back_fin, arm_fin = finish_angles[-1]
            idx_fin = np.argmin([abs(t - t_fin) for t in times])
            xs_fin, ys_fin = frames[idx_fin]
            xs_n_fin, ys_n_fin = normalize(xs_fin, ys_fin)
            sh_y = np.mean([ys_n_fin[joint_indices.index(11)], ys_n_fin[joint_indices.index(12)]])
            hp_y = np.mean([ys_n_fin[joint_indices.index(23)], ys_n_fin[joint_indices.index(24)]])
            wr_y = np.mean([ys_n_fin[joint_indices.index(15)], ys_n_fin[joint_indices.index(16)]])
            handle_height_ratio = (wr_y - hp_y) / (sh_y - hp_y + 1e-6) * 100
            finish_metrics = [
                ("手臂夹脚", f"{back_fin:.0f}", 110, 1125, "°"),
                ("背部后倾", f"{leg_fin:.0f}", 200, 220, "°"),
                ("握桨高度", f"{handle_height_ratio:.0f}", 32, 55, "%")
            ]
        if catch_angles:
            t_cat, leg_cat, back_cat, arm_cat = catch_angles[-1]
            catch_metrics = [
                ("背部夹脚", f"{leg_cat:.0f}", 240, 260, "°"),
                ("前倾角度", f"{back_cat:.0f}", 25, 45, "°"),
                ("肘部弯曲", f"{arm_cat:.0f}", 160, 176, "°")
            ]
        if finish_metrics or catch_metrics:
            self.metrics.update_metrics(finish_metrics, catch_metrics)
        else:
            self.metrics.show_nodata()
        self.ax_stick.relim()
        self.ax_stick.autoscale_view()
        self.ax_curve.relim()
        self.ax_curve.autoscale_view()
        self.ax_angle.relim()
        self.ax_angle.autoscale_view()
        self.ax_curve.set_ylim(-25, 200)
        # AI 智能建议
        suggestions = []
        # 遍历所有metrics，直接用区间判断
        all_metrics = []
        if finish_metrics:
            all_metrics += finish_metrics
        if catch_metrics:
            all_metrics += catch_metrics
        for name, value, low, high, unit in all_metrics:
            try:
                val = float(value)
            except Exception:
                continue
            # 针对不同指标给出更自然的建议
            if name == "手臂夹脚":
                if val < low:
                    suggestions.append("手臂夹脚：夹得不够，建议手臂再靠近身体")
                elif val > high:
                    suggestions.append("手臂夹脚：夹得过多，建议手臂适当外展")
                else:
                    suggestions.append("手臂夹脚：动作良好")
            elif name == "背部后倾":
                if val < low:
                    suggestions.append("背部后倾：后倾不足，建议加大后倾幅度")
                elif val > high:
                    suggestions.append("背部后倾：后倾过大，建议减少后倾")
                else:
                    suggestions.append("背部后倾：动作良好")
            elif name == "握桨高度":
                if val < low:
                    suggestions.append("握桨高度：手部过低，建议抬高手部")
                elif val > high:
                    suggestions.append("握桨高度：手部过高，建议降低手部")
                else:
                    suggestions.append("握桨高度：动作良好")
            elif name == "背部夹脚":
                if val < low:
                    suggestions.append("背部夹脚：夹得不够，建议背部再靠近腿部")
                elif val > high:
                    suggestions.append("背部夹脚：夹得过多，建议背部适当外展")
                else:
                    suggestions.append("背部夹脚：动作良好")
            elif name == "前倾角度":
                if val < low:
                    suggestions.append("前倾角度：前倾不足，建议加大前倾")
                elif val > high:
                    suggestions.append("前倾角度：前倾过大，建议减少前倾")
                else:
                    suggestions.append("前倾角度：动作良好")
            elif name == "肘部弯曲":
                if val < low:
                    suggestions.append("肘部弯曲：弯曲不足，建议加大肘部弯曲")
                elif val > high:
                    suggestions.append("肘部弯曲：弯曲过大，建议减少肘部弯曲")
                else:
                    suggestions.append("肘部弯曲：动作良好")
            else:
                # 其它指标默认
                if val < low:
                    suggestions.append(f"{name}：数值偏低")
                elif val > high:
                    suggestions.append(f"{name}：数值偏高")
                else:
                    suggestions.append(f"{name}：动作良好")
        # 汇总显示
        self.suggestion_label.setText("\n".join(suggestions))
        self.canvas.draw_idle()

    def refresh(self):
        if self.is_playing:
            self.idx += 1
            if self.idx >= len(frames):
                self.idx = 0
            self.slider.setValue(self.idx)
        else:
            self.draw_all(self.idx)

# ----------- 启动入口 -------------
if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    win = MainWindow()
    win.show()
    app.exec_()