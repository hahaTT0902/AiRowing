import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import threading
import time
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
            phase.append(row['Phase'])
            if row['Switch']:
                switch_time.append(float(row['Time']))
                switch_type.append(row['Switch'])
                leg_angle.append(float(row['leg_drive_angle']) if row['leg_drive_angle'] else None)
                back_angle.append(float(row['back_angle']) if row['back_angle'] else None)
                arm_angle.append(float(row['arm_angle']) if row['arm_angle'] else None)
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
    scale = max(np.max(np.abs(xs)), np.max(np.abs(ys)), 1e-6)
    xs = xs / scale
    ys = ys / scale
    return xs, ys

fig = plt.figure(figsize=(14, 7))
gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.2], height_ratios=[1, 1])
ax_stick = fig.add_subplot(gs[:, 0])
ax_curve = fig.add_subplot(gs[0, 1])
ax_angle = fig.add_subplot(gs[1, 1])
ax_slider = fig.add_axes([0.15, 0.03, 0.7, 0.03])  # 底部添加slider

bone_idx = [(joint_indices.index(a), joint_indices.index(b)) for a, b in bones]

# 1. 初始化所有可变对象（只创建一次）
stick_lines = []
for a, b in bone_idx:
    line, = ax_stick.plot([], [], 'k-', lw=3)
    stick_lines.append(line)
joints_scatter = ax_stick.scatter([], [], c='r')
head_patch = plt.Circle((0,0), 0.18, color='orange', fill=False, lw=3)
ax_stick.add_patch(head_patch)
info_text = ax_stick.text(
    0.98, 0.98, "", fontsize=12, color='purple',
    bbox=dict(facecolor='white', alpha=0.7),
    ha='right', va='top', transform=ax_stick.transAxes
)

leg_line, = ax_curve.plot([], [], label='Leg', color='green')
back_line, = ax_curve.plot([], [], label='Back', color='blue')
arm_line, = ax_curve.plot([], [], label='Arm', color='magenta')
vline_curve = ax_curve.axvline(0, color='red', linestyle='--', lw=2, alpha=0.7)
ax_curve.legend()

leg_angle_line, = ax_angle.plot([], [], 'o-', label='Leg Drive Angle', color='green')
back_angle_line, = ax_angle.plot([], [], 'o-', label='Back Angle', color='blue')
arm_angle_line, = ax_angle.plot([], [], 'o-', label='Arm Angle', color='magenta')
vline_angle = ax_angle.axvline(0, color='red', linestyle='--', lw=2, alpha=0.7)
ax_angle.legend()

# 在初始化时创建角度文本对象
leg_angle_text = ax_stick.text(0, 0, "", color='green', fontsize=12, ha='center', va='bottom', zorder=11)
back_angle_text = ax_stick.text(0, 0, "", color='blue', fontsize=12, ha='center', va='bottom', zorder=11)
arm_angle_text = ax_stick.text(0, 0, "", color='magenta', fontsize=12, ha='center', va='bottom', zorder=11)

def draw_stickman(ax, xs, ys):
    ax.cla()
    xs_n, ys_n = normalize(xs, ys)
    for a, b in bone_idx:
        ax.plot([xs_n[a], xs_n[b]], [ys_n[a], ys_n[b]], 'k-', lw=3)
    ax.scatter(xs_n, ys_n, c='r')
    left_shoulder = np.array([xs_n[joint_indices.index(11)], ys_n[joint_indices.index(11)]])
    right_shoulder = np.array([xs_n[joint_indices.index(12)], ys_n[joint_indices.index(12)]])
    neck = (left_shoulder + right_shoulder) / 2
    head_radius = 0.18
    head_center = neck + np.array([0, head_radius * 1.5])
    head = plt.Circle(head_center, head_radius, color='orange', fill=False, lw=3)
    ax.add_patch(head)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1.5, 1.5)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('2D Stickman')

def draw_curves(ax, t, leg, back, arm, cur_idx):
    ax.cla()
    ax.plot(t, leg, label='Leg', color='green')
    ax.plot(t, back, label='Back', color='blue')
    ax.plot(t, arm, label='Arm', color='magenta')
    ax.axvline(t[cur_idx], color='red', linestyle='--', lw=2, alpha=0.7)
    ax.set_ylabel('Movement (px)')
    ax.set_title('Rowing Movement Curve')
    ax.legend()

def draw_angles(ax, switch_time, leg_angle, back_angle, arm_angle, cur_time):
    ax.cla()
    ax.plot(switch_time, leg_angle, 'o-', label='Leg Drive Angle', color='green')
    ax.plot(switch_time, back_angle, 'o-', label='Back Angle', color='blue')
    ax.plot(switch_time, arm_angle, 'o-', label='Arm Angle', color='magenta')
    for t, s in zip(switch_time, switch_type):
        if s:
            ax.axvline(t, color='gray', linestyle='--', alpha=0.3)
            ax.text(t, ax.get_ylim()[1], s, rotation=90, va='top', fontsize=8, alpha=0.6)
    ax.axvline(cur_time, color='red', linestyle='--', lw=2, alpha=0.7)
    ax.set_ylabel('Angle (°)')
    ax.set_title('Angles at Phase Switch')
    ax.legend()

def draw_all(idx):
    xs, ys = frames[idx]
    xs_n, ys_n = normalize(xs, ys)
    for i, (a, b) in enumerate(bone_idx):
        stick_lines[i].set_data([xs_n[a], xs_n[b]], [ys_n[a], ys_n[b]])
    joints_scatter.set_offsets(np.c_[xs_n, ys_n])
    left_shoulder = np.array([xs_n[joint_indices.index(11)], ys_n[joint_indices.index(11)]])
    right_shoulder = np.array([xs_n[joint_indices.index(12)], ys_n[joint_indices.index(12)]])
    neck = (left_shoulder + right_shoulder) / 2
    head_radius = 0.18
    head_center = neck + np.array([0, head_radius * 1.5])
    head_patch.center = head_center
    info_text.set_text(f"Time: {times[idx]:.2f}s\nPhase: {phase[idx]}\nSPM: {spm[idx]:.1f}")

    highlight_joints = {
        'Leg': joint_indices.index(23),
        'Back': joint_indices.index(11),
        'Arm': joint_indices.index(13)
    }
    def get_nearest_angle(tlist, alist):
        if not tlist or not alist: return None
        diffs = [abs(times[idx] - t) for t in tlist]
        min_idx = np.argmin(diffs)
        return alist[min_idx] if alist[min_idx] is not None else None

    leg_val = get_nearest_angle(switch_time, leg_angle)
    back_val = get_nearest_angle(switch_time, back_angle)
    arm_val = get_nearest_angle(switch_time, arm_angle)

    # 只更新文本内容和位置，不新建/删除
    if leg_val is not None:
        leg_angle_text.set_text(f"{leg_val:.1f}°")
        leg_angle_text.set_position((xs_n[highlight_joints['Leg']], ys_n[highlight_joints['Leg']] + 0.05))
    else:
        leg_angle_text.set_text("")
    if back_val is not None:
        back_angle_text.set_text(f"{back_val:.1f}°")
        back_angle_text.set_position((xs_n[highlight_joints['Back']], ys_n[highlight_joints['Back']] + 0.05))
    else:
        back_angle_text.set_text("")
    if arm_val is not None:
        arm_angle_text.set_text(f"{arm_val:.1f}°")
        arm_angle_text.set_position((xs_n[highlight_joints['Arm']], ys_n[highlight_joints['Arm']] + 0.05))
    else:
        arm_angle_text.set_text("")

    window = min(21, len(times) // 2 * 2 + 1)  # 窗口必须为奇数且小于数据长度
    if len(times) >= window:
        leg_smooth = savgol_filter(leg, window_length=window, polyorder=3)
        back_smooth = savgol_filter(back, window_length=window, polyorder=3)
        arm_smooth = savgol_filter(arm, window_length=window, polyorder=3)
    else:
        leg_smooth = leg
        back_smooth = back
        arm_smooth = arm

    leg_line.set_data(times, leg_smooth)
    back_line.set_data(times, back_smooth)
    arm_line.set_data(times, arm_smooth)
    vline_curve.set_xdata([times[idx]])

    leg_angle_line.set_data(switch_time, leg_angle)
    back_angle_line.set_data(switch_time, back_angle)
    arm_angle_line.set_data(switch_time, arm_angle)
    vline_angle.set_xdata([times[idx]])

    ax_stick.relim()
    ax_stick.autoscale_view()
    ax_curve.relim()
    ax_curve.autoscale_view()
    ax_angle.relim()
    ax_angle.autoscale_view()
    ax_curve.set_ylim(-25, 200)

    fig.canvas.draw_idle()

slider = Slider(ax_slider, 'Frame', 0, len(frames)-1, valinit=0, valstep=1)

def on_slider(val):
    idx = int(slider.val)
    draw_all(idx)

slider.on_changed(on_slider)

is_playing = [False]  # 用列表包裹，方便闭包修改
play_thread = [None]

def play_loop():
    while is_playing[0]:
        cur_idx = int(slider.val)
        if cur_idx < len(frames) - 1:
            slider.set_val(cur_idx + 1)
            time.sleep(0.03)
        else:
            is_playing[0] = False

def on_key(event):
    if event.key == ' ':
        is_playing[0] = not is_playing[0]
        if is_playing[0]:
            # 启动新线程自动播放
            if play_thread[0] is None or not play_thread[0].is_alive():
                play_thread[0] = threading.Thread(target=play_loop, daemon=True)
                play_thread[0].start()

fig.canvas.mpl_connect('key_press_event', on_key)

draw_all(0)
plt.show()