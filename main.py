import os
import sys
import cv2
import time
import csv
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from utils.pose_utils import get_relevant_angles
from utils.video_stream import setup_video_capture, release_video_capture

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import mediapipe as mp
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

skeleton_pairs = [
    (11, 13), (13, 15),  # 左臂
    (12, 14), (14, 16),  # 右臂
    (11, 12),            # 双肩
    (11, 23), (12, 24),  # 躯干
    (23, 25), (25, 27),  # 左腿
    (24, 26), (26, 28),  # 右腿
    (23, 24)             # 骨盆
]

# 添加平滑函数
def smooth_append(series, value, alpha=0.3):
    if not series:
        series.append(value)
    else:
        smoothed = alpha * value + (1 - alpha) * series[-1]
        series.append(smoothed)

# 理想角度区间（用于反馈）
angle_ranges = {
    'leg_drive_angle': (60, 110),
    'back_angle': (20, 50),
    'arm_angle': (150, 170)
}

# 标准切换角度区间
switch_angle_ranges = {
    "Drive→Recovery": {
        "leg_drive_angle": (190, 220),
        "back_angle": (105, 135),
        "arm_angle": (80, 110),
    },
    "Recovery→Drive": {
        "leg_drive_angle": (275, 300),
        "back_angle": (20, 45),
        "arm_angle": (160, 180),
    }
}

# 阶段追踪，检测状态切换点
toggle_angles = []

class StrokeStateTracker:
    def __init__(self):
        self.state = "Unknown"
        self.previous_wrist_x = None
        self.last_state = "Unknown"
        self.stroke_count = 0
        self.stroke_timestamps = deque(maxlen=30)
        self.last_angles = {}
        self.stable_counter = 0
        self.stable_required = 3  # 连续帧数确认切换
        self.pending_state = None

    def update(self, wrist_x, current_time, angles):
        if self.previous_wrist_x is None:
            self.previous_wrist_x = wrist_x
            return self.state, self.stroke_count, 0.0, None

        dx = wrist_x - self.previous_wrist_x
        self.previous_wrist_x = wrist_x

        # 判断新状态
        new_state = self.state
        if dx < -5:
            candidate_state = "Drive"
        elif dx > 5:
            candidate_state = "Recovery"
        else:
            candidate_state = self.state

        switch = None
        # 防抖动切换
        if candidate_state != self.state:
            if self.pending_state == candidate_state:
                self.stable_counter += 1
            else:
                self.pending_state = candidate_state
                self.stable_counter = 1

            if self.stable_counter >= self.stable_required:
                new_state = candidate_state
                self.stable_counter = 0
                self.pending_state = None
        else:
            self.stable_counter = 0
            self.pending_state = None

        if new_state != self.last_state:
            if self.last_state == "Drive" and new_state == "Recovery":
                self.stroke_count += 1
                self.stroke_timestamps.append(current_time)
                switch = "Drive→Recovery"
            elif self.last_state == "Recovery" and new_state == "Drive":
                switch = "Recovery→Drive"

            # 记录切换时的角度
            toggle_angles.append((current_time, switch, angles.copy()))
            self.last_state = new_state

        self.state = new_state

        # SPM
        spm = 0.0
        if len(self.stroke_timestamps) >= 2:
            durations = [self.stroke_timestamps[i] - self.stroke_timestamps[i - 1] for i in range(1, len(self.stroke_timestamps))]
            avg_duration = sum(durations) / len(durations)
            spm = 60.0 / avg_duration if avg_duration > 0 else 0.0

        return new_state, self.stroke_count, spm, switch

# 相对位移
def relative_movement(p1, p2):
    if p1 is None or p2 is None:
        return 0.0
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return np.sqrt(dx ** 2 + dy ** 2)

# 初始化缓存
time_series = deque(maxlen=100)
leg_series = deque(maxlen=100)
back_series = deque(maxlen=100)
arm_series = deque(maxlen=100)
phase_labels = deque(maxlen=100)
phase_spans = []

# 新增：判断可见性函数
def get_joint_if_visible(joints, idx, threshold=0.5):
    vis = joints.get(f"{idx}_vis", 1.0)
    return joints[idx] if vis > threshold else None

# 主程序
def main():
    cap = setup_video_capture()
    if not cap.isOpened():
        print("Camera failed to open.")
        return

    tracker = StrokeStateTracker()
    prev_hip = prev_shoulder = prev_wrist = None
    start_time = time.time()
    frame_count = 0

    # 统一日志文件
    log_f = open('log.csv', 'w', newline='')
    log_writer = csv.writer(log_f)
    
    # 关节索引列表（不含头部）
    joint_indices = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]  # 右/左肩、肘、腕、髋、膝、踝

    # 写入csv表头
    log_writer.writerow([
        'Time', 'Phase', 'SPM', 'Switch',
        *[f'joint_{idx}_x' for idx in joint_indices],
        *[f'joint_{idx}_y' for idx in joint_indices],
        *[f'joint_{idx}_z' for idx in joint_indices],
        *[f'vec_{a}_{b}_dx' for a, b in skeleton_pairs],
        *[f'vec_{a}_{b}_dy' for a, b in skeleton_pairs],
        *[f'vec_{a}_{b}_dz' for a, b in skeleton_pairs],
        'Leg Movement', 'Back Movement', 'Arm Movement',
        'leg_drive_angle', 'back_angle', 'arm_angle'
    ])

    # 图表初始化
    plt.ion()
    fig1, ax1 = plt.subplots()
    ax1.set_title("Real-Time Movement")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Movement (px)")
    fig2, ax2 = plt.subplots()
    ax2.set_title("Angle at Phase Switch")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Angle (°)")

    last_feedback_msgs = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(rgb_frame)
        joints = {}

        if result.pose_landmarks:
            mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            for idx, landmark in enumerate(result.pose_landmarks.landmark):
                joints[idx] = (int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0]))
                joints[f"{idx}_vis"] = landmark.visibility

            angles = get_relevant_angles(joints)

            # 骨架高亮与角度显示
            for name, joint_ids in [
                ('back_angle', (12, 24, 26)),         # 右肩(12), 右髋(24), 右膝(26)
                ('leg_drive_angle', (24, 26, 28)),    # 右髋(24), 右膝(26), 右踝(28)
                ('arm_angle', (12, 14, 16))           # 右肩(12), 右肘(14), 右腕(16)
            ]:
                if name in angles and all(j in joints for j in joint_ids):
                    p1, p2, p3 = joints[joint_ids[0]], joints[joint_ids[1]], joints[joint_ids[2]]
                    cv2.line(frame, p1, p2, (0, 255, 0), 2)
                    cv2.line(frame, p2, p3, (0, 255, 0), 2)
                    angle = angles[name]
                    cv2.putText(frame, f"{int(angle)}°", (p2[0] + 10, p2[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 5)

            shoulder = get_joint_if_visible(joints, 12)
            hip = get_joint_if_visible(joints, 24)
            wrist = get_joint_if_visible(joints, 16)

            leg_move = relative_movement(prev_hip, hip)
            back_move = relative_movement(prev_shoulder, shoulder)
            arm_move = relative_movement(prev_wrist, wrist)

            # 只有在可见时才更新 prev_*
            if hip is not None:
                prev_hip = hip
            if shoulder is not None:
                prev_shoulder = shoulder
            if wrist is not None:
                prev_wrist = wrist

            t = time.time() - start_time
            time_series.append(t)
            smooth_append(leg_series, leg_move)
            smooth_append(back_series, back_move)
            smooth_append(arm_series, arm_move)

            if wrist is not None:
                wrist_x = wrist[0]
            else:
                wrist_x = 0  # 或者可以选择跳过本帧的 update

            stroke_phase, stroke_count, spm, switch = tracker.update(wrist_x, t, angles)
            phase_labels.append(stroke_phase)
            if len(phase_spans) == 0 or phase_spans[-1][1] != stroke_phase:
                phase_spans.append((t, stroke_phase))
            if len(phase_spans) > 200:
                phase_spans.pop(0)

            # 统一写入所有数据
            log_writer.writerow([
                t, stroke_phase, spm, switch if switch is not None else '',
                *[result.pose_landmarks.landmark[idx].x for idx in joint_indices],
                *[result.pose_landmarks.landmark[idx].y for idx in joint_indices],
                *[result.pose_landmarks.landmark[idx].z for idx in joint_indices],
                *[relative_movement(joints[a], joints[b]) for a, b in skeleton_pairs],
                *[joints[b][1] - joints[a][1] for a, b in skeleton_pairs],  # dy
                *[result.pose_landmarks.landmark[b].z - result.pose_landmarks.landmark[a].z for a, b in skeleton_pairs],  # dz
                leg_move, back_move, arm_move,
                angles.get('leg_drive_angle', 0),
                angles.get('back_angle', 0),
                angles.get('arm_angle', 0)
            ])


            if switch in switch_angle_ranges:
                phase_name = "Finish" if switch == "Drive→Recovery" else "Catch"
                feedback_msgs = []
                for name in ['leg_drive_angle', 'back_angle', 'arm_angle']:
                    angle = angles.get(name)
                    minv, maxv = switch_angle_ranges[switch][name]
                    if angle is None:
                        msg = f"{phase_name}: {name} Unknown"
                    elif minv <= angle <= maxv:
                        msg = f"{phase_name}: {name} OK"
                    elif angle < minv:
                        msg = f"{phase_name}: {name} Too Small"
                    else:
                        msg = f"{phase_name}: {name} Too Large"
                    feedback_msgs.append(msg)
                last_feedback_msgs = feedback_msgs  # 更新提示

        # 状态信息
        cv2.putText(frame, f"Phase: {stroke_phase}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (0, 255, 255), 6)
        cv2.putText(frame, f"Strokes: {stroke_count}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (255, 255, 255), 6)
        cv2.putText(frame, f"SPM: {spm:.1f}", (10, 220), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (0, 255, 0), 6)

        # 每帧都显示上一次切换的提示
        if last_feedback_msgs:
            for i, msg in enumerate(last_feedback_msgs):
                # 如果有问题（不是OK），用红色，否则绿色
                color = (0, 0, 255) if ("Too" in msg or "Unknown" in msg) else (0, 255, 0)
                cv2.putText(frame, msg, (10, 400 + i*80), cv2.FONT_HERSHEY_SIMPLEX, 2.5, color, 6)

        cv2.imshow("Rowing Technique", frame)
        frame_count += 1
        if frame_count % 10 == 0:
            ax1.cla()
            ax2.cla()

            # 只显示最近10秒的数据
            t_now = time_series[-1] if time_series else 0
            t_min = max(time_series[0], t_now - 10) if time_series else 0

            # 找到最近10秒的索引
            indices = [i for i, t in enumerate(time_series) if t >= t_min]

            # 动作幅度曲线
            ax1.set_title("Real-Time Movement")
            ax1.set_xlabel("Time (s)")
            ax1.set_ylabel("Movement (px)")
            ax1.plot([time_series[i] for i in indices], [leg_series[i] for i in indices], label='Buttocks', color='green')
            ax1.plot([time_series[i] for i in indices], [back_series[i] for i in indices], label='Back', color='blue')
            ax1.plot([time_series[i] for i in indices], [arm_series[i] for i in indices], label='Arms', color='magenta')
            # 阶段区间
            for i in range(1, len(phase_spans)):
                t_start, phase = phase_spans[i - 1]
                t_end = phase_spans[i][0]
                if t_end < t_min:
                    continue
                ax1.axvspan(max(t_start, t_min), t_end, facecolor='#ffe6cc' if phase == 'Drive' else '#e6f2ff', alpha=0.3)
            ax1.set_xlim(t_min, t_now)
            ax1.legend()

            # 切换瞬间角度图
            if toggle_angles:
                times = [x[0] for x in toggle_angles if x[0] >= t_min]
                for name in ['leg_drive_angle', 'back_angle', 'arm_angle']:
                    values = [a[2].get(name, 0) for a in toggle_angles if a[0] >= t_min]
                    ax2.plot(times, values, label=name)
                ax2.set_title("Angle at Phase Switch")
                ax2.set_xlabel("Time (s)")
                ax2.set_ylabel("Angle (°)")
                ax2.set_xlim(t_min, t_now)
                ax2.legend()

            plt.pause(0.001)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    release_video_capture(cap)
    plt.ioff()
    plt.close('all')
    log_f.close()
    print("\n程序结束")

if __name__ == '__main__':
    main()
