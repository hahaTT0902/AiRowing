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
    log_writer.writerow([
        'Time', 'Leg Movement', 'Back Movement', 'Arm Movement', 'Phase', 'SPM',
        'Switch', 'leg_drive_angle', 'back_angle', 'arm_angle'
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

            angles = get_relevant_angles(joints)
            shoulder = joints[11]
            hip = joints[23]
            wrist = joints[15]

            leg_move = relative_movement(prev_hip, hip)
            back_move = relative_movement(prev_shoulder, shoulder)
            arm_move = relative_movement(prev_wrist, wrist)
            prev_hip, prev_shoulder, prev_wrist = hip, shoulder, wrist

            t = time.time() - start_time
            time_series.append(t)
            smooth_append(leg_series, leg_move)
            smooth_append(back_series, back_move)
            smooth_append(arm_series, arm_move)

            stroke_phase, stroke_count, spm, switch = tracker.update(wrist[0], t, angles)
            phase_labels.append(stroke_phase)
            if len(phase_spans) == 0 or phase_spans[-1][1] != stroke_phase:
                phase_spans.append((t, stroke_phase))
            if len(phase_spans) > 200:
                phase_spans.pop(0)

            # 统一写入所有数据
            log_writer.writerow([
                t, leg_move, back_move, arm_move, stroke_phase, spm,
                switch if switch is not None else '',
                angles.get('leg_drive_angle', ''),
                angles.get('back_angle', ''),
                angles.get('arm_angle', '')
            ])

            # 绘图逻辑每10帧刷新一次
            frame_count += 1
            if frame_count % 10 == 0:
                ax1.cla()
                ax2.cla()
                t_min = max(time_series[0], time_series[-1] - 10)
                ax1.set_xlim(t_min, time_series[-1])
                ax1.plot(time_series, leg_series, label='Buttocks', color='green')
                ax1.plot(time_series, back_series, label='Back', color='blue')
                ax1.plot(time_series, arm_series, label='Arms', color='magenta')
                for i in range(1, len(phase_spans)):
                    t_start, phase = phase_spans[i - 1]
                    t_end = phase_spans[i][0]
                    color = '#ffe6cc' if phase == 'Drive' else '#e6f2ff'
                    ax1.axvspan(t_start, t_end, facecolor=color, alpha=0.3)
                ax1.legend()
                ax1.set_title("Real-Time Movement")

                # 角度切换图
                if toggle_angles:
                    times = [x[0] for x in toggle_angles]
                    for name in ['leg_drive_angle', 'back_angle', 'arm_angle']:
                        values = [a[2].get(name, 0) for a in toggle_angles]
                        ax2.plot(times, values, label=name)
                    ax2.legend()
                    ax2.set_title("Angle at Phase Switch")

                plt.pause(0.001)

            # overlay
            cv2.putText(frame, f"Phase: {stroke_phase}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
            cv2.putText(frame, f"Strokes: {stroke_count}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
            cv2.putText(frame, f"SPM: {spm:.1f}", (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

            for name, joint_ids in [('back_angle', (11, 23, 25)), ('leg_drive_angle', (23, 25, 27)), ('arm_angle', (11, 13, 15))]:
                if name in angles:
                    p1, p2, p3 = joints[joint_ids[0]], joints[joint_ids[1]], joints[joint_ids[2]]
                    cv2.line(frame, p1, p2, (0, 255, 0), 2)
                    cv2.line(frame, p2, p3, (0, 255, 0), 2)
                    angle = angles[name]
                    cv2.putText(frame, f"{int(angle)}°", (p2[0] + 10, p2[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 5)

        cv2.imshow("Rowing Technique", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    release_video_capture(cap)
    plt.ioff()
    plt.close('all')
    log_f.close()
    print("\n程序结束")

if __name__ == '__main__':
    main()
