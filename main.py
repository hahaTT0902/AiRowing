import os
import sys
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from matplotlib.animation import FuncAnimation
from utils.pose_utils import get_relevant_angles
from utils.video_stream import setup_video_capture, release_video_capture

# 添加平滑函数
def smooth_append(series, value, alpha=0.3):
    if not series:
        series.append(value)
    else:
        smoothed = alpha * value + (1 - alpha) * series[-1]
        series.append(smoothed)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import mediapipe as mp
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# 阶段追踪
class StrokeStateTracker:
    def __init__(self):
        self.state = "Unknown"
        self.previous_wrist_x = None

    def update(self, wrist_x):
        if self.previous_wrist_x is None:
            self.previous_wrist_x = wrist_x
            return self.state
        dx = wrist_x - self.previous_wrist_x
        self.previous_wrist_x = wrist_x
        if dx < -5:
            self.state = "Drive"
        elif dx > 5:
            self.state = "Recovery"
        return self.state

def relative_movement(p1, p2):
    if p1 is None or p2 is None:
        return 0.0
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return np.sqrt(dx**2 + dy**2)

# 缓存数据
time_series = deque(maxlen=100)
leg_series = deque(maxlen=100)
back_series = deque(maxlen=100)
arm_series = deque(maxlen=100)
phase_labels = deque(maxlen=100)

phase_spans = []
last_phase = None

# 主程序
def main():
    cap = setup_video_capture()
    if not cap.isOpened():
        print("Camera failed to open.")
        return

    tracker = StrokeStateTracker()
    prev_hip = prev_shoulder = prev_wrist = None
    start_time = time.time()

    plt.ion()
    fig, ax = plt.subplots()
    ax.set_ylim(0, 100)
    ax.set_title("Rowing Technique Analysis")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Movement (px)")
    leg_line, = ax.plot([], [], label='Buttocks', color='green')
    back_line, = ax.plot([], [], label='Back', color='blue')
    arm_line, = ax.plot([], [], label='Arms', color='magenta')
    ax.legend(loc='upper right')

    def update_plot(_):
        ax.clear()
        if time_series:
            t_max = time_series[-1]
            t_min = max(time_series[0], t_max - 10)
            idx_start = 0
            for i, t in enumerate(time_series):
                if t >= t_min:
                    idx_start = i
                    break
            ts = list(time_series)[idx_start:]
            legs = list(leg_series)[idx_start:]
            backs = list(back_series)[idx_start:]
            arms = list(arm_series)[idx_start:]
            phases = list(phase_labels)[idx_start:]

            for i in range(1, len(ts)):
                if phases[i] != phases[i - 1]:
                    phase_spans.append((ts[i], phases[i]))

            current_bg = "#e6f2ff"
            if phases:
                current_bg = "#ffe6cc" if phases[-1] == "Drive" else "#e6f2ff"

            last_span_time = t_min
            for i in range(len(phase_spans)):
                span_time, phase = phase_spans[i]
                if span_time > t_max:
                    break
                color = '#ffe6cc' if phase == 'Drive' else '#e6f2ff'
                ax.axvspan(last_span_time, span_time, facecolor=color, alpha=0.3, edgecolor='none')
                last_span_time = span_time
            ax.axvspan(last_span_time, t_max, facecolor=current_bg, alpha=0.3, edgecolor='none')

            ax.plot(ts, legs, label='Buttocks', color='green')
            ax.plot(ts, backs, label='Back', color='blue')
            ax.plot(ts, arms, label='Arms', color='magenta')
            ax.set_xlim(t_min, t_max)
            max_val = max(max(legs, default=0), max(backs, default=0), max(arms, default=0))
            ax.set_ylim(0, max(20, max_val + 5))
            ax.set_title("Rowing Technique Analysis")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Movement (px)")
            ax.legend(loc='upper right')
        return leg_line, back_line, arm_line

    ani = FuncAnimation(fig, update_plot, interval=30)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        for (x, y, w, h) in faces:
            face_roi = frame[y:y+h, x:x+w]
            blurred_face = cv2.GaussianBlur(face_roi, (99, 99), 30)
            frame[y:y+h, x:x+w] = blurred_face

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

            stroke_phase = tracker.update(wrist[0])
            phase_labels.append(stroke_phase)

            phase_color = (0, 255, 255) if stroke_phase == "Drive" else (255, 255, 255)
            cv2.putText(frame, f"Phase: {stroke_phase}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, phase_color, 6)

            for name, joints_ids in [('back_angle', (11, 23, 25)), ('leg_drive_angle', (23, 25, 27)), ('arm_angle', (11, 13, 15))]:
                if name in angles:
                    p1, p2, p3 = joints[joints_ids[0]], joints[joints_ids[1]], joints[joints_ids[2]]
                    cv2.line(frame, p1, p2, (0, 255, 0), 2)
                    cv2.line(frame, p2, p3, (0, 255, 0), 2)
                    angle = angles[name]
                    cv2.putText(
                        frame,
                        f"{int(angle)}°",
                        (p2[0]+10, p2[1]-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        2,
                        (255, 255, 255),
                        4
                    )

        cv2.imshow("Rowing Technique", frame)
        # plt.pause(0.001)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    release_video_capture(cap)
    plt.ioff()
    plt.close()
    print("程序结束")

if __name__ == '__main__':
    main()
