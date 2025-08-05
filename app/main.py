import os
import sys
import cv2
import mediapipe as mp
from utils.pose_utils import get_relevant_angles
from utils.video_stream import setup_video_capture, release_video_capture



# Append project root directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

def calculate_deviation(angle, ideal_range):
    """Calculate the deviation score based on an ideal range."""
    if ideal_range[0] <= angle <= ideal_range[1]:
        return 100  # Perfect alignment
    deviation = min(abs(angle - ideal_range[0]), abs(angle - ideal_range[1]))
    return max(0, 100 - (deviation * 2))  # Adjust multiplier as needed

def analyze_frame(frame):
    """Analyze the frame to extract angles and joints using MediaPipe Pose."""
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(rgb_frame)

    joints = {}
    if result.pose_landmarks:
        # Draw landmarks and connections
        mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        # Extract joint positions
        for idx, landmark in enumerate(result.pose_landmarks.landmark):
            joints[idx] = (int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0]))

        # Calculate angles based on relevant joints
        angles = get_relevant_angles(joints)
        return frame, angles, joints
    return frame, None, None

def main():
    print("Setting up video capture...")
    cap = setup_video_capture()
    
    if not cap.isOpened():
        print("Failed to open video capture.")
        return

    print("Video capture started.")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame.")
            break

        # Analyze frame to get angles and joint positions
        frame, angles, joints = analyze_frame(frame)
        if angles and joints:
            # Calculate deviation scores for each angle
            back_score = calculate_deviation(angles['back_angle'], (90, 120))
            leg_drive_score = calculate_deviation(angles['leg_drive_angle'], (165, 180))
            arm_score = calculate_deviation(angles['arm_angle'], (160, 180))

            # Define relevant joints for angle visualization
            shoulder = joints[11]  # Example joint IDs
            hip = joints[23]
            knee = joints[25]
            elbow = joints[13]
            wrist = joints[15]

            # Draw lines for back angle
            cv2.line(frame, shoulder, hip, (0, 255, 255), 2)  # Back angle in yellow
            cv2.putText(frame, f"{angles['back_angle']:.2f}", shoulder, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

            # Draw lines for leg drive angle
            cv2.line(frame, hip, knee, (255, 0, 0), 2)  # Leg drive angle in blue
            cv2.putText(frame, f"{angles['leg_drive_angle']:.2f}", hip, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # Draw lines for arm angle
            cv2.line(frame, shoulder, elbow, (0, 255, 0), 2)  # Arm angle in green
            cv2.line(frame, elbow, wrist, (0, 255, 0), 2)
            cv2.putText(frame, f"{angles['arm_angle']:.2f}", elbow, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Display each score with color indicators
            cv2.putText(frame, f"Back Score: {back_score}%", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0) if back_score == 100 else (0, 0, 255), 2)
            cv2.putText(frame, f"Leg Drive Score: {leg_drive_score}%", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0) if leg_drive_score == 100 else (0, 0, 255), 2)
            cv2.putText(frame, f"Arm Score: {arm_score}%", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0) if arm_score == 100 else (0, 0, 255), 2)

            # Calculate and display an overall form score
            overall_score = (back_score + leg_drive_score + arm_score) // 3
            cv2.putText(frame, f"Overall Score: {overall_score}%", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0) if overall_score > 80 else (0, 0, 255), 2)

        cv2.imshow('Rowing Technique Analysis', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print("Releasing video capture...")
    release_video_capture(cap)
    print("Program finished.")

if __name__ == '__main__':
    main()
