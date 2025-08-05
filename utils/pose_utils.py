import math

def calculate_angle(p1, p2, p3):
    angle = math.degrees(math.atan2(p3[1] - p2[1], p3[0] - p2[0]) -
                         math.atan2(p1[1] - p2[1], p1[0] - p2[0]))
    return abs(angle) if angle >= 0 else abs(angle + 360)

def angle_with_horizontal(p1, p2):
    """Calculate the angle between a vector (p1 to p2) and the horizontal axis."""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    angle = math.degrees(math.atan2(dy, dx))
    return angle

def vertical_angle(p1, p2):
    """Angle between the line p1-p2 and the vertical axis."""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    angle = math.degrees(math.atan2(dx, dy))
    return abs(angle)

def get_relevant_angles(joints):
    """Calculate multiple biomechanical angles for rowing analysis."""
    angles = {
        'back_angle': calculate_angle(joints[11], joints[23], joints[25]),  # shoulder-hip-knee
        'leg_drive_angle': calculate_angle(joints[23], joints[25], joints[27]),  # hip-knee-ankle
        'arm_angle': calculate_angle(joints[11], joints[13], joints[15]),  # shoulder-elbow-wrist
        'elbow_angle': calculate_angle(joints[13], joints[15], (joints[15][0]+10, joints[15][1])),  # elbow-wrist-horizontal
        'shins_angle': vertical_angle(joints[27], joints[25]),  # ankle-knee relative to vertical
        'torso_angle': angle_with_horizontal(joints[11], joints[23]),  # shoulder-hip relative to horizontal
        'handle_height': joints[15][1] - joints[11][1]  # wrist y - shoulder y (pixel difference)
    }
    return angles
