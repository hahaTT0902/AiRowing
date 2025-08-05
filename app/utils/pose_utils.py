import math

def calculate_angle(p1, p2, p3):
    """Calculate the angle at p2 between points p1 and p3."""
    angle = math.degrees(math.atan2(p3[1] - p2[1], p3[0] - p2[0]) - 
                         math.atan2(p1[1] - p2[1], p1[0] - p2[0]))
    return abs(angle) if angle >= 0 else abs(angle + 360)

def get_relevant_angles(joints):
    """Calculate back, leg drive, and arm angles based on joint positions."""
    angles = {
        'back_angle': calculate_angle(joints[11], joints[23], joints[25]),  # Back angle
        'leg_drive_angle': calculate_angle(joints[23], joints[25], joints[27]),  # Leg drive angle
        'arm_angle': calculate_angle(joints[11], joints[13], joints[15])  # Arm angle
    }
    return angles
