import cv2

def setup_video_capture():
    cap = cv2.VideoCapture(0)  # 0 for the default webcam
    return cap

def release_video_capture(cap):
    cap.release()
    cv2.destroyAllWindows()
