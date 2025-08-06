import cv2

def setup_video_capture():
    cap = cv2.VideoCapture(0)  # 0 为默认摄像头，1 为第二个摄像头
    return cap

def release_video_capture(cap):
    cap.release()
    cv2.destroyAllWindows()
