import cv2
import darknet


def get_frame(cap):
    ret, frame = cap.read()
    if ret:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    else:
        return None


cap = cv2.VideoCapture("rtsp://192.168.11.5:8554/video0_unicast")

while True:
    frame = get_frame(cap)
    if frame is None:
        break

    cv2.putText(
        frame,
        f"{debug_info}",
        (20, 150),
        cv2.FONT_HERSHEY_SIMPLEX,
        4,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.rectangle(frame, (120, 220), (270, 260), (20, 20, 20), -1)

    cv2.imshow("Object Detection", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break
