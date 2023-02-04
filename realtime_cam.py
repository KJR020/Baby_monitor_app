import time
import torch

import cv2

from ultralytics import YOLO
from ultralytics.yolo.utils import DEFAULT_CFG


def get_frame(cap):
    ret, frame = cap.read()
    if ret:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    else:
        return None


def display_results(frame, results, classes=[]):

    boxes = results[0].boxes
    for box in boxes.xywh:
        x, y, w, h = box
        cv2.rectangle(
            frame,
            (int(x - w / 2), int(y - h / 2)),
            (int(x + w / 2), int(y + h / 2)),
            (255, 0, 0),
        )

        # cv2.putText(
        #     frame,
        #     str(classes[int(cat)]) + " [" + str(score) + "]",
        #     (int(x), int(y)),
        #     cv2.FONT_HERSHEY_SIMPLEX,
        #     1,
        #     (255, 255, 0),
        #     2,
        # )
    cv2.imshow("Object Detection", frame)


def put_texts(frame, texts, font_scale=0.7, color=(255, 255, 255), thickness=1):
    h, w, c = frame.shape
    offset_x = 10
    initial_y = 0
    dy = int(40 * font_scale)

    texts = [texts] if isinstance(texts, str) else texts

    for i, text in enumerate(texts):
        offset_y = initial_y + (i + 1) * dy
        cv2.putText(
            frame,
            text,
            (offset_x, offset_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA,
        )


cap = cv2.VideoCapture("rtsp://192.168.11.5:8554/video0_unicast")
model = YOLO("yolov8n.pt")
model.model.cuda()

results = model(stream="rtsp://192.168.11.5:8554/video0_unicast", show=True)

for i in enumerate(results):
    print(i)

# while True:
#     start = time.time()
#     frame = get_frame(cap)
#     if frame is None:
#         break

#     frame_capture_duration = (time.time() - start) * 1000

#     start = time.time()

#     results = model(source=frame)

#     display_results(frame, results)

#     write_rectangle_duration = (time.time() - start) * 1000

#     debug_info = [
#         f"frame_capture_duration: {frame_capture_duration:.3f}ms",
#         f"write_rectangle_duration: {write_rectangle_duration:.3f}ms",
#     ]

#     put_texts(frame, debug_info)

#     cv2.imshow("Object Detection", frame)

#     key = cv2.waitKey(1)
#     if key == 27:
#         break
