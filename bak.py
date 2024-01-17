from ultralytics import YOLO
import cv2
import math
import torch

torch.cuda.set_device(0)

# start webcam
cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture("pexels-george-morina-5962271 (1080p).mp4")
frame_width = 640
frame_height = 480
cap.set(3, frame_width)
cap.set(4, frame_height)

model = YOLO("yolov8m.pt")
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
color = (
    (255, 0, 0),
    (
        0,
        255,
        0,
    ),
    (0, 0, 255),
    (255, 255, 255),
)  # red / green / blue / white
thickness = 2
detection_area = (
    (int(frame_width / 2) - 20, 0),
    (int(frame_width / 2) + 20, frame_height),
)

passing = 0

while True:
    _, frame = cap.read()
    results = model(
        frame,
        conf=0.6,
        stream=True,
    )

    for r in results:
        boxes = r.boxes

        cv2.line(
            frame,
            (int(frame_width / 2) - 20, 0),
            (int(frame_width / 2) - 20, frame_height),
            color=color[0],
            thickness=thickness,
        )

        cv2.line(
            frame,
            (int(frame_width / 2) + 20, 0),
            (int(frame_width / 2) + 20, frame_height),
            color=color[0],
            thickness=thickness,
        )

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # convert to int values

            confidence = math.ceil((box.conf[0] * 100)) / 100
            class_id = int(box.cls[0])  # 0 = person

            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            if class_id == 0:
                cv2.rectangle(
                    frame, (x1, y1), (x2, y2), color=color[1], thickness=thickness
                )
                cv2.circle(
                    frame, (cx, cy), radius=1, color=color[0], thickness=thickness
                )

                cv2.putText(
                    frame,
                    f"person passing {passing}",
                    (x1, y1 - 2),
                    font,
                    font_scale,
                    color=color[3],
                    thickness=thickness,
                )

                if cx >= int(frame_width / 2 - 20) and cx <= int(frame_width / 2 + 20):
                    passing += 1

    cv2.imshow("Webcam", frame)
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
