from ultralytics import YOLO
import supervision as sv
import cv2
import torch

torch.cuda.set_device(0)

cap = cv2.VideoCapture("enter video source")
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap.set(3, frame_width)
cap.set(4, frame_height)

model = YOLO("models/yolov8m.pt")
bounding_box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()
tracker = sv.ByteTrack()

# vertical line
# LINE_START = sv.Point(0, int(frame_height / 2))
# LINE_END = sv.Point(frame_width, int(frame_height / 2))

# horizontal line
LINE_START = sv.Point(int(frame_width / 2), 0)
LINE_END = sv.Point(int(frame_width / 2), frame_height)

line_counter = sv.LineZone(LINE_START, LINE_END)
line_counter_ann = sv.LineZoneAnnotator()

while True:
    _, frame = cap.read()
    results = model(frame, conf=0.5, stream=True, verbose=False)
    results = list(results)[0]

    detections = sv.Detections.from_ultralytics(results)
    detections = detections[detections.class_id == 0]
    detections = tracker.update_with_detections(detections)

    labels = [
        f"#{tracker_id} {results.names[class_id]}"
        for class_id, tracker_id in zip(detections.class_id, detections.tracker_id)
    ]

    annotated_image = bounding_box_annotator.annotate(
        scene=frame, detections=detections
    )
    annotated_image = label_annotator.annotate(
        scene=annotated_image, detections=detections, labels=labels
    )
    line_counter.trigger(detections)
    line_counter_ann.annotate(frame, line_counter)
    print(line_counter.in_count)
    print(line_counter.out_count)

    cv2.imshow("webcam", frame)
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
