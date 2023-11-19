# https://pytorch.org/hub/ultralytics_yolov5/
# https://pytorch.org/get-started/locally/
import cv2
# conda activate feeder ( should say it in the terminal instead of 'base')
import torch
from ultralytics.utils.plotting import Annotator

# create model
model = torch.hub.load('ultralytics/yolov8', 'yolov8n.pt', pretrained=True)

cap = cv2.VideoCapture(0) #Set the webcam

while True:
    ret, frame = cap.read()

    if(ret):
        # train the model
        results = model.predict(frame)

        # capture output image; array length will change w input change
        # check objects detected
        #print(results)

        for r in results:

            annotator = Annotator(frame)

            boxes = r.boxes
            for box in boxes:
                b = box.xyxy[0]  # get box coordinates in (top, left, bottom, right) format
                c = box.cls
                annotator.box_label(b, model.names[int(c)])

            img = annotator.result()
            cv2.imshow('YOLO V8 Detection', img)
            if cv2.waitKey(1) & 0xFF == ord(' '):
                break

    # press q to quit and close windows
    if 0xFF == ord("q"):
        break

# Close the window
cap.release()

# De-allocate any associated memory usage
cv2.destroyAllWindows()