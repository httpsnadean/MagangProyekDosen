import os
import random
import cv2
import numpy as np
from util import get_detections, load_class_names

# (1) Define paths
cfg_path = './models/mask_rcnn_inception/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt'
weights_path = './models/mask_rcnn_inception/frozen_inference_graph.pb'
class_names_path = './models/mask_rcnn_inception/class.names'

img_path = './windah.jpeg'

# (2) Load class names
class_names = load_class_names(class_names_path)

# (3) Load image
img = cv2.imread(img_path)
H, W, C = img.shape

# (4) Load model
net = cv2.dnn.readNetFromTensorflow(weights_path, cfg_path)

# (5) Convert image to blob
blob = cv2.dnn.blobFromImage(img)

# (6) Get detections
boxes, masks = get_detections(net, blob)

# (7) Draw masks and labels
colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for j in range(len(class_names))]
empty_img = np.zeros((H, W, C))

detection_th = 0.5
for j in range(boxes.shape[2]):
    bbox = boxes[0, 0, j]
    class_id = int(bbox[1])
    score = bbox[2]

    if score > detection_th:
        x1, y1, x2, y2 = int(bbox[3] * W), int(bbox[4] * H), int(bbox[5] * W), int(bbox[6] * H)
        label = f"{class_names[class_id]}: {score:.2f}"
        
        # Draw bounding box
        img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 5)
        img = cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        mask = masks[j, class_id]
        mask = cv2.resize(mask, (x2 - x1, y2 - y1))
        _, mask = cv2.threshold(mask, 0.5, 1, cv2.THRESH_BINARY)

        for c in range(3):
            empty_img[y1:y2, x1:x2, c] = mask * colors[class_id][c]

# (8) Visualization
overlay = ((0.6 * empty_img) + (0.4 * img)).astype("uint8")
cv2.imshow('mask', empty_img)
cv2.imshow('img', img)
cv2.imshow('overlay', overlay)

cv2.waitKey(0)
cv2.destroyAllWindows()
