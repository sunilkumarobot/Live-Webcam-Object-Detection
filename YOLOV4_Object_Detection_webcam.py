#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np

# -------------------------------
# Parameters
# -------------------------------
objectnessThreshold = 0.3
confThreshold = 0.3
nmsThreshold = 0.4

inpWidth = 416
inpHeight = 416

# -------------------------------
# Load Classes
# -------------------------------
classesFile = r"D:\Opencv oct-Nov\15_V4\coco.names"

with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

# -------------------------------
# Load YOLOv4 Tiny
# -------------------------------
modelConfiguration = r"D:\Opencv oct-Nov\15_V4\yolov4-tiny.cfg"
modelWeights = r"D:\Opencv oct-Nov\15_V4\yolov4-tiny.weights"

net = cv2.dnn.readNetFromDarknet(
    modelConfiguration,
    modelWeights
)

# -------------------------------
# Output Layer Names
# -------------------------------
def getOutputsNames(net):

    layersNames = net.getLayerNames()

    return [
        layersNames[i - 1]
        for i in net.getUnconnectedOutLayers()
    ]

# -------------------------------
# Text Display Function
# -------------------------------
FONTFACE = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.7
THICKNESS = 1

def display_text(im, text, x, y):

    textSize = cv2.getTextSize(
        text,
        FONTFACE,
        FONT_SCALE,
        THICKNESS
    )

    dim = textSize[0]
    baseline = textSize[1]

    cv2.rectangle(
        im,
        (x, y),
        (x + dim[0], y + dim[1] + baseline),
        (0, 0, 0),
        cv2.FILLED
    )

    cv2.putText(
        im,
        text,
        (x, y + dim[1]),
        FONTFACE,
        FONT_SCALE,
        (0,255,255),
        THICKNESS,
        cv2.LINE_AA
    )

# -------------------------------
# Detection Function
# -------------------------------
def display_objects(frame, outs):

    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    classIds = []
    confidences = []
    boxes = []

    for out in outs:

        for detection in out:

            objectness = detection[4]

            if objectness > objectnessThreshold:

                scores = detection[5:]

                classId = np.argmax(scores)

                confidence = scores[classId]

                if confidence > confThreshold:

                    center_x = int(detection[0] * frameWidth)
                    center_y = int(detection[1] * frameHeight)

                    width = int(detection[2] * frameWidth)
                    height = int(detection[3] * frameHeight)

                    left = int(center_x - width / 2)
                    top = int(center_y - height / 2)

                    boxes.append(
                        [left, top, width, height]
                    )

                    confidences.append(
                        float(confidence)
                    )

                    classIds.append(classId)

    indices = cv2.dnn.NMSBoxes(
        boxes,
        confidences,
        confThreshold,
        nmsThreshold
    )

    if len(indices) > 0:

        for i in indices.flatten():

            box = boxes[i]

            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]

            cv2.rectangle(
                frame,
                (left, top),
                (left + width, top + height),
                (0,255,0),
                2
            )

            label = "{} : {:.2f}".format(
                classes[classIds[i]],
                confidences[i]
            )

            display_text(
                frame,
                label,
                left,
                top
            )

# -------------------------------
# Webcam
# -------------------------------
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot Open Camera")
    exit()

while True:

    ret, frame = cap.read()

    if not ret:
        break

    blob = cv2.dnn.blobFromImage(
        frame,
        1/255,
        (inpWidth, inpHeight),
        [0,0,0],
        swapRB=True,
        crop=False
    )

    net.setInput(blob)

    outs = net.forward(
        getOutputsNames(net)
    )

    display_objects(frame, outs)

    t, _ = net.getPerfProfile()

    inference_text = "Inference Time: %.2f ms" % (
        t * 1000.0 / cv2.getTickFrequency()
    )

    cv2.putText(
        frame,
        inference_text,
        (10,30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0,0,255),
        2
    )

    cv2.imshow(
        "YOLOv4 Tiny Live Object Detection",
        frame
    )

    key = cv2.waitKey(1) & 0xFF

    if key == 27:      # ESC key
        break

cap.release()
cv2.destroyAllWindows()


# In[ ]:




