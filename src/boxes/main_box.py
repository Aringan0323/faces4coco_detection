#!/usr/bin/env python

import torch
import torchvision
import matplotlib.pyplot as plt
import PIL
import numpy as np
import cv2
from fast_rcnn import FastRCNN

if __name__ == "__main__":

    videoCaptureObject = cv2.VideoCapture(2)

    model = FastRCNN()

    font = cv2.FONT_HERSHEY_SIMPLEX

    red = (0,0,255)

    green = (0, 255, 0)

    while(True):
        cap, frame = videoCaptureObject.read()
        if cap:
            # cv2.imshow('Frame', frame)
            boxes, labels = model.person_boxes(frame)
            for i, box in enumerate(boxes):
                cX = int((box[0] + box[2])/2)
                cY = int((box[1] + box[3])/2)
                cv2.circle(frame, (cX, cY), 5, red, -1)
                cv2.putText(frame, labels[i], (cX+5, cY+5), font , 0.7, green, 2)
                # cv2.rectangle(frame, box, (0,0,255), 4)
            cv2.imshow("Boxes", frame)
        if(cv2.waitKey(1) and 0xFF == ord('q')):
            videoCaptureObject.release()
            cv2.destroyAllWindows()
            break
