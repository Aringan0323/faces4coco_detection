#!/usr/bin/env python

import torch
import torchvision
import matplotlib.pyplot as plt
import PIL
import numpy as np
import cv2
from mask_rcnn import MaskRCNN


if __name__ == "__main__":

    videoCaptureObject = cv2.VideoCapture(2)

    model = MaskRCNN()


    while(True):
        cap, frame = videoCaptureObject.read()
        if cap:
            # cv2.imshow('Frame', frame)
            mask, boxes = model.person_mask(frame)
            for box in boxes:
                cX = int((box[0] + box[2])/2)
                cY = int((box[1] + box[3])/2)
                cv2.rectangle(frame, box, (0,0,255), 4)
                cv2.circle(frame, (cX, cY), 5, (0,0,255), -1)
            cv2.imshow("Mask", mask)
            cv2.imshow("Centroid", frame)
        if(cv2.waitKey(1) and 0xFF == ord('q')):
            videoCaptureObject.release()
            cv2.destroyAllWindows()
            break
