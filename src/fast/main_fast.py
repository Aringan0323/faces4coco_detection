#!/usr/bin/env python

import torch
import torchvision
import matplotlib.pyplot as plt
import PIL
import numpy as np
import cv2
from faster_rcnn import FasterRCNN


if __name__ == "__main__":

    videoCaptureObject = cv2.VideoCapture(2)

    model = FasterRCNN()


    while(True):
        cap, frame = videoCaptureObject.read()
        if cap:
            # cv2.imshow('Frame', frame)
            mask = model.person_mask(frame)
            cv2.imshow("Mask", mask)
        if(cv2.waitKey(1) and 0xFF == ord('q')):
            videoCaptureObject.release()
            cv2.destroyAllWindows()
            break
