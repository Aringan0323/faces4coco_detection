import torch
import torchvision
from torch.nn import Sigmoid
import numpy as np

class FastRCNN:

    def __init__(self):

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True).to(self.device)
        self.model.eval()


    def person_boxes(self, img):
        image_tensor = torch.from_numpy(img).to(self.device)
        image_tensor = torch.transpose(image_tensor, 0,2)
        image_tensor = torch.transpose(image_tensor, 1,2)
        image_tensor = image_tensor/255
        output = self.model([image_tensor])
        boxes = output[0]['boxes']
        labels = output[0]['labels']
        scores = output[0]['scores']
        person_boxes = []
        box_labels = []
        for i in range(labels.shape[0]):
            if scores[i] > 0.8 and labels[i] == 1:
                person_boxes.append(boxes[i])
                box_labels.append("Person: {}%".format(round(float(scores[i])*100, 2)))
        return person_boxes, box_labels
