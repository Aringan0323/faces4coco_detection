import torch
import torchvision
from torch.nn import Sigmoid
import numpy as np

class MaskRCNN:

    def __init__(self):

        self.device = torch.device('cuda')

        self.model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True).to(self.device)
        self.model.eval()

        self.sig = Sigmoid()


    def person_mask(self, img):
        image_tensor = torch.from_numpy(img).to(self.device)
        image_tensor = torch.transpose(image_tensor, 0,2)
        image_tensor = torch.transpose(image_tensor, 1,2)
        image_tensor = image_tensor/255
        output = self.model([image_tensor])
        masks = output[0]['masks']
        boxes = output[0]['boxes']
        labels = output[0]['labels']
        scores = output[0]['scores']
        all_masks = torch.zeros((1,480,640), device=self.device)
        person_boxes = []
        for i in range(labels.shape[0]):
            if scores[i] > 0.6 and labels[i] == 1:
                this_mask = masks[i]
                all_masks = all_masks + this_mask
                person_boxes.append(boxes[i].detach().cpu().numpy())
        all_masks = torch.transpose(all_masks, 1,2)
        all_masks = torch.transpose(all_masks, 0,2)
        return self.sig(all_masks).detach().cpu().numpy(), person_boxes
