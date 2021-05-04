import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torch.nn import Sigmoid
import numpy as np

class FasterRCNN:

    def __init__(self):

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        self.model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

        # get number of input features for the classifier
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained head with a new one
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)

        # now get the number of input features for the mask classifier

        # num_classes = 2
        #
        # in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        #
        # self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        self.model.to(self.device)

        self.model.eval()

        self.sig = Sigmoid()


    def person_mask(self, img):
        image_tensor = torch.from_numpy(img).to(self.device)
        image_tensor = torch.transpose(image_tensor, 0,2)
        image_tensor = torch.transpose(image_tensor, 1,2)
        image_tensor = image_tensor/255
        output = self.model([image_tensor])
        # print(output)
        masks = output[0]['masks']
        all_masks = torch.zeros((1,480,640), device=self.device)
        for i in range(masks.shape[0]):
            this_mask = masks[i]
            all_masks = all_masks + this_mask
        all_masks = self.sig(all_masks)
        all_masks = torch.transpose(all_masks, 1,2)
        all_masks = torch.transpose(all_masks, 0,2) * 255
        return all_masks.detach().cpu().numpy()
