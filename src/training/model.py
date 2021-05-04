import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import numpy as np
import pickle

class FaceRCNN:

    def save_model_txt(self, model, path):
        fout = open(path, 'w')
        for k, v in model.state_dict().items():
            fout.write(str(k) + '\n')
            fout.write(str(v.tolist()) + '\n')
        fout.close()


    def __init__(self):


        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
        # get number of input features for the classifier
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained head with a new one
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 3)
        self.model.load_state_dict(torch.load('../torch_model/model_state_dict.pt'))
        # self.save_model_txt(self.model, '../torch_model/model_state_dict.txt')
        # torch.save(self.model.state_dict(),"../torch_model/model_state_dict.pth", pickle_protocol=1, _use_new_zipfile_serialization=False)
        # with open(r"../torch_model/entire_model.pkl", 'wb') as output_file:
        #     pickle.dump(self.model, output_file, protocol=2)
        self.model.to(self.device)

        self.model.eval()


        self.labels_dict =  {1:'Person', 2:"Face"}
        self.colors_dict = {1:(0,255,0), 2:(0,0,255)}


    def simple_detection(self, img):
        image_tensor = torch.from_numpy(img).to(self.device)
        image_tensor = torch.transpose(image_tensor, 0,2)
        image_tensor = torch.transpose(image_tensor, 1,2)
        image_tensor_normal = image_tensor/255
        output = self.model([image_tensor_normal])
        boxes = output[0]['boxes']
        labels = output[0]['labels']
        scores = output[0]['scores']
        inds_clean_scores = []
        for i in range(scores.shape[0]):
            if scores[i] >= 0.8:
                inds_clean_scores.append(i)
        inds_clean_scores = torch.LongTensor(inds_clean_scores, device=self.device)
        clean_boxes = torch.index_select(boxes, 0, inds_clean_scores)
        clean_labels = torch.index_select(labels, 0, inds_clean_scores)
        return clean_boxes, clean_labels




    def person_boxes(self, img):
        image_tensor = torch.from_numpy(img).to(self.device)
        image_tensor = torch.transpose(image_tensor, 0,2)
        image_tensor = torch.transpose(image_tensor, 1,2)
        image_tensor_normal = image_tensor/255
        output = self.model([image_tensor_normal])
        boxes = output[0]['boxes']
        labels = output[0]['labels']
        scores = output[0]['scores']
        inds_clean_scores = []
        for i in range(scores.shape[0]):
            if scores[i] >= 0.8:
                inds_clean_scores.append(i)
        inds_clean_scores = torch.LongTensor(inds_clean_scores).to(self.device)
        clean_boxes = torch.index_select(boxes, 0, inds_clean_scores)
        clean_labels = torch.index_select(labels, 0, inds_clean_scores).tolist()
        clean_scores = torch.index_select(scores, 0, inds_clean_scores)
        image_tensor = torchvision.utils.draw_bounding_boxes(image_tensor.detach().cpu(),
                                                clean_boxes,
                                                [self.labels_dict[id] + ": {}%".format(str(round(float(scores[i])*100, 2))) for i, id in enumerate(clean_labels)],
                                                [self.colors_dict[id] for id in clean_labels],
                                                fill=False,
                                                width=2,
                                                font_size=15)
        image_tensor = torch.transpose(image_tensor, 2,1)
        image_tensor = torch.transpose(image_tensor, 2,0)
        return clean_boxes, clean_labels, image_tensor.numpy()




    def trace_and_save(self):
        dummy_input = torch.rand((1,3,480,640))
        tracable_model = self.model.cpu()
        traced = torch.jit.trace(tracable_model, dummy_input)
        print(traced.code)
        traced.save("../torch_model/jit_model.pt")
