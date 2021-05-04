from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torch
import torchvision
import torch
import torchvision
from dataset import cocoFace

def get_transform():
    custom_transforms = []
    custom_transforms.append(torchvision.transforms.ToTensor())
    return torchvision.transforms.Compose(custom_transforms)

def collate_fn(batch):
    return tuple(zip(*batch))


def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model

if __name__ == '__main__':

    torch.cuda.empty_cache()


    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


    train_data_dir = '../data/coco/train2017'
    train_coco = '../data/coco_faces/cocoface/cocoface_instances_train2017.json'
    test_data_dir =  '../data/coco/val2017'
    test_coco = '../data/coco_faces/cocoface/cocoface_instances_val2017.json'

    # create own Dataset
    # train_dataset = cocoFace(root=train_data_dir,
    #                           annotation=train_coco,
    #                           transforms=get_transform()
    #                           )
    test_dataset = cocoFace(root=test_data_dir,
                              annotation=test_coco,
                              transforms=get_transform()
                              )

    # collate_fn needs for batch


    # Batch size
    train_batch_size = 2

    # own DataLoader
    train_data_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=train_batch_size,
                                              shuffle=True,
                                              num_workers=4,
                                              collate_fn=collate_fn)

    # test_data_loader = torch.utils.data.DataLoader(test_dataset,
    #                                           batch_size=train_batch_size,
    #                                           shuffle=True,
    #                                           num_workers=4,
    #                                           collate_fn=collate_fn)
    # 2 classes; Only target class or background
    num_classes = 3
    num_epochs = 2
    model = get_model_instance_segmentation(num_classes)
    model.load_state_dict(torch.load('../torch_model/model_state_dict.pt'))
    # move model to the right device
    model.to(device)

    # parameters
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    len_dataloader = len(train_data_loader)

    for epoch in range(num_epochs):
        model.train()
        i = 0
        for imgs, annotations in train_data_loader:
            i += 1
            imgs = list(img.to(device) for img in imgs)
            annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]
            # if i%100 == 0:
            #     print(annotations)
            loss_dict = model(imgs, annotations)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            print(f'Iteration: {i}/{len_dataloader}, Loss: {losses}')
        torch.save(model.state_dict(), '../torch_model/model_state_dict.pt')
    # torch.save(model.state_dict(), '../torch_model/model_state_dict.pt')
