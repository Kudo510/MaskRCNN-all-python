import torch
from utils.engine import train_one_epoch, evaluate
from utils import utils
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
from torchvision.io import read_image
import torch.optim as optim
from data.dataloader import ArticleSegmentationDataset, XYZDataset, RealXYZDataset
from utils.utils import get_transform, calculate_f1_score
from models.model import get_model_instance_segmentation, Dinov2ForSemanticSegmentation
from MLFlowUtils import MLFlowUtils
import time
from datetime import datetime


def train():

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # our dataset has two classes only - background and person
    num_classes = 15
    # use our dataset and defined transformations
    # dataset = XYZDataset('datasets_finetuning_MaskRCNN', get_transform(train=True), train_pbr=True, test=False)

    dataset = RealXYZDataset('xyz_splits/val', get_transform(train=True))

    # dataset_test = XYZDataset('datasets_finetuning_MaskRCNN', get_transform(train=False), train_pbr=False, test=True)

    dataset_test = RealXYZDataset('xyz_splits/test', get_transform(train=False))

    # split the dataset in train and test set
    indices = torch.randperm(len(dataset)).tolist()
    indices_test = torch.randperm(len(dataset_test)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:])
    dataset_test = torch.utils.data.Subset(dataset_test, indices_test[:])

    batch_size = 2
    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4, #4,
        collate_fn=utils.collate_fn
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=2,
        shuffle=False,
        num_workers=4, #4,
        collate_fn=utils.collate_fn
    )

    # # get the model using our helper function
    model = get_model_instance_segmentation(num_classes)
    # load latest model
    model.load_state_dict(torch.load('output/xyz_maskrcnn_best_model.pth'))

    # model = Dinov2ForSemanticSegmentation()
    # model.load_state_dict(torch.load('output/dinov2_finetune.pth'))
    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=0.000001, weight_decay=0.0001)

    #and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=50,
        gamma=0.6
    )

    # let's train it for 5 epochs
    num_epochs = 100


    # correct here
    eval = evaluate(model, data_loader_test, device=device)
    metric_best_val = eval.coco_eval["segm"].stats[0] # mAP score
    print("current metric_best_val", metric_best_val)

    # for epoch in range(num_epochs):
    #     # train for one epoch, printing every 10 iterations
    #     train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
    #     # update the learning rate
    #     lr_scheduler.step()
    #     # evaluate on the test dataset

    #     #update here
    #     if epoch%10 == 9:
    #         model_eval = evaluate(model, data_loader_test, device=device)
    #         eval_time = datetime.now()
    #         eval_time = float(eval_time.strftime("%Y%m%d%H%M%S"))
    #         saved_checkpoint_path = f'output/newxyz_xyz_mask_rcnn_model_real_images_{str(epoch)}_{eval_time}.pth'
    #         torch.save(model.state_dict(), saved_checkpoint_path)
    #         metric_val = model_eval.coco_eval["segm"].stats[0] # mAP score
    #         # metric_val = calculate_f1_score(model_eval.coco_eval["bbox"].stats[0], model_eval.coco_eval["bbox"].stats[8])
    #         if (metric_val - metric_best_val) > 0:
    #             metric_best_val = metric_val
    #             print("saving the best model with acc: ", metric_val)
    #             torch.save(model.state_dict(), 'output/newxyz_xyz_maskrcnn_real_images_best_model.pth')
        # if epoch%25 == 0:
        #     torch.save(model.state_dict(), f'output/dinov2_finetned_ver_2_real_images_{epoch}.pth')   
    

if __name__ == '__main__':
    torch.cuda.empty_cache()
    train()