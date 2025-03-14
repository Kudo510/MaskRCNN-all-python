import os
import torch
import glob
import random

from torchvision.io import read_image, ImageReadMode
from torchvision.ops.boxes import masks_to_boxes
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F

random.seed(10)

class XYZIDB(torch.utils.data.Dataset):
    def __init__(self, root = "xyz_splits/val", transforms=True):
        self.root = root
        self.imgs = list()

        scenes = glob.glob(os.path.join(root, "*"))
        for scene in scenes:
            self.imgs = self.imgs + list(sorted(glob.glob(os.path.join(scene, "rgb/*.png"))))
        random.shuffle(self.imgs)

    def __getitem__(self, idx):
        # load images and masks
        img_path = self.imgs[idx]
        img_id = img_path.split("/")[-1].split(".")[0]
        mask_scene_path = os.path.join(img_path.split("rgb")[0], "mask_visib")
        mask_paths = glob.glob(os.path.join(mask_scene_path, f"{img_id}_*.png"))
        # obj_id for each mask


        # use try except here - also use log not print
        if not os.path.exists(img_path):
           print(f"The path '{img_path}' does not exist or is invalid.")

        img = read_image(img_path)

        mask_list = []
        i = 1
        for mask_path in mask_paths:
            mask = read_image(mask_path).squeeze(0)
            
            # Check if mask is not entirely black
            if mask.sum() > 0:
                # Scale the mask by (i+1) only for non-black masks
                scaled_mask = mask / 255 *i
                mask_list.append(scaled_mask)
                i += 1 

        mask = torch.stack(mask_list).to(torch.uint8)
        stacked_mask = (torch.sum(mask, 0)>0).to(torch.uint8)
        # masks = (masks > 0.5)
        # mask = torch.clamp(torch.sum(torch.stack(mask_list), dim=0), min=0, max = 1)
        #print("mask.shape, img.shape", mask.shape, img.shape)
        #print("mask", mask)
        # instances are encoded as different colors
        obj_ids = torch.tensor(range(len(mask_list)+1))
        # obj_ids = torch.arange(len(mask_list)+1)
        #print("obj_ids",obj_ids)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]
        #print("obj_ids_2",obj_ids)
        num_objs = len(obj_ids)
        #print("num_objs",num_objs)

        # split the color-encoded mask into a set
        # of binary masks
        masks = (mask == obj_ids[:, None, None]).to(dtype=torch.uint8)
        #print("masks.shape", masks.shape)
        # get bounding box coordinates for each mask
        # boxes = torch.stack([masks_to_boxes(m.unsqueeze(0)) for m in mask_list]).squeeze(1)
        boxes = masks_to_boxes(masks)
        boxes = validate_and_fix_boxes(boxes=boxes, max_height= img.shape[1], max_width=img.shape[2])
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)

        image_id = img_path
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        # Wrap sample and targets into torchvision tv_tensors:
        img = tv_tensors.Image(img)

        target = {}
        target["boxes"] = tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=F.get_size(img))
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        target["masks"] = tv_tensors.Mask(masks)

        # Do for Dinov2 
        if self.transforms:
            # img = self.transforms(img)
            # stacked_mask = self.transforms(stacked_mask.unsqueeze(0))
            img, target = self.transforms(img, target)
        
        # target["masks"] = tv_tensors.Mask(stacked_mask)

        return img, target

    def __len__(self):
        return len(self.imgs)



class XYZDataset(torch.utils.data.Dataset):
    def __init__(self, root = "xyz_splits/val", transforms=True, train_pbr = True, test=False):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list()
        self.train_pbr = train_pbr
        self.test = test
        if self.train_pbr:
            for dataset_name in dataset_names:
                img = list(sorted(glob.glob(os.path.join(root, f"{dataset_name}/train_pbr/rgb/*.jpg"))))
                self.imgs= self.imgs + img
        elif self.train_pbr is False and self.test is False:
            for dataset_name in dataset_names:
                img = list(sorted(glob.glob(os.path.join(root, f"{dataset_name}/rgb/*.png"))))
                self.imgs = self.imgs + img
        if self.test:
            test_folders = os.listdir("test_mask_rcnn")
            for test_folder in test_folders:
                # img_path = glob.glob(f"test_mask_rcnn/{test_folder}/rgb/*.png")[0]
                # self.imgs.append(img_path)
                # img_path = glob.glob(f"test_mask_rcnn/{test_folder}/rgb_realsense/*.png")
                img_path = glob.glob(f"test_mask_rcnn/{test_folder}/rgb/*.png")
                img_path = img_path[0]
                # if len(img_path) == 0:
                #     img_path = glob.glob(f"test_mask_rcnn/{test_folder}/gray_xyz/*.png")[0]
                # else:
                #     img_path = img_path[0]
                self.imgs.append(img_path)
                
        random.shuffle(self.imgs)
        
        # self.masks = list(sorted(os.listdir(os.path.join(root, "mask_visib"))))
        #print('len', len(self.masks),len(self.imgs))

    def __getitem__(self, idx):
        # load images and masks
        img_path = self.imgs[idx]
        img_id = img_path.split("/")[-1].split(".")[0]
        if "gray_xyz" in img_path:
            mask_path = f"{img_path.split('gray_xyz')[0]}mask_visib_xyz"
        elif "gray_photoneo" in img_path:
            mask_path = f"{img_path.split('gray_photoneo')[0]}mask_visib_photoneo"

        elif "realsense" in img_path:
            mask_path = f"{img_path.split('rgb_realsense')[0]}mask_visib_realsense"
            if not os.path.exists(mask_path):
                mask_path = f"{img_path.split('rgb_realsense')[0]}mask_visib_xyz"
        else:
            mask_path = f"{img_path.split('rgb')[0]}mask_visib"
        single_mask_paths = glob.glob(os.path.join(mask_path, f"{img_id}_*.png"))

        if not os.path.exists(img_path):
           print(f"The path '{img_path}' does not exist or is invalid.")

        img = read_image(img_path)

        mask_list = []
        i = 1
        for mask_path in single_mask_paths:
            mask = read_image(mask_path).squeeze(0)
            
            # Check if mask is not entirely black
            if mask.sum() > 0:
                # Scale the mask by (i+1) only for non-black masks
                scaled_mask = mask / 255 *i
                mask_list.append(scaled_mask)
                i = i +1 
        mask = torch.stack(mask_list).to(torch.uint8)
        stacked_mask = (torch.sum(mask, 0)>0).to(torch.uint8)
        # masks = (masks > 0.5)
        # mask = torch.clamp(torch.sum(torch.stack(mask_list), dim=0), min=0, max = 1)
        #print("mask.shape, img.shape", mask.shape, img.shape)
        #print("mask", mask)
        # instances are encoded as different colors
        obj_ids = torch.tensor(range(len(mask_list)+1))
        # obj_ids = torch.arange(len(mask_list)+1)
        #print("obj_ids",obj_ids)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]
        #print("obj_ids_2",obj_ids)
        num_objs = len(obj_ids)
        #print("num_objs",num_objs)

        # split the color-encoded mask into a set
        # of binary masks
        masks = (mask == obj_ids[:, None, None]).to(dtype=torch.uint8)
        #print("masks.shape", masks.shape)
        # get bounding box coordinates for each mask
        # boxes = torch.stack([masks_to_boxes(m.unsqueeze(0)) for m in mask_list]).squeeze(1)
        boxes = masks_to_boxes(masks)
        boxes = validate_and_fix_boxes(boxes=boxes, max_height= img.shape[1], max_width=img.shape[2])
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)

        image_id = img_path
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        # Wrap sample and targets into torchvision tv_tensors:
        img = tv_tensors.Image(img)

        target = {}
        target["boxes"] = tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=F.get_size(img))
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        target["masks"] = tv_tensors.Mask(masks)

        # Do for Dinov2 
        if self.transforms is not None:
            # img = self.transforms(img)
            # stacked_mask = self.transforms(stacked_mask.unsqueeze(0))
            img, target = self.transforms(img, target)
        
        # target["masks"] = tv_tensors.Mask(stacked_mask)

        return img, target

    def __len__(self):
        return len(self.imgs)

def validate_and_fix_boxes(boxes, max_height, max_width):
    """
    Validate and correct bounding boxes to ensure positive height and width.
    
    Args:
        boxes (torch.Tensor): Input bounding boxes [x_min, y_min, x_max, y_max]
        max_height (float): Maximum image height
        max_width (float): Maximum image width
    
    Returns:
        torch.Tensor: Corrected bounding boxes
    """
    corrected_boxes = []
    for box in boxes:
        x_min, y_min, x_max, y_max = box
        
        # Ensure minimum box size (e.g., 1 pixel)
        if x_max <= x_min:
            x_max = min(x_min + 1, max_width)
        
        if y_max <= y_min:
            y_max = min(y_min + 1, max_height)
        
        # Clamp coordinates to image bounds
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(x_max, max_width)
        y_max = min(y_max, max_height)
        
        corrected_boxes.append([x_min, y_min, x_max, y_max])
    
    return torch.tensor(corrected_boxes, dtype=boxes.dtype, device=boxes.device)


