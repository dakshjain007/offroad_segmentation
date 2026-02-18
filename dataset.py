import os
import cv2
import torch
from torch.utils.data import Dataset

class OffroadDataset(Dataset):
    def __init__(self, root):
        self.img_dir = os.path.join(root, "color_images")
        self.mask_dir = os.path.join(root, "segmentation")
        self.images = os.listdir(self.img_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]

        img = cv2.imread(os.path.join(self.img_dir, img_name))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img,(256,256))/255.0

        mask = cv2.imread(os.path.join(self.mask_dir, img_name),0)
        mask = cv2.resize(mask,(256,256),interpolation=cv2.INTER_NEAREST)

        mask = torch.tensor(mask,dtype=torch.long)
        img = torch.tensor(img,dtype=torch.float32).permute(2,0,1)

        return img, mask
