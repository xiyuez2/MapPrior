import os
import numpy as np
import torch
import albumentations
from torch.utils.data import Dataset
# from det3d.datasets import build_dataloader, build_dataset
from mmdet3d.datasets import build_dataloader, build_dataset
# from det3d.torchie import Config
from torchpack.utils.config import configs
from mmcv import Config
from mmdet3d.utils import recursive_eval

from taming.data.base import ImagePaths, NumpyPaths, ConcatDatasetWithIndex
import cv2
import glob
import random
# from tools.demo_utils import Box,_second_det_to_nusc_box

def take_threshold(data,threshold = (0.50, 0.45, 0.45, 0.40, 0.35, 0.40)): #(0.5,0.45,0.45,0.40,0.35,0.4)
    for i in range(6):
        data[i] = data[i] > threshold[i]
    return data.astype(int)

class CustomBase(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.data = None
        self.pre_data = None
        self.gt_data = None
        self.flip = True
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        res_example  = {}
        gt = np.load(self.gt_data[i]).astype(int)
        
        if self.flip:
            if random.random()<0.5:
                gt = gt[:,::-1,:].copy()
            if random.random()<0.5:
                gt = gt[:,:,::-1].copy()

        res_example['image'] = gt
        
        c,w,h = gt.shape # 6 200 200
        res_example['pure_nos'] = np.random.rand(1,w,h)

        return res_example

class CustomTrain(CustomBase):
    def __init__(self, size, training_images_list_file):
        super().__init__()

        self.data = glob.glob("./data/train_"+training_images_list_file+"/gt_*.npy")
        self.gt_data = []

        for i in range(len(self.data)):
            self.gt_data.append("./data/train_"+training_images_list_file+"/gt_"+str(i)+".npy")

class CustomTest(CustomBase):
    def __init__(self, size, test_images_list_file):
        super().__init__()

        self.gt_data = []
        self.pre_data = []
        self.data = glob.glob("./data/val_"+test_images_list_file+"/gt_*.npy")
        self.flip = False
        for i in range(len(self.data)):
            self.gt_data.append("./data/val_"+test_images_list_file+"/gt_"+str(i)+".npy")

class CustomPointBase(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.data = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        example = self.data[i]
        res_example  = {}

        res_example['image'] = np.array(example['gt_masks_bev'])
        c,w,h = res_example['image'].shape # 6 200 200
        res_example['pure_nos'] = np.random.rand(1,w,h)
        
        return res_example

class CustomPointTrain(CustomPointBase):
    def __init__(self, size, config_name):
        super().__init__()
        # print(cfg)
        config_name = config_name
        configs.load(config_name, recursive=True)
        cfg = Config(recursive_eval(configs), filename=config_name)
        dataset = build_dataset(cfg.data.train)
        self.data = dataset

class CustomPointTest(CustomPointBase):
    def __init__(self, size, config_name):
        super().__init__()
        # cfg = Config.fromfile("bev_data.py")
        config_name = config_name
        configs.load(config_name, recursive=True)
        cfg = Config(recursive_eval(configs), filename=config_name)
        dataset = build_dataset(cfg.data.test)
        self.data = dataset
