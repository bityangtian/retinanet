import torch
from torch.utils.data import Dataset
import os
from PIL import Image
from utils import iou
import pandas as pd

class Retinanetdataset(Dataset):
    def __init__(self, csv_file, image_dir, label_dir, anchor, classes=20, s=[80, 40, 20, 10, 5], transform=None):
        self.s = s
        self.annotation = pd.read_csv(csv_file)
        self.anchor = torch.tensor(anchor[0]+anchor[1]+anchor[2]+anchor[3]+anchor[4])
        self.classes = classes
        self.transform =transform
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.iou_thresh = 0.5
        
    def __len__(self):
        return len(self.annotation)
    
    def __getitem__(self, index):
        boxes=[]
        target = [torch.zeros((9, i, i, self.classes + 5)) for i in self.s]
        path_label = os.path.join(self.label_dir, self.annotation.iloc[index, 1])
        #with open(path_label) as f:
        #    for label in f.readline():
        #        boxes+=[
                    #float(m) if float(m)!=int(float(m)) else int(m) for m in label.replace("\n", "").split()]
        #            [float(x) if float(x) != int(float(x)) else int(x) for x in label.replace("\n", "").split(" ")]
        #        ]
        
        with open(path_label) as f:
            for label in f.readlines():
                l1, l2, l3, l4, l5 = [
                    float(x) if float(x) != int(float(x)) else int(x) for x in label.replace("\n", "").split()
                ]
                boxes+=[[l1, l2, l3, l4, l5]]
        image = Image.open(os.path.join(self.image_dir, self.annotation.iloc[index, 0]))
        if self.transform:
            image = self.transform(image)
        for box in boxes:
            #print("进入box的循环")
            anchor_iou = iou(torch.tensor(box[3:5]), self.anchor)
            anchor_iou = anchor_iou.reshape(-1, 45).squeeze(0)
            #print(anchor_iou.shape)
            idx_anchor = anchor_iou.argsort(descending=True)
            #has_anchor = [False]*9     这个似乎没用
            c, x, y, width, height = box[0], box[1], box[2], box[3], box[4]
            #print("c=", c)
            for idx,sort in enumerate(idx_anchor):
                #print("进入选择")
                sort = sort.item()
                feature_idx = idx//9
                scale_idx = idx%9
                feature_s = self.s[feature_idx]
                i = int(feature_s*x)
                j = int(feature_s*y)
                if sort == 0:#sort == 0 说明是iou最大的
                    #print(f"判断成功。。。。。。。。。。。。。。,idx为{idx}")
                    target[feature_idx][scale_idx, i, j, 0] = 1
                    transition = torch.zeros(20)
                    transition[c] = 1
                    x_cell=feature_s*x-i
                    y_cell=feature_s*y-j
                    width_cell = width * feature_s
                    height_cell = height * feature_s
                    #print("x_cell", x_cell)
                    #print(torch.tensor([x_cell, y_cell, width_cell, height_cell]))
                    target[feature_idx][scale_idx, i, j, 1:5] = torch.tensor([x_cell, y_cell, width_cell, height_cell])
                    target[feature_idx][scale_idx, i, j, 5:25] = transition
                elif anchor_iou[idx] >0.4 and anchor_iou[idx]<0.5:
                    #print("忽略样本分配。。。。。。。。。")
                    target[feature_idx][scale_idx, i, j, 0] = -1
        #print(target[0][..., 0])
        return image, target