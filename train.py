import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from utils import anchor
from loss import Focal_loss
from model import Retinanet
import torch.optim as optim
from tqdm import tqdm
from dataset import Retinanetdataset
import numpy as np

batch_size=2

anchors = anchor
epoch=20
device = "cuda"
retinanet = Retinanet().to(device)
transform = transforms.Compose(
    [transforms.Resize((640, 640)),
     transforms.ToTensor()]
)
my_dataset = Retinanetdataset("D:\\dataset\\8examples.csv", "D:\\dataset\\images", "D:\\dataset\\labels", anchors, transform=transform)
my_dataload = DataLoader(my_dataset,batch_size=batch_size,shuffle=False)
my_loss = Focal_loss()

optimizer = optim.Adam(retinanet.parameters(), lr=1e-5 + 2e-6)#我用+1e-6怎么会nan？



loss_anchor0 =torch.tensor(anchors[0]).unsqueeze(0).unsqueeze(0).reshape(9, 1, 1, 2).repeat(1, 80, 80, 1).to(device)
loss_anchor1 =torch.tensor(anchors[1]).unsqueeze(0).unsqueeze(0).reshape(9, 1, 1, 2).repeat(1, 40, 40, 1).to(device)
loss_anchor2 =torch.tensor(anchors[2]).unsqueeze(0).unsqueeze(0).reshape(9, 1, 1, 2).repeat(1, 20, 20, 1).to(device)
loss_anchor3 =torch.tensor(anchors[3]).unsqueeze(0).unsqueeze(0).reshape(9, 1, 1, 2).repeat(1, 10, 10, 1).to(device)
loss_anchor4 =torch.tensor(anchors[4]).unsqueeze(0).unsqueeze(0).reshape(9, 1, 1, 2).repeat(1, 5, 5, 1).to(device)
#print("loss_anchor",loss_anchor0)
#print("losss_anchor.shape=", loss_anchor0.shape)
#print(loss_anchor1.shape)
#print(loss_anchor2.shape)
#print(loss_anchor3.shape)
#print(loss_anchor4.shape)

meanloss = []
for i in range(100):
    loop = tqdm(my_dataload, leave=True)
    for batch_idx, (x, y) in enumerate(loop):
        x=x.to(device)
        y0,y1,y2,y3,y4=(y[0].to(device),
        y[1].to(device),
        y[2].to(device),
        y[3].to(device),
        y[4].to(device))
        #print(y0[..., 2])
        box, classes = retinanet(x)
        #print((box[0][..., 2:4]/loss_anchor0).shape)
        #print( my_loss(y0, box[0], classes[0], loss_anchor0))
        #obj = y4[..., 0]==1
        #print(y4[..., 1:5][obj])
        loss =(
            my_loss(y0, box[0], classes[0], loss_anchor0)+
            my_loss(y1, box[1], classes[1], loss_anchor1)+
            my_loss(y2, box[2], classes[2], loss_anchor2)+
            my_loss(y3, box[3], classes[3], loss_anchor3)+
            my_loss(y4, box[4], classes[4], loss_anchor4))
        #print(loss)
        meanloss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #print(meanloss)
        mean_loss = sum(meanloss) / len(meanloss)
        loop.set_postfix(loss=mean_loss)
    print(f"第{i+1}epoch完成")