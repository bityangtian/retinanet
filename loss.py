from ast import Lambda
import torch
import torch.nn as nn

class Focal_loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.lambda1 = 2
        self.sigma =0.25
        self.mse = nn.MSELoss() 
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, target, pred_box, pred_class, anchor):
        obj = target[..., 0]==1
        noobj = target[..., 0]==0
        #a=torch.tensor([1e-6])
        #a=a.to("cuda")
        #classloss = self.ce(pred_class, target[..., 4:5][obj])
        #focalloss = self.sigma * (1-torch.exp(-classloss))**self.lambda1 * classloss
        after_sigmoid_obj = self.sigmoid(pred_class[obj])
        #a=target[obj]*torch.log(after_sigmoid_obj)
        #part1=-((1-after_sigmoid_obj)**self.lambda1)*(target[obj]*torch.log(after_sigmoid_obj))
        #part2=-(after_sigmoid_obj**self.lambda1)*((1-target[obj])*torch.log(1-after_sigmoid_obj))
        focal_loss1 = self.sigma*(-((1-after_sigmoid_obj)**self.lambda1)*(target[..., 5:25][obj]*torch.log(after_sigmoid_obj))-(after_sigmoid_obj**self.lambda1)*((1-target[..., 5:25][obj])*torch.log(1-after_sigmoid_obj)))
        after_sigmoid_noobj = self.sigmoid(pred_class[noobj])
        focal_loss2 = (1-self.sigma)*(-(1-after_sigmoid_noobj)**self.lambda1*target[..., 5:25][noobj]*torch.log(after_sigmoid_noobj)-after_sigmoid_noobj**self.lambda1*(1-target[..., 5:25][noobj])*torch.log(1-after_sigmoid_noobj))
        focal_loss = torch.cat((focal_loss1, focal_loss2), dim=0)
        focal_loss =torch.mean(focal_loss)
        
        pred_box[..., 0:2]=self.sigmoid(pred_box[..., 0:2])
        target[..., 3:5] = torch.log(1e-6 + target[..., 3:5]/anchor)
        if pred_box[..., 0:4][obj].shape[0] == 0:
            boxloss=torch.tensor([0]).to("cuda")
        else:
            boxloss = torch.sum((pred_box[..., 0:4][obj] - target[..., 1:5][obj])**2)/pred_box[..., 0:4][obj].shape[0]
        #print(pred_box[..., 0:4][obj].shape[0])
        #print(target[..., 1:5][obj])
        #print( "target=",target[..., 1:5][obj])
        #print("pred_box",pred_box[obj])
        #这里会出现nan好像是有的特征图没有分配到的anchor box，加了[obj]后就是空的
        #boxloss = self.mse(pred_box[...,0:4][obj], target[..., 1:5][obj])
        #a=pred_box[obj]
        #b=target[..., 1:5][obj]
        #print(boxloss)
        #print(boxloss+focal_loss)

        
        #return boxloss+focal_loss
        return focal_loss + boxloss