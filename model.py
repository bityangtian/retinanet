from re import S
from turtle import forward
import torch
import torch.nn as nn

class BasicBlock(nn.Module):
  def __init__(self, inchannl, midchannl, stride, downsample=None):
    super(BasicBlock, self).__init__()
    self.relu=nn.ReLU()
    self.conv1=nn.Conv2d(inchannl, midchannl, kernel_size=1, padding=0, stride=stride, bias=False)
    self.bn1=nn.BatchNorm2d(midchannl)
    self.conv2=nn.Conv2d(midchannl, midchannl, kernel_size=3, padding=1, bias=False)
    self.bn2=nn.BatchNorm2d(midchannl)
    self.conv3=nn.Conv2d(midchannl, midchannl*4, kernel_size=1, padding=0, bias=False)
    self.bn3=nn.BatchNorm2d(midchannl*4)
    self.downsample=downsample
  def forward(self, x):
    out=self.relu(self.bn1(self.conv1(x)))
    out=self.relu(self.bn2(self.conv2(out)))
    out=self.conv3(out)
    out=self.bn3(out)
    if self.downsample !=None:
      return self.relu(out+self.downsample(x))
    else:
      return self.relu(out+x)
    
def layer(block, inchannl, stride, num_block):
  layer=[]
  downsample=None
  if stride==1 :
    downsample=nn.Sequential(
        nn.Conv2d(inchannl, inchannl*4, kernel_size=1, padding=0, stride=stride),
        nn.BatchNorm2d(inchannl*4)
    )
    layer.append(block(inchannl=inchannl, midchannl=inchannl, stride=stride, downsample=downsample))
  else:
    downsample=nn.Sequential(
        nn.Conv2d(inchannl*2, inchannl*4, kernel_size=1, padding=0, stride=stride),
        nn.BatchNorm2d(inchannl*4)
    )
    layer.append(block(inchannl=inchannl*2, midchannl=inchannl, stride=stride, downsample=downsample))
  for i in range(num_block-1):
    layer.append(block(inchannl=inchannl*4, midchannl=inchannl, stride=1, downsample=None))
  return nn.Sequential(*layer)

class Feature(nn.Module):
  def __init__(self, layers=[3, 4, 6, 3], block=BasicBlock):
    super(Feature,self).__init__()
    self.conv51 = nn.Conv2d(2048, 256, kernel_size=1, stride=1)
    self.conv41 = nn.Conv2d(1024, 256, kernel_size=1, stride=1)
    self.conv31 = nn.Conv2d(512, 256, kernel_size=1,stride=1)
    self.conv3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
    self.conv32 = nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=2)
    self.bn = nn.BatchNorm2d(256)
    self.upsample = nn.Upsample(scale_factor=2)
    self.block=block
    self.conv0=nn.Conv2d(3, 64, kernel_size=7, padding=3, stride=2, bias=False)
    self.bn0=nn.BatchNorm2d(64)
    self.relu=nn.ReLU()
    self.pool=nn.MaxPool2d(kernel_size=3, padding=1, stride=2)
    self.stage2=layer(block, 64, 1, num_block=layers[0])
    self.stage3=layer(block, 128, 2, layers[1])
    self.stage4=layer(block, 256, 2, layers[2])
    self.stage5=layer(block, 512, 2, layers[3])
  def forward(self, x):
    x=self.pool(self.relu(self.bn0(self.conv0(x))))
    x2=self.stage2(x)
    x3=self.stage3(x2)
    x4=self.stage4(x3)
    x5=self.stage5(x4)
    mid = self.conv51(x5)
    p5 = self.conv3(mid)
    upsample1 = self.upsample(mid)
    upsample2 = self.upsample(self.conv41(x4)+upsample1)
    p4 = self.conv3(self.conv41(x4)+upsample1)
    p3 = self.conv3(self.conv31(x3)+upsample2)
    p6 = self.conv32(p5)
    p7 = self.conv32(self.bn(self.relu(p6)))
    return [p3, p4, p5, p6, p7]

class Detection(nn.Module):
    def __init__(self, class_num = 20, anchor_num = 9, mode = 'class'):
        super(Detection, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU()
        )
        if mode == 'class':
            self.lastconv = nn.Conv2d(256, class_num*anchor_num, kernel_size=3, padding=1)
        else:
            self.lastconv = nn.Conv2d(256, 4*anchor_num, kernel_size=3, padding=1)
    def forward(self, x):
        x=self.layer(x)
        x=self.layer(x)
        x=self.layer(x)
        x=self.lastconv(x).permute(0, 2, 3, 1)
        x=x.reshape(x.shape[0], 9, x.shape[1], x.shape[2], -1)
        return x


class Retinanet(nn.Module):
    def __init__(self ):
        super(Retinanet, self).__init__()
        self.backbone = Feature()
        self.classnet = Detection(mode = "class")
        self.boxnet = Detection(mode = "box")
    def forward(self, x):
        feature_list = self.backbone(x)
        box_list = []
        class_list = []
        box_list = [self.boxnet(i) for i in feature_list]
        class_list = [self.classnet(i) for i in feature_list]
        return box_list, class_list
        


if __name__ == "__main__":
    input = torch.rand(1, 3, 640, 640)
    model = Retinanet()
    box, class_i=model(input)
    print("box[0", box[0].shape)
    print("box[1", box[1].shape)
    print("box[2", box[2].shape)
    print("box[3", box[3].shape)
    print("box[4", box[4].shape)
    print("class[0", class_i[0].shape)
    print("class[1", class_i[1].shape)
    print("class[2", class_i[2].shape)
    print("class[3", class_i[3].shape)
    print("class[4", class_i[4].shape)
""" box[0 torch.Size([1, 9, 80, 80, 4])
box[1 torch.Size([1, 9, 40, 40, 4])
box[2 torch.Size([1, 9, 20, 20, 4])
box[3 torch.Size([1, 9, 10, 10, 4])
box[4 torch.Size([1, 9, 5, 5, 4])
class[0 torch.Size([1, 9, 80, 80, 20])
class[1 torch.Size([1, 9, 40, 40, 20])
class[2 torch.Size([1, 9, 20, 20, 20])
class[3 torch.Size([1, 9, 10, 10, 20])
class[4 torch.Size([1, 9, 5, 5, 20]) """