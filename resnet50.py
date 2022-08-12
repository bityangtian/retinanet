import torch.nn as nn
import torch 
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
    #print('x第一次relu：',out.shape)
    out=self.relu(self.bn2(self.conv2(out)))
    #print('x第二次relu：',out.shape)
    out=self.conv3(out)
    #print('x遇到残差时',out.shape)
    out=self.bn3(out)
    #print('归一化后：',out.shape)
    #print(downsample(x).shape)
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

class Residual(nn.Module):
  def __init__(self, layers, block=BasicBlock, num_classes=1000):
    super(Residual,self).__init__()
    self.block=block
    self.conv0=nn.Conv2d(3, 64, kernel_size=7, padding=3, stride=2, bias=False)
    self.bn0=nn.BatchNorm2d(64)
    self.relu=nn.ReLU()
    self.pool=nn.MaxPool2d(kernel_size=3, padding=1, stride=2)
    self.stage1=layer(block, 64, 1, num_block=layers[0])
    self.stage2=layer(block, 128, 2, layers[1])
    self.stage3=layer(block, 256, 2, layers[2])
    self.stage4=layer(block, 512, 2, layers[3])
    self.avgpool=nn.AdaptiveAvgPool2d(7)
    self.fc=nn.Linear(2048*7*7, num_classes)
  def forward(self, x):
    x=self.pool(self.relu(self.bn0(self.conv0(x))))
    #print('x池化后：',x.shape)
    x=self.stage1(x)
    #print('第一块执行完：',x.shape)
    x=self.stage2(x)
    x=self.stage3(x)
    x=self.stage4(x)
    x=self.avgpool(x)
    x=x.reshape(-1, 2048*7*7)
    #print(x.shape)
    x=self.fc(x)
    return x
if __name__ == "__main__":
    model = Residual([3, 4, 6, 3])
    print(model)