'''ResNet in PyTorch.

BasicBlock and Bottleneck module is from the original ResNet paper:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385

PreActBlock and PreActBottleneck module is from the later paper:
[2] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks. arXiv:1603.05027
'''
from distutils.log import error
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

#----maggie-----
import random
import numpy as np

#---------------

#----------------
def hidden_mixup_process(out, y, defense_mode, beta_alpha):
    alpha=beta_alpha
    lam = np.random.beta(alpha, alpha)                              #   根据beta分布的alpha参数生成随机数lam
    batch_size = out.size()[0]
    index = torch.randperm(batch_size).cuda()                       #   生成一组长度为batchsize的随机数组 32
    print("index:",index)                                           #   [93 84 34 19 59...] 
    print("indices.len:",len(index))                                #   indices.len: 100                32
    print("lam:",lam)                                               #   lam: 0.0967587
    print("out.shape:",out.shape)                                   #   out.shape: torch.Size([100, 64, 32, 32])   100个样本 64通道 32x32
    print("y.shape:",y.shape)                                       #   out.shape: torch.Size([100, 64, 32, 32])   100个样本 64通道 32x32

    raise error
    out = out * lam + out[index,:] * (1 - lam)                      #   把out和打乱样本顺序后的out mixup      
    ratio = torch.ones(out.shape[0], device='cuda') * lam
    print("out.shape:",out.shape)
    print("ratio.shape:",ratio.shape)                               #   ratio.shape: torch.Size([100])
    # print("ratio:",ratio)                                           #   ratio: tensor([0.0968, 0.0968,...]  
    mixed_target = lam * y + (1 - lam) * y[index, :]
    return out, mixed_target
#----------------

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out)
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class PreActBottleneck(nn.Module):
    '''Pre-activation version of the original Bottleneck module.'''
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out)
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        out += shortcut
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = conv3x3(3,64)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    # def forward(self, x, lin=0, lout=5):
    #     out = x
    #     if lin < 1 and lout > -1:
    #         out = self.conv1(out)
    #         out = self.bn1(out)
    #         out = F.relu(out)
    #     if lin < 2 and lout > 0:
    #         out = self.layer1(out)
    #     if lin < 3 and lout > 1:
    #         out = self.layer2(out)
    #     if lin < 4 and lout > 2:
    #         out = self.layer3(out)
    #     if lin < 5 and lout > 3:
    #         out = self.layer4(out)
    #     if lout > 4:
    #         out = F.avg_pool2d(out, 4)
    #         out = out.view(out.size(0), -1)
    #         out = self.linear(out)
    #     return out

    # maggie 20220713
    # def forward(self, x, lin=0, lout=5):
    def forward(self, x, lin=0, lout=5, y=None, defense_mode=None, beta_alpha=None):

        #---------------     
        if defense_mode == 'manifoldmixup':
            print("defense_mode",defense_mode)
            print("y.shape",y.shape)
            print("beta_alpha",beta_alpha)
            print("maggie test2 20220718")
            """
            defense_mode manifoldmixup
            y.shape torch.Size([32, 10])
            """
            print("manifold mixup -----maggie")
            layer_mix = random.randint(1, 3)                                            #   从1 2 3中随机选
            print("layer_mix:",layer_mix)                                               #   layer_mix：1  

        else:
            # print("standard training -----maggie")
            layer_mix = None
        #---------------        

        out = x                                                                         #   把x直接赋值给out

        if lin < 1 and lout > -1:
            out = self.conv1(out)
            out = self.bn1(out)
            out = F.relu(out)

        if lin < 2 and lout > 0:
            out = self.layer1(out)                                                      #   第一个layer计算                             

        if lin < 3 and lout > 1:
            #-------------
            if layer_mix == 1:                                                          #   如果laymix=1，就在进入layer2前进行潜层混合，否则不混合
                out, mixed_y = hidden_mixup_process(out, y, defense_mode, beta_alpha)
            #-------------            
            out = self.layer2(out)                                                      #   第二个layer计算

        if lin < 4 and lout > 2:
            #-------------
            if layer_mix == 2:                                                          #   如果laymix=2，就在进入layer3前进行潜层混合，否则不混合
                out, mixed_y = hidden_mixup_process(out, y, defense_mode, beta_alpha)
            #-------------                 
            out = self.layer3(out)                                                      #   第三个layer计算

        if lin < 5 and lout > 3:
            #-------------
            if layer_mix == 3:                                                          #   如果laymix=3，就在进入layer4前进行潜层混合，否则不混合
                out, mixed_y = hidden_mixup_process(out, y, defense_mode, beta_alpha)
            #-------------               
            out = self.layer4(out)                                                      #   第四个layer计算

        if lout > 4:
            out = F.avg_pool2d(out, 4)
            out = out.view(out.size(0), -1)
            out = self.linear(out)

        if defense_mode == 'manifoldmixup':
            return out, mixed_y
        else:
            return out

def ResNet18():
    return ResNet(PreActBlock, [2,2,2,2])

def ResNet34():
    # return ResNet(BasicBlock, [3,4,6,3])
    return ResNet(PreActBlock, [3,4,6,3])   #   maggie changed

def ResNet50():
    # return ResNet(Bottleneck, [3,4,6,3])
    return ResNet(PreActBlock, [3,4,6,3])   #   maggie changed

def ResNet101():
    return ResNet(Bottleneck, [3,4,23,3])

def ResNet152():
    return ResNet(Bottleneck, [3,8,36,3])


def test():
    net = ResNet18()
    y = net(Variable(torch.randn(1,3,32,32)))
    print(y.size())

# test()

#---------maggie-------
def preactresnet18():
    return ResNet18()

def preactresnet34():
    return ResNet34()

def preactresnet50():
    return ResNet50()

#----------------------
