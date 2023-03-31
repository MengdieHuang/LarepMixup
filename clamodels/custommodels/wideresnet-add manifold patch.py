
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

import random
import numpy as np
import utils.sampler

__all__ = ['wideResNet','wideResnet28_10']


#-------maggie add 20230330---------
def hidden_patchmixup_process(out, y, defense_mode,seed):      
    batch_size = out.size()[0]
    
    torch.manual_seed(seed)
    index = torch.randperm(batch_size).cuda()                       #   生成一组长度为batchsize的随机数组 32
    # print("index:",index)                                           
    # print("indices.len:",len(index))                                
    # print("out.shape:",out.shape)                                   
    # print("y.shape:",y.shape)       

    """
    index: tensor([0, 1, 2, 3], device='cuda:0')
    indices.len: 4
    out.shape: torch.Size([4, 64, 32, 32])
    y.shape: torch.Size([4, 10])
    """                      
    is_2d = True if len(out.size()) == 2 else False                 #   out有4维 不是2d

    m = utils.sampler.BernoulliSampler(out.size(0), out.size(1), is_2d, p=None)
    # print("m:",m)
    # print("m.shape:",m.shape)                                       #   m.shape: torch.Size([4, 64, 1, 1])

    lam = []
    for i in range(len(m)):
        lam_i = (torch.nonzero(m[i]).size(0)) / m.size(1)
        lam.append(lam_i)

    lam = np.asarray(lam)
    lam = torch.tensor(lam).unsqueeze(1)

    m1 = m.cpu()
    m2 = (1.-m).cpu()
    lam1 = lam.cpu()
    lam2 = (1.-lam).cpu() 

    # print("m1:",m1)
    # print("m2:",m2)
    # print("lam1:",lam1)
    # print("lam2:",lam2)

    """
    lam1: tensor([[0.5039],[0.4727],[0.5547],[0.4922]], dtype=torch.float64)
    lam2: tensor([[0.4961],[0.5273],[0.4453],[0.5078]], dtype=torch.float64)
    """

    # print("m1.shape:",m1.shape)
    # print("m2.shape:",m2.shape)
    # print("lam1.shape:",lam1.shape)
    # print("lam2.shape:",lam2.shape)

    """
    m1.shape: torch.Size([4, 64, 1, 1])
    m2.shape: torch.Size([4, 64, 1, 1])
    lam1.shape: torch.Size([4, 1])
    lam2.shape: torch.Size([4, 1])
    """

    w1 = out.cpu()
    w2 = out[index,:].cpu()        
    y1 = y.cpu()
    y2 = y[index,:].cpu()

    # print("w1:",w1)
    # print("w2:",w2)
    # print("y1:",y1)
    # print("y2:",y2)

    """
    y1: tensor([[0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
                [0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]])
    y2: tensor([[0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
                [0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]])
    """
    out = m1*w1 + m2*w2
    mixed_y = lam1*y1 + lam2*y2

    # print("out.shape:", out.shape)                                
    # print("out:",out) 
    # print("mixed_y.shape:", mixed_y.shape)                        
    # print("mixed_y:",mixed_y)      
    """
    out.shape: torch.Size([4, 128, 16, 16])
    mixed_y.shape: torch.Size([4, 10])
    mixed_y: tensor([
                    [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.0000, 0.0000, 0.0000, 0.0000],
                    [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.4688, 0.0000, 0.0000, 0.5312], 
                    [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.5781, 0.0000, 0.0000, 0.4219],
                    [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.0000]
                    ], dtype=torch.float64)
    mixed_y: tensor([[0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
                    [0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
                    [0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
                    [0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]], dtype=torch.float64)                    
    """                               
    # raise error

    out=out.cuda()
    mixed_y=mixed_y.cuda()

    return out, mixed_y

def hidden_manifoldmixup_process(out, y, defense_mode, beta_alpha, seed):
    alpha=beta_alpha

    np.random.seed(seed)    
    lam = np.random.beta(alpha, alpha)                              #   根据beta分布的alpha参数生成随机数lam
    batch_size = out.size()[0]
    
    torch.manual_seed(seed)    
    index = torch.randperm(batch_size).cuda()                       #   生成一组长度为batchsize的随机数组 32
    
    # print("index:",index)                                           
    # print("indices.len:",len(index))                                
    # print("lam:",lam)                                               
    # print("out.shape:",out.shape)                                   
    # print("y.shape:",y.shape)                                       

    """
    index: tensor([24, 31, ..., 12, 29, 19],  device='cuda:0')
    indices.len: 32
    lam: 0.09522716670648239
    out.shape: torch.Size([32, 128, 16, 16])
    y.shape: torch.Size([32, 10])
    """
    out = lam*out + (1-lam)*out[index,:]                            #   把out和打乱样本顺序后的out mixup      
    mixed_y = lam*y + (1-lam)*y[index,:]
    
    # print("out.shape:", out.shape)                                #   out.shape: torch.Size([32, 64, 32, 32])
    # print("out:",out) 
    # print("mixed_y.shape:", mixed_y.shape)                        #   mixed_y.shape: torch.Size([32, 10])
    # print("mixed_y:",mixed_y)                                     
    # raise error

    return out, mixed_y
#----------------


def conv3x3(c_in, c_out, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(c_in, c_out, kernel_size=3, stride=stride,padding=1,bias=False)
def conv1x1(c_in, c_out, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(c_in, c_out, kernel_size=1, stride=stride,bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, c_in, c_out, stride=1, dropout_rate=0.3, downsample=None):
        super(BasicBlock, self).__init__()

        self.bn1 = nn.BatchNorm2d(c_in)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = conv3x3(c_in, c_out, stride)
        
        self.dropout = nn.Dropout(p=dropout_rate)
        self.downsample=downsample
        
        self.bn2 = nn.BatchNorm2d(c_out)
        self.conv2 = conv3x3(c_out, c_out)
        self.stride = stride

    def forward(self, x):
        
        identity = x
        out = self.bn1(x)
        out = self.relu(out)
        
        out = self.conv1(out)
        out = self.dropout(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        if self.downsample is not None:
            identity = self.downsample(identity)
        out += identity
        return out


class wideResNet(nn.Module):

    # def __init__(self, block, layers, widen, dropout_rate=0.3, num_classes=10):
    def __init__(self, block, layers, widen, dropout_rate=0.3, n_outputs=10, n_channels=3):
        
        num_classes=n_outputs
        
        super(wideResNet, self).__init__()

        self.conv1 = conv3x3(3,16)

        self.layer1 = self._make_layer(block, 16, 16*widen, layers[0], dropout_rate)
        self.layer2 = self._make_layer(block, 16*widen, 32*widen, layers[1], dropout_rate, stride=2)
        self.layer3 = self._make_layer(block, 32*widen, 64*widen, layers[2], dropout_rate, stride=2)

        self.batch_norm = nn.BatchNorm2d(64*widen)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64*widen, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0.1)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0.1)


    def _make_layer(self, block, c_in, c_out, blocks, dropout_rate, stride=1):
        downsample=None
        if c_in!=c_out or stride!=1:
            downsample = nn.Sequential(conv1x1(c_in*block.expansion, c_out * block.expansion, stride))
            #downsample = nn.Sequential(nn.MaxPool2d(stride,stride),nn.ConstantPad3d([0,0,0,0,0,(c_out - c_in)* block.expansion],0))#functional.pad(x,[0,0,0,0,0,(c_out - c_in)* block.expansion]))
        layers = []
        layers.append(block(c_in*block.expansion, c_out, stride, dropout_rate, downsample))
        for _ in range(1, blocks):
            layers.append(block(c_out * block.expansion, c_out, dropout_rate=dropout_rate))

        return nn.Sequential(*layers)

    # def forward(self, x):
    #     x = self.conv1(x)

    #     x = self.layer1(x)
    #     x = self.layer2(x)
    #     x = self.layer3(x)

    #     x = self.batch_norm(x)
    #     x = self.relu(x)
    #     x = self.avgpool(x)
    #     x = x.view(x.size(0), -1)
    #     x = self.fc(x)

    #     return x
    
    # def forward(self, x):
    def forward(self, x, y=None, defense_mode=None, beta_alpha=None, imagenetmixed10=False, mixup_hidden=False, seed=0):

        if defense_mode in ['manifoldmixup','patchmixup']:
            mixup_hidden=True

        if mixup_hidden:                        #   隐层混合           
            random.seed(seed)            
            layer_mix = random.randint(1, 3)    #   从1 2 3中随机选
            print("layer_mix:",layer_mix)            
        else:
            layer_mix = None                    #   不混合
                    
        x = self.conv1(x)
        x = self.layer1(x)

        if layer_mix == 1:                                                          #   如果laymix=1，就在进入layer2前进行潜层混合，否则不混合
            if defense_mode == 'manifoldmixup':
                x, mixed_y = hidden_manifoldmixup_process(x, y, defense_mode, beta_alpha,seed)
            elif defense_mode == 'patchmixup':
                x, mixed_y = hidden_patchmixup_process(x, y, defense_mode,seed)
        #-------------   
                
        
        x = self.layer2(x)

        if layer_mix == 2:                                                          #   如果laymix=2，就在进入layer3前进行潜层混合，否则不混合
            if defense_mode == 'manifoldmixup':
                x, mixed_y = hidden_manifoldmixup_process(x, y, defense_mode, beta_alpha,seed)
            elif defense_mode == 'patchmixup':
                x, mixed_y = hidden_patchmixup_process(x, y, defense_mode,seed)            
        
        x = self.layer3(x)

        if layer_mix == 3:                                                          #   如果laymix=3，就在进入layer4前进行潜层混合，否则不混合
            if defense_mode == 'manifoldmixup':
                x, mixed_y = hidden_manifoldmixup_process(x, y, defense_mode, beta_alpha,seed)
            elif defense_mode == 'patchmixup':
                x, mixed_y = hidden_patchmixup_process(x, y, defense_mode,seed)   
                
        x = self.batch_norm(x)
        x = self.relu(x)
        
        
        # x = self.avgpool(x)
        #---------20230330 maggie---------
        if imagenetmixed10 == True:
            avg = nn.AdaptiveAvgPool2d((1, 1))        #   不用原来的池化函数
            x = avg(x)
        else:
            x = self.avgpool(x)
        #----------------------------------         
        
        
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        # return x
        if defense_mode in ['manifoldmixup','patchmixup']:
            return x, mixed_y  
        else: 
            return x  
        
# def resnet18(pretrained=False, **kwargs):
#     """Constructs a ResNet-18 model.

#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#     """
#     model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
#     if pretrained:
#         model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
#     return model


# def resnet34(pretrained=False, **kwargs):
#     """Constructs a ResNet-34 model.

#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#     """
#     model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
#     if pretrained:
#         model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
#     return model


def wideResnet28_10(**kwargs):
    """Constructs a wideResnet28_10 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = wideResNet(BasicBlock, [4, 4, 4], 10,**kwargs)
    return model

def resnet110():
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [18, 18, 18])
    return model
# def resnet101(pretrained=False, **kwargs):
#     """Constructs a ResNet-101 model.

#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#     """
#     model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
#     if pretrained:
#         model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
#     return model


# def resnet152(pretrained=False, **kwargs):
#     """Constructs a ResNet-152 model.

#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#     """
#     model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
#     if pretrained:
#         model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
#     return model
if __name__=='__main__':
    a=wideResnet28_10()
    b=torch.randn(1,3,32,32)
    a.to(0)
    b=b.to(0)
    c=a(b)
    print(c.shape)
