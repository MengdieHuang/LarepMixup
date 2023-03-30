### dropout has been removed in this code. original code had dropout#####
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable

import sys, os
import numpy as np
import random
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# from utils import to_one_hot, mixup_process, get_lambda
#----maggie-----
# import random
# import numpy as np
import utils.sampler

# from load_data import per_image_standardization
act = torch.nn.ReLU()


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

def per_image_standardization(x):
    y = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
    mean = y.mean(dim=1, keepdim = True).expand_as(y)    
    std = y.std(dim=1, keepdim = True).expand_as(y)      
    adjusted_std = torch.max(std, 1.0/torch.sqrt(torch.cuda.FloatTensor([x.shape[1]*x.shape[2]*x.shape[3]])))    
    y = (y- mean)/ adjusted_std
    standarized_input =  y.view(x.shape[0],x.shape[1],x.shape[2],x.shape[3])  
    return standarized_input  


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)

def conv_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_uniform(m.weight, gain=np.sqrt(2))
        init.constant(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        init.constant(m.weight, 1)
        init.constant(m.bias, 0)

class wide_basic(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(wide_basic, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=True),
            )

    def forward(self, x):
        out = self.conv1(act(self.bn1(x)))
        out = self.conv2(act(self.bn2(out)))
        out += self.shortcut(x)

        return out

class Wide_ResNet(nn.Module):
    
    def __init__(self, depth, widen_factor, num_classes, per_img_std= False, stride = 1):
        super(Wide_ResNet, self).__init__()
        self.num_classes = num_classes
        self.per_img_std = per_img_std
        self.in_planes = 16

        assert ((depth-4)%6 ==0), 'Wide-resnet_v2 depth should be 6n+4'
        n = int((depth-4)/6)
        k = widen_factor

        print('| Wide-Resnet %dx%d' %(depth, k))
        nStages = [16, 16*k, 32*k, 64*k]

        self.conv1 = conv3x3(3,nStages[0], stride = stride)
        self.layer1 = self._wide_layer(wide_basic, nStages[1], n, stride=1)
        self.layer2 = self._wide_layer(wide_basic, nStages[2], n, stride=2)
        self.layer3 = self._wide_layer(wide_basic, nStages[3], n, stride=2)
        self.bn1 = nn.BatchNorm2d(nStages[3], momentum=0.9)
        self.linear = nn.Linear(nStages[3], num_classes)

    def _wide_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes

        return nn.Sequential(*layers)
    
    """
    ## Modified WRN architecture###
    def __init__(self, depth, widen_factor, dropout_rate, num_classes):
        super(Wide_ResNet, self).__init__()
        self.in_planes = 16
        assert ((depth-4)%6 ==0), 'Wide-resnet_v2 depth should be 6n+4'
        n = (depth-4)/6
        k = widen_factor
        #self.mixup_hidden = mixup_hidden
        print('| Wide-Resnet %dx%d' %(depth, k))
        nStages = [16, 16*k, 32*k, 64*k]
        self.conv1 = conv3x3(3,nStages[0])
        self.bn1 = nn.BatchNorm2d(nStages[0])
        self.layer1 = self._wide_layer(wide_basic, nStages[1], n, dropout_rate, stride=1)
        self.layer2 = self._wide_layer(wide_basic, nStages[2], n, dropout_rate, stride=2)
        self.layer3 = self._wide_layer(wide_basic, nStages[3], n, dropout_rate, stride=2)
        #self.bn1 = nn.BatchNorm2d(nStages[3], momentum=0.9)
        self.linear = nn.Linear(nStages[3], num_classes)
    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)
    """
    # def forward(self, x, target= None, mixup=False, mixup_hidden=False, mixup_alpha=None, defense_mode=None,imagenetmixed10=False):
    def forward(self, x, y=None, defense_mode=None, beta_alpha=None, imagenetmixed10=False, mixup=False, mixup_hidden=False, seed=0):

        if defense_mode in ['manifoldmixup','patchmixup']:
            mixup_hidden=True
        
        #print x.shape
        if self.per_img_std:
            x = per_image_standardization(x)
        
        if mixup_hidden:                        #   隐层混合
            # layer_mix = random.randint(0,2)   #   0-2之间的随机整数，[0, 2]包含0和2 0,1,2
            
            random.seed(seed)            
            layer_mix = random.randint(1, 3)    #   从1 2 3中随机选
            print("layer_mix:",layer_mix)
            
            
        elif mixup:                             #   输入层混合
            layer_mix = 0
        else:
            layer_mix = None                    #   不混合
        
        out = x
        
        # if mixup_alpha is not None:
        #     lam = get_lambda(mixup_alpha)
        #     lam = torch.from_numpy(np.array([lam]).astype('float32')).cuda()
        #     lam = Variable(lam)
        
        # if target is not None :
        #     target_reweighted = to_one_hot(target,self.num_classes)
            
        # if layer_mix == 0:  #不会进入这里 因为不进行输入层mixup
        #         out, target_reweighted = mixup_process(out, target_reweighted, lam=lam)

        out = self.conv1(out)
        out = self.layer1(out)
        
        
        # if layer_mix == 1:  #   layer 1 mixup
        #     out, target_reweighted = mixup_process(out, target_reweighted, lam=lam)
        if layer_mix == 1:                                                          #   如果laymix=1，就在进入layer2前进行潜层混合，否则不混合
            if defense_mode == 'manifoldmixup':
                out, mixed_y = hidden_manifoldmixup_process(out, y, defense_mode, beta_alpha,seed)
            elif defense_mode == 'patchmixup':
                out, mixed_y = hidden_patchmixup_process(out, y, defense_mode,seed)
        #-------------   

        out = self.layer2(out)

        # if layer_mix == 2:
        #     out, target_reweighted = mixup_process(out, target_reweighted, lam=lam)
        if layer_mix == 2:                                                          #   如果laymix=2，就在进入layer3前进行潜层混合，否则不混合
            if defense_mode == 'manifoldmixup':
                out, mixed_y = hidden_manifoldmixup_process(out, y, defense_mode, beta_alpha,seed)
            elif defense_mode == 'patchmixup':
                out, mixed_y = hidden_patchmixup_process(out, y, defense_mode,seed)     
                    
        
        out = self.layer3(out)
        
        # if  layer_mix == 3:
        #     out, target_reweighted = mixup_process(out, target_reweighted, lam=lam)
        if layer_mix == 3:                                                          #   如果laymix=3，就在进入layer4前进行潜层混合，否则不混合
            if defense_mode == 'manifoldmixup':
                out, mixed_y = hidden_manifoldmixup_process(out, y, defense_mode, beta_alpha,seed)
            elif defense_mode == 'patchmixup':
                out, mixed_y = hidden_patchmixup_process(out, y, defense_mode,seed)   
                    
                    
        out = act(self.bn1(out))
        
        # out = F.avg_pool2d(out, 8)
        #---------20230330 maggie---------
        if imagenetmixed10 == True:
            # print("out.shape", out.shape)           #   out.shape torch.Size([16, 512, 32, 32])
            avg = nn.AdaptiveAvgPool2d((1, 1))        #   不用原来的池化函数
            out = avg(out)
            # print("out.shape", out.shape)           #   out.shape torch.Size([16, 512, 1, 1])
        else:
            # out = F.avg_pool2d(out, 4)   
            out = F.avg_pool2d(out, 8) 
        #---------------------------------- 
            
        
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        
        # if target is not None:
        #     return out, target_reweighted
        if defense_mode in ['manifoldmixup','patchmixup']:
            return out, mixed_y  
        else: 
            return out
        
                  
        
def wrn28_10(num_classes=10, dropout = False, per_img_std = False, stride = 1):
    #print ('this')
    model = Wide_ResNet(depth=28, widen_factor=10, num_classes=num_classes, per_img_std = per_img_std, stride = stride)
    return model

def wrn28_2(num_classes=10, dropout = False, per_img_std = False, stride = 1):
    #print ('this')
    model = Wide_ResNet(depth =28, widen_factor =2, num_classes = num_classes, per_img_std = per_img_std, stride = stride)
    return model



# if __name__ == '__main__':
#     net=Wide_ResNet(28, 10, 0.3, 10)
#     y = net(Variable(torch.randn(1,3,32,32)))

#     print(y.size())

#----------maggie------------
def wideresnet2810():
    return wrn28_10()
#----------------------------