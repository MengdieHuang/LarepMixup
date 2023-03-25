import torch

#--------------
class CustomAlexnet(torch.nn.Module):
    def __init__(self, name='alexnet',n_channels=3, n_outputs=10):
        super(CustomAlexnet, self).__init__()

        self.name = name
        self.num_classes = n_outputs

        self.conv1 = torch.nn.Conv2d(n_channels, 48, 5, stride=1, padding=2)
        self.conv1.bias.data.normal_(0, 0.01)
        self.conv1.bias.data.fill_(0) 
        
        self.relu = torch.nn.ReLU()        
        self.lrn = torch.nn.LocalResponseNorm(2)        
        self.pad = torch.nn.MaxPool2d(3, stride=2)
        
        self.batch_norm1 = torch.nn.BatchNorm2d(48, eps=0.001)
        
        self.conv2 = torch.nn.Conv2d(48, 128, 5, stride=1, padding=2)
        self.conv2.bias.data.normal_(0, 0.01)
        self.conv2.bias.data.fill_(1.0)  
        
        self.batch_norm2 = torch.nn.BatchNorm2d(128, eps=0.001)
        
        self.conv3 = torch.nn.Conv2d(128, 192, 3, stride=1, padding=1)
        self.conv3.bias.data.normal_(0, 0.01)
        self.conv3.bias.data.fill_(0)  
        
        self.batch_norm3 = torch.nn.BatchNorm2d(192, eps=0.001)
        
        self.conv4 = torch.nn.Conv2d(192, 192, 3, stride=1, padding=1)
        self.conv4.bias.data.normal_(0, 0.01)
        self.conv4.bias.data.fill_(1.0)  
        
        self.batch_norm4 = torch.nn.BatchNorm2d(192, eps=0.001)
        
        self.conv5 = torch.nn.Conv2d(192, 128, 3, stride=1, padding=1)
        self.conv5.bias.data.normal_(0, 0.01)
        self.conv5.bias.data.fill_(1.0)  
        
        self.batch_norm5 = torch.nn.BatchNorm2d(128, eps=0.001)
        
        self.fc1 = torch.nn.Linear(1152,512)
        self.fc1.bias.data.normal_(0, 0.01)
        self.fc1.bias.data.fill_(0) 
        
        self.drop = torch.nn.Dropout(p=0.5)
        
        self.batch_norm6 = torch.nn.BatchNorm1d(512, eps=0.001)
        
        self.fc2 = torch.nn.Linear(512,256)
        self.fc2.bias.data.normal_(0, 0.01)
        self.fc2.bias.data.fill_(0) 
        
        self.batch_norm7 = torch.nn.BatchNorm1d(256, eps=0.001)
        
        self.fc3 = torch.nn.Linear(256,self.num_classes)
        self.fc3.bias.data.normal_(0, 0.01)
        self.fc3.bias.data.fill_(0) 
        
        # self.soft = torch.nn.Softmax()    #去掉softmax层
        
    def forward(self, x):
        layer1 = self.batch_norm1(self.pad(self.lrn(self.relu(self.conv1(x)))))
        layer2 = self.batch_norm2(self.pad(self.lrn(self.relu(self.conv2(layer1)))))
        layer3 = self.batch_norm3(self.relu(self.conv3(layer2)))
        layer4 = self.batch_norm4(self.relu(self.conv4(layer3)))
        layer5 = self.batch_norm5(self.pad(self.relu(self.conv5(layer4))))
        flatten = layer5.view(-1, 128*3*3)
        fully1 = self.relu(self.fc1(flatten))
        fully1 = self.batch_norm6(self.drop(fully1))
        fully2 = self.relu(self.fc2(fully1))
        fully2 = self.batch_norm7(self.drop(fully2))
        logits = self.fc3(fully2)
        #softmax_val = self.soft(logits)

        return logits

class CustomVGG19(torch.nn.Module):
    def __init__(self, name="VGG19", n_channels = 3, n_outputs = 10):
        super(CustomVGG19, self).__init__()
        self.name = name
        self.num_classes = n_outputs
        
        self.n_channels = n_channels
        self.features = self._make_layers([64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'])
        self.classifier = torch.nn.Linear(512, self.num_classes)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = torch.nn.functional.dropout(out, p = .5, training = self.training)
        out = self.classifier(out)
        return out

    def _make_layers(self, configuration):
        layers = []
        in_channels = self.n_channels
        for x in configuration:
            if x == 'M':
                layers += [ torch.nn.MaxPool2d(kernel_size = 2, stride = 2) ]
            else:
                layers += [
                    torch.nn.Conv2d(in_channels, x, kernel_size = 3, padding = 1),
                    torch.nn.BatchNorm2d(x),
                    torch.nn.ReLU(inplace=True)
                ]
                in_channels = x
        layers += [torch.nn.AvgPool2d(kernel_size = 1, stride = 1)]
        return torch.nn.Sequential(*layers)




class BasicBlock(torch.nn.Module):
    def __init__(self, in_features, out_features) -> None:
        super().__init__()
 
        self.in_features = in_features
        self.out_features = out_features
 
        stride = 1
        _features = out_features
        if self.in_features != self.out_features:
            # 在输入通道和输出通道不相等的情况下计算通道是否为2倍差值
            if self.out_features / self.in_features == 2.0:
                stride = 2  # 在输出特征是输入特征的2倍的情况下 要想参数不翻倍 步长就必须翻倍
            else:
                raise ValueError("输出特征数最多为输入特征数的2倍！")
 
        self.conv1 = torch.nn.Conv2d(in_features, _features, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu = torch.nn.ReLU(inplace=True)
        self.conv2 = torch.nn.Conv2d(_features, _features, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
 
        # 下采样
        self.downsample = None if self.in_features == self.out_features else torch.nn.Sequential(
            torch.nn.Conv2d(in_features, out_features, kernel_size=1, stride=2, bias=False),
            torch.nn.BatchNorm2d(out_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
 
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
 
        # 输入输出的特征数不同时使用下采样层
        if self.in_features != self.out_features:
            identity = self.downsample(x)
 
        # 残差求和
        out += identity
        out = self.relu(out)
        return out

class CustomResnNet18(torch.nn.Module):

    def __init__(self,name="CustomResnNet18", n_channels = 1, n_outputs = 10) -> None:
        super(CustomResnNet18, self).__init__()
        self.name = name
        self.num_classes = n_outputs
         
        self.conv1 = torch.nn.Conv2d(n_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu = torch.nn.ReLU(inplace=True)
        # self.maxpool = MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.layer1 = torch.nn.Sequential(
            BasicBlock(64, 64),
            BasicBlock(64, 64)
        )
        self.layer2 = torch.nn.Sequential(
            BasicBlock(64, 128),
            BasicBlock(128, 128)
        )
        self.layer3 = torch.nn.Sequential(
            BasicBlock(128, 256),
            BasicBlock(256, 256)
        )
        self.layer4 = torch.nn.Sequential(
            BasicBlock(256, 512),
            BasicBlock(512, 512)
        )
        self.avgpool = torch.nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = torch.nn.Linear(in_features=512, out_features=self.num_classes, bias=True)
 
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)  # <---- 输出为{Tensor:(64,512,1,1)}
        x = torch.flatten(x, 1)  # <----------------这里是个坑 很容易漏 从池化层到全连接需要一个压平 输出为{Tensor:(64,512)}
        x = self.fc(x)  # <------------ 输出为{Tensor:(64,10)}
        return x  




class CustomNet(torch.nn.Module):
    def __init__(self, name="CustomNet", n_channels = 1, n_outputs = 10):
        super(CustomNet, self).__init__()
        
        in_channels = n_channels
        out_features = n_outputs
        self.name = name
                
        self.conv_1 = torch.nn.Conv2d(in_channels=1, out_channels=4, kernel_size=5, stride=1)
        self.conv_2 = torch.nn.Conv2d(in_channels=4, out_channels=10, kernel_size=5, stride=1)
        self.fc_1 = torch.nn.Linear(in_features=4 * 4 * 10, out_features=100)
        self.fc_2 = torch.nn.Linear(in_features=100, out_features=10)

    def forward(self, x):
        x = torch.nn.functional.relu(self.conv_1(x))
        x = torch.nn.functional.max_pool2d(x, 2, 2)
        x = torch.nn.functional.relu(self.conv_2(x))
        x = torch.nn.functional.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 10)
        x = torch.nn.functional.relu(self.fc_1(x))
        x = self.fc_2(x)
        return x
