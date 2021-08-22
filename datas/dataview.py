"""
Author: maggie
Date:   2021-06-15
Place:  Xidian University
@copyright
"""

import os
import torch.utils.data
import PIL.Image
import utils.util


#------------------------------------可视化数据集-----------------------------------
def ViewMNISTTrain(target_dataset):
    data = target_dataset               #   zip格式的数据集path
    assert data is not None
    assert isinstance(data, str)
    print('data=%s'%data)

    training_set_kwargs = utils.util.EasyDict(class_name='utils.training.dataset.ImageFolderDataset', path=data, use_labels=True, max_size=None, xflip=False)  # 解析data给出的训练集路径
    training_set = utils.util.construct_class_by_name(**training_set_kwargs) # subclass of training.dataset.Dataset
    print('Num images: ', len(training_set))
    print('Image shape:', training_set.image_shape)
    print('Label shape:', training_set.label_shape)

    savedir='/home/data/maggie/mnist/mnist4stylegan2ada/datasets/viewmnist'
    os.makedirs(savedir,exist_ok=True)  
    device = torch.device('cuda')

    classification = ['zero','one','two','three','four','five','size','seven','eight','nine']
    for index, (imgs, labels) in enumerate(training_set):
        if index < 1000:
            print('index= %s' % index)
            print('one hot label= %s' % labels)     # one hot label
            label = torch.argmax(torch.tensor(labels), -1)
            print('label= %s' % label)              # int label

            img = torch.tensor(imgs,device=device)        
            # print(img.shape)    
            img = img[-1]   
            # print(img.shape)    
            Tesnsor_img = img.permute(1,0).clamp(0, 255).to(torch.uint8)  #    PyTorch 中的张量默认采用 N×D×H×W 的顺序，并且数据范围在 [0, 1]，需要进行转置和规范化。(1,2,0)的意思是第0维放原来的第1维，第1维放原来的第2维，第2维放原来的第0维[256,256,3]
            array_img = Tesnsor_img.cpu().numpy()                           #   https://blog.csdn.net/weixin_40756000/article/details/113805235
            PIL_image = PIL.Image.fromarray(array_img,'L')  

            """对于彩色图像，不管其图像格式是PNG，还是BMP，或者JPG，
            在PIL中，使用Image模块的open()函数打开后，返回的图像对象的模式都是“RGB”。
            而对于灰度图像，不管其图像格式是PNG，还是BMP，或者JPG，打开后，其模式为“L”。"""

            target_fname = f'{savedir}/{index:08d}-{label:000d}-{classification[label]}.png'
            PIL_image.save(target_fname)
            print('target_fname=%s' % target_fname)
    
    print('Viewing *mnist* dataset finished !')
    return savedir

def ViewCIFAR10Train(target_dataset):
    data = target_dataset               #   zip格式的数据集path
    assert data is not None
    assert isinstance(data, str)
    print('data=%s'%data)

    training_set_kwargs = utils.util.EasyDict(class_name='utils.training.dataset.ImageFolderDataset', path=data, use_labels=True, max_size=None, xflip=False)  # 解析data给出的训练集路径
    training_set = utils.util.construct_class_by_name(**training_set_kwargs) # subclass of training.dataset.Dataset
    print('Num images: ', len(training_set))
    print('Image shape:', training_set.image_shape)
    print('Label shape:', training_set.label_shape)

    savedir='/home/data/maggie/cifar10/cifar104stylegan2ada/datasets/viewcifar10'
    os.makedirs(savedir,exist_ok=True)  
    device = torch.device('cuda')

    classification = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
    for index, (imgs, labels) in enumerate(training_set):
        if index < 1000:
            print('index= %s' % index)
            print('one hot label= %s' % labels)     # one hot label
            label = torch.argmax(torch.tensor(labels), -1)
            print('label= %s' % label)              # int label

            img = torch.tensor(imgs,device=device)        
            print(img.shape)                                                
            Tesnsor_img = img.permute(1,2,0).clamp(0, 255).to(torch.uint8)  #    PyTorch 中的张量默认采用 N×D×H×W 的顺序，并且数据范围在 [0, 1]，需要进行转置和规范化。(1,2,0)的意思是第0维放原来的第1维，第1维放原来的第2维，第2维放原来的第0维[256,256,3]
            array_img = Tesnsor_img.cpu().numpy()                           #   https://blog.csdn.net/weixin_40756000/article/details/113805235
            PIL_image = PIL.Image.fromarray(array_img,'RGB')
            target_fname = f'{savedir}/{index:08d}-{label:000d}-{classification[label]}.png'
            PIL_image.save(target_fname)
            print('target_fname=%s' % target_fname)
    
    print('Viewing *cifar10* dataset finished !')
    return savedir
#----------------------------------------------------------------------------------