# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import functools
import io
import json
import os
import pickle
from re import X
import sys
import tarfile
import gzip
import zipfile
from pathlib import Path
from typing import Callable, Optional, Tuple, Union

import click
import numpy as np
import PIL.Image
from torch._C import dtype
from tqdm import tqdm

import scipy.io #maggie
#------------------------------------maggie----------------------------------

def open_kmnist(images_gz: str, *, max_images: Optional[int]):                                  #   images_gz(.gz)等于输入的source路径
    labels_gz = images_gz.replace('-images-idx3-ubyte.gz', '-labels-idx1-ubyte.gz')
    assert labels_gz != images_gz
    images = []
    labels = []

    with gzip.open(images_gz, 'rb') as f:
        images = np.frombuffer(f.read(), np.uint8, offset=16)                                   #   读取数据集样本
    with gzip.open(labels_gz, 'rb') as f:
        labels = np.frombuffer(f.read(), np.uint8, offset=8)

    images = images.reshape(-1, 28, 28)
    images = np.pad(images, [(0,0), (2,2), (2,2)], 'constant', constant_values=0)               #   np.pad(images, [(0,0), (2,2), (2,2)] 把1x28x28的样本填充为1x32x32的样本
    """
    
    第1轴两边缘分别填充0个、0个数值0
    第2轴两边缘分别填充2个、2个数值0
    第2轴两边缘分别填充2个、2个数值0

    """
    assert images.shape == (60000, 32, 32) and images.dtype == np.uint8
    assert labels.shape == (60000,) and labels.dtype == np.uint8
    assert np.min(images) == 0 and np.max(images) == 255
    assert np.min(labels) == 0 and np.max(labels) == 9

    max_idx = maybe_min(len(images), max_images)

    def iterate_images():
        for idx, img in enumerate(images):
            yield dict(img=img, label=int(labels[idx]))
            if idx >= max_idx-1:
                break

    return max_idx, iterate_images()

def open_cifar100(tarball: str, *, max_images: Optional[int]):
    images = []
    labels = []

    with tarfile.open(tarball, 'r:gz') as tar:
        # for batch in range(1, 6):
        member = tar.getmember(f'cifar-100-python/train')
        with tar.extractfile(member) as file:
            data = pickle.load(file, encoding='latin1')
        print(data)
        print(data.keys())                                                                  #   dict_keys(['filenames', 'batch_label', 'fine_labels', 'coarse_labels', 'data'])
        
        images.append(data['data'].reshape(-1, 3, 32, 32))
        labels.append(data['fine_labels'])

    images = np.concatenate(images)
    labels = np.concatenate(labels)
    images = images.transpose([0, 2, 3, 1]) # NCHW -> NHWC
    assert images.shape == (50000, 32, 32, 3) and images.dtype == np.uint8
    assert labels.shape == (50000,) and labels.dtype in [np.int32, np.int64]
    assert np.min(images) == 0 and np.max(images) == 255
    assert np.min(labels) == 0 and np.max(labels) == 99                                             #   共100个类别

    max_idx = maybe_min(len(images), max_images)

    def iterate_images():
        for idx, img in enumerate(images):
            yield dict(img=img, label=int(labels[idx]))
            if idx >= max_idx-1:
                break

    return max_idx, iterate_images()

def open_svhn(matpath: str, *, max_images: Optional[int]):                                              #   读取mat格式数据集
    
    mat = scipy.io.loadmat(matpath)
    # print("mat:",mat)
    # print(mat.keys())                                                                       #   dict_keys(['__header__', '__version__', '__globals__', 'X', 'y'])
    
    images = mat['X']
    # print("images[0]: ",images[0])
    # print("images.type: ",type(images))                                                     #   images.type:  <class 'numpy.ndarray'>
    # print("images.shape: ",images.shape)                                                    #   images.shape:  (32, 32, 3, 73257)

    images = images.transpose([3, 2, 0, 1]) # HMCN ->NCHW
    # print("images[0]: ",images[0])
    # print("images.type: ",type(images))                                                     #   images.type:  <class 'numpy.ndarray'>
    # print("images.shape: ",images.shape)                                                    #   images.shape:  (73257, 3, 32, 32)
    # print("images.dtype: ",images.dtype)                                                    #    uint8

    labels = mat['y']
    # print("labels[0]: ",labels[0])
    # print("labels.type: ",type(labels))                                                     #   labels.type:  <class 'numpy.ndarray'>
    # print("images.shape: ",labels.shape)                                                    #   images.shape:  (73257, 1)
            
    labels = np.concatenate(labels)
    # print("labels.shape: ",labels.shape)                                                    #   images.shape:  (73257,)
    # print("labels.dtype: ",labels.dtype)                                                    #    uint8
    labels = np.int32(labels)
    # print("labels.dtype: ",labels.dtype)                                                    #    uint8
    # print("labels.type: ",type(labels))                                                     #   labels.type:  <class 'numpy.ndarray'>
    # print("np.min(labels)： ",np.min(labels))
    # print("np.max(labels)： ",np.max(labels))

    images = images.reshape(-1, 3, 32, 32)                                                      #   NCHW
    images = images.transpose([0, 2, 3, 1])                                                     #   NCHW -> NHWC
    assert images.shape == (73257, 32, 32, 3) and images.dtype == np.uint8
    assert labels.shape == (73257,) and labels.dtype in [np.int32, np.int64]
    assert np.min(images) == 0 and np.max(images) == 255
    assert np.min(labels) == 1 and np.max(labels) == 10                                             #   共10个类别

    max_idx = maybe_min(len(images), max_images)

    def iterate_images():
        for idx, img in enumerate(images):
            yield dict(img=img, label=int(labels[idx]))
            if idx >= max_idx-1:
                break

    return max_idx, iterate_images()

def open_stl10(tarball: str, *, max_images: Optional[int]):
    
    """
    10 classes: airplane, bird, car, cat, deer, dog, horse, monkey, ship, truck.
    96x96 pixel
    """
    images = []
    labels = []

    with tarfile.open(tarball, 'r:gz') as tar:  
        print(tar.getnames())                                                               #   ['stl10_binary', 'stl10_binary/test_X.bin', 'stl10_binary/test_y.bin', 'stl10_binary/train_X.bin', 'stl10_binary/train_y.bin', 'stl10_binary/unlabeled_X.bin', 'stl10_binary/class_names.txt', 'stl10_binary/fold_indices.txt']

        trainset_x_bin_path = '/home/data/maggie/stl10/stl10_binary/train_X.bin'
        trainset_y_bin_path = '/home/data/maggie/stl10/stl10_binary/train_y.bin'

        with open(trainset_x_bin_path, 'rb') as trainset_x_bin_file:
            images = np.fromfile(trainset_x_bin_file, dtype=np.uint8)
            # print("images: ",images)                                                                #   images:  [146 146 146 ... 170 174 170]
            # print("images.type: ",type(images))                                                     #   images.type:  <class 'numpy.ndarray'>
            # print("images.shape: ",images.shape)                                                    #   images.shape:  (138240000,)
            images = images.reshape(-1, 3, 96, 96)                                                     
            images = images.transpose([0, 1, 3, 2])      
            # print("images: ",images)
            # print("images.type: ",type(images))                                                     #   images.type:  <class 'numpy.ndarray'>
            # print("images.shape: ",images.shape)                                                    #   images.shape:  (5000, 3, 96, 96)                       
        
        with open(trainset_y_bin_path, 'rb') as trainset_y_bin_file:
            labels = np.fromfile(trainset_y_bin_file, dtype=np.uint8) - 1  # 0-based
            labels = np.int32(labels)
            # print("labels: ",labels)                                                                #   labels:  [1 5 1 ... 1 7 5]
            # print("labels.type: ",type(labels))                                                     #   labels.type:  <class 'numpy.ndarray'>
            # print("labels.shape: ",labels.shape)                                                    #   labels.shape:  (5000,)
                                           
    images = images.reshape(-1, 3, 96, 96)                                                          #   SVHN原是96x96像素，但为了stylegan2ada使用，需改成2的幂次方                                               
    images = np.pad(images, [(0,0),(0,0),(16,16), (16,16)], 'constant', constant_values=0)          
    """
    第1轴两边缘分别填充0个、0个数值0
    第2轴两边缘分别填充0个、0个数值0
    第3轴两边缘分别填充2个、2个数值0
    第4轴两边缘分别填充2个、2个数值0
    """    
    # print("images: ",images)                                                                
    # print("images.type: ",type(images))                                                     #   images.type:  <class 'numpy.ndarray'>
    # print("images.shape: ",images.shape)                                                    #   images.shape:  (5000, 3, 128, 128)
    images = images.transpose([0, 2, 3, 1])
    # print("images[0]: ",images[0])
    # print("images.type: ",type(images))                                                     #   images.type:  <class 'numpy.ndarray'>
    # print("images.dtype: ",images.dtype)                                                    #   images.dtype:  uint8
    # print("images.shape: ",images.shape)                                                    #   images.shape:  (5000, 128, 128, 3)

    assert images.shape == (5000, 128, 128, 3) and images.dtype == np.uint8
    assert labels.shape == (5000,) and labels.dtype in [np.int32, np.int64]
    assert np.min(images) == 0 and np.max(images) == 255
    assert np.min(labels) == 0 and np.max(labels) == 9
    max_idx = maybe_min(len(images), max_images)

    def iterate_images():
        for idx, img in enumerate(images):
            yield dict(img=img, label=int(labels[idx]))
            if idx >= max_idx-1:
                break

    return max_idx, iterate_images()

def open_imagenet(tarball: str, *, max_images: Optional[int]):                                  #   ILSVRC2012_img_train.tar
    
    images = []
    labels = []
    with tarfile.open(tarball, 'r:gz') as tar:  
        print(tar.getnames())                                   #    train


        os.makedirs("/home/data/ImageNet", exist_ok=True)

        train_dataset = torchvision.datasets.ImageNet(                                             #   用 torchvision.datasets.MNIST类的构造函数返回值给DataLoader的参数 dataset: torch.utils.data.dataset.Dataset[T_co]赋值 https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset
            "/home/data/ImageNet",
            split='train',
            download=False,
            transform=transforms.Compose(                               #   组合多个图像变换
                [
                    transforms.Resize(crop_size),                       #   通过Resize(size, interpolation=Image.BILINEAR)函数将输入的图像转换为期望的尺寸，此处期望的输出size=img_size
                    transforms.CenterCrop(crop_size),                    #   基于给定输入图像的中心，按照期望的尺寸（img_size）裁剪图像！！！解决了ImageNet数据集中个别样本size不符合3x256x256引发的问题
                    transforms.ToTensor(),                                  #   将PIL图像或者ndArry数据转换为tensor
                    transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
                ]     
            ),
        ) 


        #   实例dataset
        cle_dataset = MaggieDataset(args)
        cle_train_dataset = cle_dataset.traindataset()
        cle_test_dataset = cle_dataset.testdataset()
        
        #   实例dataloader
        cle_dataloader = MaggieDataloader(args,cle_train_dataset,cle_test_dataset)
        cle_train_dataloader = cle_dataloader.traindataloader()
        # cle_test_dataloader = cle_dataloader.testdataloader()

        dataloader = cle_train_dataloader
        xset_tensor = []
        for img_index in range(len(dataloader.dataset)):
            xset_tensor.append(dataloader.dataset[img_index][0])
        xset_tensor = torch.stack(xset_tensor)       
    labels = np.concatenate(labels)

        yset_tensor = []
        for img_index in range(len(dataloader.dataset)):
            yset_tensor.append(dataloader.dataset[img_index][1])
        yset_tensor = LongTensor(yset_tensor)    
    labels = np.concatenate(labels)
                        
        raise Exception("stop here")
        images = XXX
        labels = XXX

    assert images.shape == (5000, 1024, 1024, 3) and images.dtype == np.uint8
    assert labels.shape == (5000,) and labels.dtype in [np.int32, np.int64]
    assert np.min(images) == 0 and np.max(images) == 255
    assert np.min(labels) == 0 and np.max(labels) == 999        #   共1000个样本
    max_idx = maybe_min(len(images), max_images)

    def iterate_images():
        for idx, img in enumerate(images):
            yield dict(img=img, label=int(labels[idx]))
            if idx >= max_idx-1:
                break

    return max_idx, iterate_images()

#------------------------------------maggie----------------------------------


def error(msg):
    print('Error: ' + msg)
    sys.exit(1)

def maybe_min(a: int, b: Optional[int]) -> int:
    if b is not None:
        return min(a, b)
    return a

def file_ext(name: Union[str, Path]) -> str:
    return str(name).split('.')[-1]

def is_image_ext(fname: Union[str, Path]) -> bool:
    ext = file_ext(fname).lower()
    return f'.{ext}' in PIL.Image.EXTENSION # type: ignore

def open_image_folder(source_dir, *, max_images: Optional[int]):
    input_images = [str(f) for f in sorted(Path(source_dir).rglob('*')) if is_image_ext(f) and os.path.isfile(f)]

    # Load labels.
    labels = {}
    meta_fname = os.path.join(source_dir, 'dataset.json')
    if os.path.isfile(meta_fname):
        with open(meta_fname, 'r') as file:
            labels = json.load(file)['labels']
            if labels is not None:
                labels = { x[0]: x[1] for x in labels }
            else:
                labels = {}

    max_idx = maybe_min(len(input_images), max_images)

    def iterate_images():
        for idx, fname in enumerate(input_images):
            arch_fname = os.path.relpath(fname, source_dir)
            arch_fname = arch_fname.replace('\\', '/')
            img = np.array(PIL.Image.open(fname))
            yield dict(img=img, label=labels.get(arch_fname))
            if idx >= max_idx-1:
                break
    return max_idx, iterate_images()

def open_image_zip(source, *, max_images: Optional[int]):
    with zipfile.ZipFile(source, mode='r') as z:
        input_images = [str(f) for f in sorted(z.namelist()) if is_image_ext(f)]

        # Load labels.
        labels = {}
        if 'dataset.json' in z.namelist():
            with z.open('dataset.json', 'r') as file:
                labels = json.load(file)['labels']
                if labels is not None:
                    labels = { x[0]: x[1] for x in labels }
                else:
                    labels = {}

    max_idx = maybe_min(len(input_images), max_images)

    def iterate_images():
        with zipfile.ZipFile(source, mode='r') as z:
            for idx, fname in enumerate(input_images):
                with z.open(fname, 'r') as file:
                    img = PIL.Image.open(file) # type: ignore
                    img = np.array(img)
                yield dict(img=img, label=labels.get(fname))
                if idx >= max_idx-1:
                    break
    return max_idx, iterate_images()

def open_lmdb(lmdb_dir: str, *, max_images: Optional[int]):
    import cv2  # pip install opencv-python
    import lmdb  # pip install lmdb # pylint: disable=import-error

    with lmdb.open(lmdb_dir, readonly=True, lock=False).begin(write=False) as txn:
        max_idx = maybe_min(txn.stat()['entries'], max_images)

    def iterate_images():
        with lmdb.open(lmdb_dir, readonly=True, lock=False).begin(write=False) as txn:
            for idx, (_key, value) in enumerate(txn.cursor()):
                try:
                    try:
                        img = cv2.imdecode(np.frombuffer(value, dtype=np.uint8), 1)
                        if img is None:
                            raise IOError('cv2.imdecode failed')
                        img = img[:, :, ::-1] # BGR => RGB
                    except IOError:
                        img = np.array(PIL.Image.open(io.BytesIO(value)))
                    yield dict(img=img, label=None)
                    if idx >= max_idx-1:
                        break
                except:
                    print(sys.exc_info()[1])

    return max_idx, iterate_images()

def open_cifar10(tarball: str, *, max_images: Optional[int]):
    images = []
    labels = []

    with tarfile.open(tarball, 'r:gz') as tar:                                                  #   打开gz格式压缩包
        for batch in range(1, 6):
            member = tar.getmember(f'cifar-10-batches-py/data_batch_{batch}')                   #   获取数据集多个batch压缩包
            with tar.extractfile(member) as file:
                data = pickle.load(file, encoding='latin1')                                     #   从batch压缩包中提取样本
            images.append(data['data'].reshape(-1, 3, 32, 32))                                  #   将提取的样本重塑形(-1,3,32,32) 其中，-1指的是-1所在的位置维度值会被自动计算
            labels.append(data['labels'])

    images = np.concatenate(images)
    labels = np.concatenate(labels)
    images = images.transpose([0, 2, 3, 1]) # NCHW -> NHWC                                      #   通道值放到第四位
    assert images.shape == (50000, 32, 32, 3) and images.dtype == np.uint8                      #   断言两个条件
    assert labels.shape == (50000,) and labels.dtype in [np.int32, np.int64]                    #   断言两个条件
    assert np.min(images) == 0 and np.max(images) == 255                                        #   断言样本像素值域
    assert np.min(labels) == 0 and np.max(labels) == 9

    max_idx = maybe_min(len(images), max_images)

    def iterate_images():
        for idx, img in enumerate(images):
            yield dict(img=img, label=int(labels[idx]))                                         #   生成数据集字典[(img0,label0),(img1,label1),...(imgn,labeln)]
            if idx >= max_idx-1:
                break

    return max_idx, iterate_images()

def open_mnist(images_gz: str, *, max_images: Optional[int]):
    labels_gz = images_gz.replace('-images-idx3-ubyte.gz', '-labels-idx1-ubyte.gz')
    assert labels_gz != images_gz
    images = []
    labels = []

    with gzip.open(images_gz, 'rb') as f:
        images = np.frombuffer(f.read(), np.uint8, offset=16)                                   #   读取数据集样本
    with gzip.open(labels_gz, 'rb') as f:
        labels = np.frombuffer(f.read(), np.uint8, offset=8)

    images = images.reshape(-1, 28, 28)
    images = np.pad(images, [(0,0), (2,2), (2,2)], 'constant', constant_values=0)               #   np.pad(images, [(0,0), (2,2), (2,2)] 把1x28x28的样本填充为1x32x32的样本
    """
    
    第1轴两边缘分别填充0个、0个数值0
    第2轴两边缘分别填充2个、2个数值0
    第2轴两边缘分别填充2个、2个数值0

    """
    assert images.shape == (60000, 32, 32) and images.dtype == np.uint8
    assert labels.shape == (60000,) and labels.dtype == np.uint8
    assert np.min(images) == 0 and np.max(images) == 255
    assert np.min(labels) == 0 and np.max(labels) == 9

    max_idx = maybe_min(len(images), max_images)

    def iterate_images():
        for idx, img in enumerate(images):
            yield dict(img=img, label=int(labels[idx]))
            if idx >= max_idx-1:
                break

    return max_idx, iterate_images()

def make_transform(
    transform: Optional[str],
    output_width: Optional[int],
    output_height: Optional[int],
    resize_filter: str
) -> Callable[[np.ndarray], Optional[np.ndarray]]:
    resample = { 'box': PIL.Image.BOX, 'lanczos': PIL.Image.LANCZOS }[resize_filter]
    def scale(width, height, img):
        w = img.shape[1]
        h = img.shape[0]
        if width == w and height == h:
            return img
        img = PIL.Image.fromarray(img)
        ww = width if width is not None else w
        hh = height if height is not None else h
        img = img.resize((ww, hh), resample)
        return np.array(img)

    def center_crop(width, height, img):
        crop = np.min(img.shape[:2])
        img = img[(img.shape[0] - crop) // 2 : (img.shape[0] + crop) // 2, (img.shape[1] - crop) // 2 : (img.shape[1] + crop) // 2]
        img = PIL.Image.fromarray(img, 'RGB')
        img = img.resize((width, height), resample)
        return np.array(img)

    def center_crop_wide(width, height, img):
        ch = int(np.round(width * img.shape[0] / img.shape[1]))
        if img.shape[1] < width or ch < height:
            return None

        img = img[(img.shape[0] - ch) // 2 : (img.shape[0] + ch) // 2]
        img = PIL.Image.fromarray(img, 'RGB')
        img = img.resize((width, height), resample)
        img = np.array(img)

        canvas = np.zeros([width, width, 3], dtype=np.uint8)
        canvas[(width - height) // 2 : (width + height) // 2, :] = img
        return canvas

    if transform is None:
        return functools.partial(scale, output_width, output_height)
    if transform == 'center-crop':
        if (output_width is None) or (output_height is None):
            error ('must specify --width and --height when using ' + transform + 'transform')
        return functools.partial(center_crop, output_width, output_height)
    if transform == 'center-crop-wide':
        if (output_width is None) or (output_height is None):
            error ('must specify --width and --height when using ' + transform + ' transform')
        return functools.partial(center_crop_wide, output_width, output_height)
    assert False, 'unknown transform'

def open_dataset(source, *, max_images: Optional[int]):
    if os.path.isdir(source):
        if source.rstrip('/').endswith('_lmdb'):
            return open_lmdb(source, max_images=max_images)
        else:
            return open_image_folder(source, max_images=max_images)
    elif os.path.isfile(source):

        if os.path.basename(source) == 'cifar-10-python.tar.gz':
            return open_cifar10(source, max_images=max_images)

        elif os.path.basename(source) == 'train-images-idx3-ubyte.gz':
            return open_mnist(source, max_images=max_images)

        elif os.path.basename(source) == 'train-images-idx3-ubyte.gz':            
            return open_kmnist(source, max_images=max_images)     

        elif os.path.basename(source) == 'cifar-100-python.tar.gz':
            return open_cifar100(source, max_images=max_images)

        elif os.path.basename(source) == 'train_32x32.mat':                                 #   /home/data/maggie/svhn/train_32x32.mat
            return open_svhn(source, max_images=max_images)         
        
        elif os.path.basename(source) == 'stl10_binary.tar.gz':
            return open_stl10(source, max_images=max_images)

        elif os.path.basename(source) == 'ILSVRC2012_img_train.tar':
            return open_imagenet(source, max_images=max_images)

        elif file_ext(source) == 'zip':
            return open_image_zip(source, max_images=max_images)
        else:
            assert False, 'unknown archive type'
    else:
        error(f'Missing input file or directory: {source}')

def open_dest(dest: str) -> Tuple[str, Callable[[str, Union[bytes, str]], None], Callable[[], None]]:
    dest_ext = file_ext(dest)

    if dest_ext == 'zip':
        if os.path.dirname(dest) != '':
            os.makedirs(os.path.dirname(dest), exist_ok=True)
        zf = zipfile.ZipFile(file=dest, mode='w', compression=zipfile.ZIP_STORED)
        def zip_write_bytes(fname: str, data: Union[bytes, str]):
            zf.writestr(fname, data)
        return '', zip_write_bytes, zf.close
    else:
        # If the output folder already exists, check that is is
        # empty.
        #
        # Note: creating the output directory is not strictly
        # necessary as folder_write_bytes() also mkdirs, but it's better
        # to give an error message earlier in case the dest folder
        # somehow cannot be created.
        if os.path.isdir(dest) and len(os.listdir(dest)) != 0:
            error('--dest folder must be empty')
        os.makedirs(dest, exist_ok=True)

        def folder_write_bytes(fname: str, data: Union[bytes, str]):
            os.makedirs(os.path.dirname(fname), exist_ok=True)
            with open(fname, 'wb') as fout:
                if isinstance(data, str):
                    data = data.encode('utf8')
                fout.write(data)
        return dest, folder_write_bytes, lambda: None

@click.command()
@click.pass_context
@click.option('--source', help='Directory or archive name for input dataset', required=True, metavar='PATH')
@click.option('--dest', help='Output directory or archive name for output dataset', required=True, metavar='PATH')
@click.option('--max-images', help='Output only up to `max-images` images', type=int, default=None)
@click.option('--resize-filter', help='Filter to use when resizing images for output resolution', type=click.Choice(['box', 'lanczos']), default='lanczos', show_default=True)
@click.option('--transform', help='Input crop/resize mode', type=click.Choice(['center-crop', 'center-crop-wide']))
@click.option('--width', help='Output width', type=int)
@click.option('--height', help='Output height', type=int)
def convert_dataset(
    ctx: click.Context,
    source: str,
    dest: str,
    max_images: Optional[int],
    transform: Optional[str],
    resize_filter: str,
    width: Optional[int],
    height: Optional[int]
):
    """Convert an image dataset into a dataset archive usable with StyleGAN2 ADA PyTorch.

    The input dataset format is guessed from the --source argument:

    \b
    --source *_lmdb/                    Load LSUN dataset
    --source cifar-10-python.tar.gz     Load CIFAR-10 dataset
    --source train-images-idx3-ubyte.gz Load MNIST dataset
    --source path/                      Recursively load all images from path/
    --source dataset.zip                Recursively load all images from dataset.zip

    Specifying the output format and path:

    \b
    --dest /path/to/dir                 Save output files under /path/to/dir
    --dest /path/to/dataset.zip         Save output files into /path/to/dataset.zip

    The output dataset format can be either an image folder or an uncompressed zip archive.
    Zip archives makes it easier to move datasets around file servers and clusters, and may
    offer better training performance on network file systems.

    Images within the dataset archive will be stored as uncompressed PNG.
    Uncompresed PNGs can be efficiently decoded in the training loop.

    Class labels are stored in a file called 'dataset.json' that is stored at the
    dataset root folder.  This file has the following structure:

    \b
    {
        "labels": [
            ["00000/img00000000.png",6],
            ["00000/img00000001.png",9],
            ... repeated for every image in the datase
            ["00049/img00049999.png",1]
        ]
    }

    If the 'dataset.json' file cannot be found, the dataset is interpreted as
    not containing class labels.

    Image scale/crop and resolution requirements:

    Output images must be square-shaped and they must all have the same power-of-two
    dimensions.

    To scale arbitrary input image size to a specific width and height, use the
    --width and --height options.  Output resolution will be either the original
    input resolution (if --width/--height was not specified) or the one specified with
    --width/height.

    Use the --transform=center-crop or --transform=center-crop-wide options to apply a
    center crop transform on the input image.  These options should be used with the
    --width and --height options.  For example:

    \b
    python dataset_tool.py --source LSUN/raw/cat_lmdb --dest /tmp/lsun_cat \\
        --transform=center-crop-wide --width 512 --height=384
    """

    PIL.Image.init() # type: ignore

    if dest == '':
        ctx.fail('--dest output filename or directory must not be an empty string')

    num_files, input_iter = open_dataset(source, max_images=max_images)
    archive_root_dir, save_bytes, close_dest = open_dest(dest)

    transform_image = make_transform(transform, width, height, resize_filter)

    dataset_attrs = None

    labels = []
    for idx, image in tqdm(enumerate(input_iter), total=num_files):
        idx_str = f'{idx:08d}'
        archive_fname = f'{idx_str[:5]}/img{idx_str}.png'

        # Apply crop and resize.
        img = transform_image(image['img'])

        # Transform may drop images.
        if img is None:
            continue

        # Error check to require uniform image attributes across
        # the whole dataset.
        channels = img.shape[2] if img.ndim == 3 else 1
        cur_image_attrs = {
            'width': img.shape[1],
            'height': img.shape[0],
            'channels': channels
        }
        if dataset_attrs is None:
            dataset_attrs = cur_image_attrs
            width = dataset_attrs['width']
            height = dataset_attrs['height']
            if width != height:
                error(f'Image dimensions after scale and crop are required to be square.  Got {width}x{height}')
            if dataset_attrs['channels'] not in [1, 3]:
                error('Input images must be stored as RGB or grayscale')
            if width != 2 ** int(np.floor(np.log2(width))):
                error('Image width/height after scale and crop are required to be power-of-two')
        elif dataset_attrs != cur_image_attrs:
            err = [f'  dataset {k}/cur image {k}: {dataset_attrs[k]}/{cur_image_attrs[k]}' for k in dataset_attrs.keys()]
            error(f'Image {archive_fname} attributes must be equal across all images of the dataset.  Got:\n' + '\n'.join(err))

        # Save the image as an uncompressed PNG.
        img = PIL.Image.fromarray(img, { 1: 'L', 3: 'RGB' }[channels])
        image_bits = io.BytesIO()
        img.save(image_bits, format='png', compress_level=0, optimize=False)
        save_bytes(os.path.join(archive_root_dir, archive_fname), image_bits.getbuffer())
        labels.append([archive_fname, image['label']] if image['label'] is not None else None)

    metadata = {
        'labels': labels if all(x is not None for x in labels) else None
    }
    save_bytes(os.path.join(archive_root_dir, 'dataset.json'), json.dumps(metadata))
    close_dest()


if __name__ == "__main__":
    convert_dataset() # pylint: disable=no-value-for-parameter
