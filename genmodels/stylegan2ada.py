# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Train a GAN using the techniques described in the paper
"Training Generative Adversarial Networks with Limited Data"."""


#---train
from logging import error, exception
import os
import json
import tempfile
import torch
from torchvision.models.shufflenetv2 import channel_shuffle
import utils.stylegan2ada.dnnlib as dnnlib       
from utils.stylegan2ada.training import training_loop
from utils.stylegan2ada.metrics import metric_main
from utils.stylegan2ada.torch_utils import training_stats
from utils.stylegan2ada.torch_utils import custom_ops

#----project
import copy
from time import perf_counter
import imageio
import numpy as np
import PIL.Image
from numpy.core.records import array
import torch.nn.functional as F
import utils.stylegan2ada.legacy as legacy
import utils.sampler
import re
from typing import List, Optional
import click
from torchvision.transforms import transforms


class MaggieStylegan2ada:

    def __init__(self, args):
        #   initialize the parameters
        self._args = args
        # self._snapshot_network_pkls = None  # snapshot_network_pkls list

#---------------------训练-----------------
    def snapshot_network_pkls(self):        
        return self._snapshot_network_pkls

    def train(self,exp_result_dir, stylegan2ada_config_kwargs):

        self._exp_result_dir = exp_result_dir
        self._stylegan2ada_config_kwargs = stylegan2ada_config_kwargs
        self.__train__()

    def __train__(self):
        snapshot_network_pkls = self.__trainmain__(self._args, self._exp_result_dir, **self._stylegan2ada_config_kwargs)
        self._snapshot_network_pkls = snapshot_network_pkls

    def __trainmain__(self, opt, exp_result_dir, **config_kwargs):                                                              #   train stylegan2ada的主函数
        print("running stylegan2ada train main()...............")

        #----------------------------maggie----------------------
        # print('maggie testing:')
        # print('opt = %s' % opt)
        # print('config_kwargs = %s' % config_kwargs)
        # outdir = exp_result_dir
        dry_run = opt.dry_run   
        #--------------------------------------------------------
            
        dnnlib.util.Logger(should_flush=True)
        
        # Setup training options.
        # try:
        #     run_desc, args = setup_training_loop_kwargs(**config_kwargs)                                                      #   **config_kwargs表示接收带key的字典
        # except Exception as err:
        #      ctx.fail(err)

        #------------------maggie-----------------
        run_desc, args = self.__setup_training_loop_kwargs__(**config_kwargs)                                                   #   解析模型训练的参数并赋值给args和存储路径run_desc
        # print('run_desc=%s' % run_desc)
        # print('args = %s' % args)
        #-----------------------------------------
        
        args.run_dir = os.path.join(exp_result_dir, f'{run_desc}')         
        # args.run_dir = f'{exp_result_dir}-{run_desc}'
        assert not os.path.exists(args.run_dir)             

        # Print options.
        print()
        print('Training options:')
        print(json.dumps(args, indent=2))
        print()
        print(f'Output directory:   {args.run_dir}')
        print(f'Training data:      {args.training_set_kwargs.path}')
        print(f'Training duration:  {args.total_kimg} kimg')
        print(f'Number of GPUs:     {args.num_gpus}')
        print(f'Number of images:   {args.training_set_kwargs.max_size}')
        print(f'Image resolution:   {args.training_set_kwargs.resolution}')
        print(f'Conditional model:  {args.training_set_kwargs.use_labels}')
        print(f'Dataset x-flips:    {args.training_set_kwargs.xflip}')
        print()

        # Dry run?
        if dry_run:
            print('Dry run; exiting.')
            return

        # Create output directory.
        print('Creating output directory...')
        os.makedirs(args.run_dir)
        with open(os.path.join(args.run_dir, 'training_options.json'), 'wt') as f:
            json.dump(args, f, indent=2)

        # Launch processes.
        print('Launching processes...')
        torch.multiprocessing.set_start_method('spawn')
        with tempfile.TemporaryDirectory() as temp_dir:
            if args.num_gpus == 1:

                # self.__subprocess_fn__(rank=0, args=args, temp_dir=temp_dir)

                #------------maggie add-----------------
                snapshot_network_pkls = self.__subprocess_fn__(rank=0, args=args, temp_dir=temp_dir)
                #---------------------------------------
            
            else:
                # torch.multiprocessing.spawn(fn=self.__subprocess_fn__, args=(args, temp_dir), nprocs=args.num_gpus)
                
                #------------------maggie add-----------------
                snapshot_network_pkls = torch.multiprocessing.spawn(fn=self.__subprocess_fn__, args=(args, temp_dir), nprocs=args.num_gpus)
                #---------------------------------------------
        #-----------maggie add---------
        return snapshot_network_pkls
        #------------------------------
    
    def __setup_training_loop_kwargs__(self,
        # General options (not included in desc).
        gpus       = None, # Number of GPUs: <int>, default = 1 gpu
        snap       = None, # Snapshot interval: <int>, default = 50 ticks
        metrics    = None, # List of metric names: [], ['fid50k_full'] (default), ...
        seed       = None, # Random seed: <int>, default = 0

        # Dataset.
        data       = None, # Training dataset (required): <path>
        cond       = None, # Train conditional model based on dataset labels: <bool>, default = False
        subset     = None, # Train with only N images: <int>, default = all
        mirror     = None, # Augment dataset with x-flips: <bool>, default = False

        # Base config.
        cfg        = None, # Base config: 'auto' (default), 'stylegan2', 'paper256', 'paper512', 'paper1024', 'cifar'
        gamma      = None, # Override R1 gamma: <float>
        kimg       = None, # Override training duration: <int>
        # batch      = None, # Override batch size: <int>
        batch_size = None, # Override batch size: <int>

        # Discriminator augmentation.
        aug        = None, # Augmentation mode: 'ada' (default), 'noaug', 'fixed'
        p          = None, # Specify p for 'fixed' (required): <float>
        target     = None, # Override ADA target for 'ada': <float>, default = depends on aug
        augpipe    = None, # Augmentation pipeline: 'blit', 'geom', 'color', 'filter', 'noise', 'cutout', 'bg', 'bgc' (default), ..., 'bgcfnc'

        # Transfer learning.
        resume     = None, # Load previous network: 'noresume' (default), 'ffhq256', 'ffhq512', 'ffhq1024', 'celebahq256', 'lsundog256', <file>, <url>
        freezed    = None, # Freeze-D: <int>, default = 0 discriminator layers

        # Performance options (not included in desc).
        fp32       = None, # Disable mixed-precision training: <bool>, default = False
        nhwc       = None, # Use NHWC memory format with FP16: <bool>, default = False
        allow_tf32 = None, # Allow PyTorch to use TF32 for matmul and convolutions: <bool>, default = False
        nobench    = None, # Disable cuDNN benchmarking: <bool>, default = False
        workers    = None, # Override number of DataLoader workers: <int>, default = 3
        pretrain_pkl_path =None, # pretrained stylegan2ADA model pkl path
    ):
        args = dnnlib.EasyDict()
        # ------------------------------------------
        # General options: gpus, snap, metrics, seed
        # ------------------------------------------

        if gpus is None:
            gpus = 1
        assert isinstance(gpus, int)
        if not (gpus >= 1 and gpus & (gpus - 1) == 0):
            raise Exception('--gpus must be a power of two')                                                                    #   gpu数目要么是1要么是2的倍数
        args.num_gpus = gpus

        if snap is None:
            snap = 50
        assert isinstance(snap, int)
        if snap < 1:
            raise Exception('--snap must be at least 1')                                                                        #   image和network的存储间隔=snap 默认是50
        args.image_snapshot_ticks = snap
        args.network_snapshot_ticks = snap

        if metrics is None:
            metrics = ['fid50k_full']                                                                                           #   fid50k_full是目录metric的一个键
        assert isinstance(metrics, list)
        if not all(metric_main.is_valid_metric(metric) for metric in metrics):
            raise Exception('\n'.join(['--metrics can only contain the following values:'] + metric_main.list_valid_metrics()))
        args.metrics = metrics

        if seed is None:
            seed = 0
        assert isinstance(seed, int)
        args.random_seed = seed                                                                                                 #   随机种子seed 默认是0

        # -----------------------------------
        # Dataset: data, cond, subset, mirror
        # -----------------------------------

        assert data is not None
        assert isinstance(data, str)
        # args.training_set_kwargs = dnnlib.EasyDict(class_name='training.dataset.ImageFolderDataset', path=data, use_labels=True, max_size=None, xflip=False)
        args.training_set_kwargs = dnnlib.EasyDict(class_name='utils.stylegan2ada.training.dataset.ImageFolderDataset', path=data, use_labels=True, max_size=None, xflip=False)  # 解析data给出的训练集路径

        args.data_loader_kwargs = dnnlib.EasyDict(pin_memory=True, num_workers=3, prefetch_factor=2)
        try:
            training_set = dnnlib.util.construct_class_by_name(**args.training_set_kwargs)                                      #   subclass of training.dataset.Dataset   #   加载data路径下的训练集样本
            args.training_set_kwargs.resolution = training_set.resolution                                                       #   be explicit about resolution                            #   解析样本分辨率（即样本内容）
            args.training_set_kwargs.use_labels = training_set.has_labels                                                       #   be explicit about labels                                #   解析标签
            args.training_set_kwargs.max_size = len(training_set)                                                               #   be explicit about dataset size                                  #   解析训练集样本数量（size）
            desc = training_set.name                                                                                            #   desc=cifar10                                                    #   把训练集名称赋值给desc变量
            del training_set # conserve memory                                                                                  #   为了节省内存，删除用过的training_set对象
        except IOError as err:
            raise Exception(f'--data: {err}')

        if cond is None:
            cond = False
        assert isinstance(cond, bool)
        if cond:                                                                                                                #   Cond有效后，变为条件GAN
            if not args.training_set_kwargs.use_labels:
                raise Exception('--cond=True requires labels specified in dataset.json')
            desc += '-cond'                                                                                                     #   if cond = true , desc=cifar10-true #   if cond = false , desc=cifar10
        else:                                                                                                                   #   否则，cond=false时，就不使用标签数据
            args.training_set_kwargs.use_labels = False

        if subset is not None:
            assert isinstance(subset, int)
            if not 1 <= subset <= args.training_set_kwargs.max_size:                                                            #   如果subset<1或subset>数据集的max_size, 显然不正常，要报错
                raise Exception(f'--subset must be between 1 and {args.training_set_kwargs.max_size}')
            desc += f'-subset{subset}'                                                                                          #   desc=cifar10-subset
            if subset < args.training_set_kwargs.max_size:                                                                      #   如果subset<数据集的max_size
                args.training_set_kwargs.max_size = subset                                                                      #   就将max_size更新为subset的size  
                args.training_set_kwargs.random_seed = args.random_seed

        if mirror is None:
            mirror = False
        assert isinstance(mirror, bool)
        if mirror:                                                                                                              #   如果mirror=true
            desc += '-mirror'                                                                                                   #   desc=cifar10-subset
            args.training_set_kwargs.xflip = True                                                                               #   就将数据集按x轴翻装（上下颠倒）

        # ------------------------------------
        # Base config: cfg, gamma, kimg, batch
        # ------------------------------------

        if cfg is None:
            cfg = 'auto'                                                                                                        #   cfg默认是auto
        assert isinstance(cfg, str)
        desc += f'-{cfg}'                                                                                                       #   desc=cifar10-subset-auto

        cfg_specs = {
            'auto':      dict(ref_gpus=-1, kimg=25000,  mb=-1, mbstd=-1, fmaps=-1,  lrate=-1,     gamma=-1,   ema=-1,  ramp=0.05, map=2), # Populated dynamically based on resolution and GPU count.
            'stylegan2': dict(ref_gpus=8,  kimg=25000,  mb=32, mbstd=4,  fmaps=1,   lrate=0.002,  gamma=10,   ema=10,  ramp=None, map=8), # Uses mixed-precision, unlike the original StyleGAN2.
            'paper256':  dict(ref_gpus=8,  kimg=25000,  mb=64, mbstd=8,  fmaps=0.5, lrate=0.0025, gamma=1,    ema=20,  ramp=None, map=8),
            'paper512':  dict(ref_gpus=8,  kimg=25000,  mb=64, mbstd=8,  fmaps=1,   lrate=0.0025, gamma=0.5,  ema=20,  ramp=None, map=8),
            'paper1024': dict(ref_gpus=8,  kimg=25000,  mb=32, mbstd=4,  fmaps=1,   lrate=0.002,  gamma=2,    ema=10,  ramp=None, map=8),
            'cifar':     dict(ref_gpus=2,  kimg=100000, mb=64, mbstd=32, fmaps=1,   lrate=0.0025, gamma=0.01, ema=500, ramp=0.05, map=2),
        }

        assert cfg in cfg_specs
        spec = dnnlib.EasyDict(cfg_specs[cfg])
        if cfg == 'auto':
            desc += f'{gpus:d}'                                                                                                 #   desc=cifar10-subset-auto1
            spec.ref_gpus = gpus
            res = args.training_set_kwargs.resolution
            spec.mb = max(min(gpus * min(4096 // res, 32), 64), gpus)                                                           #   keep gpu memory consumption at bay组织gpu内存消耗
            spec.mbstd = min(spec.mb // gpus, 4)                                                                                #   other hyperparams behave more predictably if mbstd group size remains fixed
            spec.fmaps = 1 if res >= 512 else 0.5                                                                               #   分辨率大于等于512时，fmaps=1 分辨率小于512时， fmaps=0.5
            spec.lrate = 0.002 if res >= 1024 else 0.0025
            spec.gamma = 0.0002 * (res ** 2) / spec.mb # heuristic formula
            spec.ema = spec.mb * 10 / 32

        # args.G_kwargs = dnnlib.EasyDict(class_name='training.networks.Generator', z_dim=512, w_dim=512, mapping_kwargs=dnnlib.EasyDict(), synthesis_kwargs=dnnlib.EasyDict())
        # args.D_kwargs = dnnlib.EasyDict(class_name='training.networks.Discriminator', block_kwargs=dnnlib.EasyDict(), mapping_kwargs=dnnlib.EasyDict(), epilogue_kwargs=dnnlib.EasyDict())
        args.G_kwargs = dnnlib.EasyDict(class_name='utils.stylegan2ada.training.networks.Generator', z_dim=512, w_dim=512, mapping_kwargs=dnnlib.EasyDict(), synthesis_kwargs=dnnlib.EasyDict())             #   赋值生成器，z维数512
        args.D_kwargs = dnnlib.EasyDict(class_name='utils.stylegan2ada.training.networks.Discriminator', block_kwargs=dnnlib.EasyDict(), mapping_kwargs=dnnlib.EasyDict(), epilogue_kwargs=dnnlib.EasyDict())    #   赋值判别器

        args.G_kwargs.synthesis_kwargs.channel_base = args.D_kwargs.channel_base = int(spec.fmaps * 32768)  
        args.G_kwargs.synthesis_kwargs.channel_max = args.D_kwargs.channel_max = 512
        args.G_kwargs.mapping_kwargs.num_layers = spec.map
        args.G_kwargs.synthesis_kwargs.num_fp16_res = args.D_kwargs.num_fp16_res = 4                                            #   enable mixed-precision training
        args.G_kwargs.synthesis_kwargs.conv_clamp = args.D_kwargs.conv_clamp = 256                                              #   clamp activations to avoid float16 overflow
        args.D_kwargs.epilogue_kwargs.mbstd_group_size = spec.mbstd

        args.G_opt_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', lr=spec.lrate, betas=[0,0.99], eps=1e-8)             #   Adam优化器·
        args.D_opt_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', lr=spec.lrate, betas=[0,0.99], eps=1e-8)
        
        # args.loss_kwargs = dnnlib.EasyDict(class_name='training.loss.StyleGAN2Loss', r1_gamma=spec.gamma)
        args.loss_kwargs = dnnlib.EasyDict(class_name='utils.stylegan2ada.training.loss.StyleGAN2Loss', r1_gamma=spec.gamma)

        args.total_kimg = spec.kimg
        args.batch_size = spec.mb
        args.batch_gpu = spec.mb // spec.ref_gpus
        args.ema_kimg = spec.ema
        args.ema_rampup = spec.ramp

        if cfg == 'cifar':
            args.loss_kwargs.pl_weight = 0                                                                                      #   disable path length regularization  如果是cifar，就让path length的正则化失效
            args.loss_kwargs.style_mixing_prob = 0                                                                              #   disable style mixing                混合风格失效
            args.D_kwargs.architecture = 'orig'                                                                                 #   disable residual skip connections   残差连接失效

        if gamma is not None:
            assert isinstance(gamma, float)
            if not gamma >= 0:
                raise Exception('--gamma must be non-negative')                                                                 #   desc时实验存放目录
            desc += f'-gamma{gamma:g}'                                                                                          #   desc=cifar10-subset-auto1-gamma
            args.loss_kwargs.r1_gamma = gamma

        if kimg is not None:
            assert isinstance(kimg, int)
            if not kimg >= 1:
                raise Exception('--kimg must be at least 1')
            desc += f'-kimg{kimg:d}'
            args.total_kimg = kimg

        # if batch is not None:
        #     assert isinstance(batch, int)
        #     if not (batch >= 1 and batch % gpus == 0):
        #         raise Exception('--batch must be at least 1 and divisible by --gpus')
        #     desc += f'-batch{batch}'
        #     args.batch_size = batch
        #     args.batch_gpu = batch // gpus

        if batch_size is not None:
            assert isinstance(batch_size, int)
            if not (batch_size >= 1 and batch_size % gpus == 0):
                raise Exception('--batch_size must be at least 1 and divisible by --gpus')
            desc += f'-batch{batch_size}'                                                                                       #   -batch是指results目录下的实验名称中的batch名称 desc=cifar10-subset-auto1-batch32
            args.batch_size = batch_size
            args.batch_gpu = batch_size // gpus

        # ---------------------------------------------------
        # Discriminator augmentation: aug, p, target, augpipe                                                                   #   stylegan2ada的创新点
        # ---------------------------------------------------

        if aug is None:
            aug = 'ada'
        else:
            assert isinstance(aug, str)
            desc += f'-{aug}'                                                                                                   #   desc=cifar10-subset-auto1-batch32-ada

        if aug == 'ada':
            args.ada_target = 0.6

        elif aug == 'noaug':
            pass

        elif aug == 'fixed':
            if p is None:
                raise Exception(f'--aug={aug} requires specifying --p')

        else:
            raise Exception(f'--aug={aug} not supported')

        if p is not None:
            assert isinstance(p, float)
            if aug != 'fixed':
                raise Exception('--p can only be specified with --aug=fixed')
            if not 0 <= p <= 1:
                raise Exception('--p must be between 0 and 1')
            desc += f'-p{p:g}'
            args.augment_p = p

        if target is not None:
            assert isinstance(target, float)
            if aug != 'ada':
                raise Exception('--target can only be specified with --aug=ada')
            if not 0 <= target <= 1:
                raise Exception('--target must be between 0 and 1')
            desc += f'-target{target:g}'
            args.ada_target = target

        assert augpipe is None or isinstance(augpipe, str)
        if augpipe is None:
            augpipe = 'bgc'
        else:
            if aug == 'noaug':
                raise Exception('--augpipe cannot be specified with --aug=noaug')
            desc += f'-{augpipe}'                                                                                               #   desc=cifar10-subset-auto1-batch32-ada-bgc

        augpipe_specs = {
            'blit':   dict(xflip=1, rotate90=1, xint=1),
            'geom':   dict(scale=1, rotate=1, aniso=1, xfrac=1),
            'color':  dict(brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1),
            'filter': dict(imgfilter=1),
            'noise':  dict(noise=1),
            'cutout': dict(cutout=1),
            'bg':     dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1),
            'bgc':    dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1),
            'bgcf':   dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1, imgfilter=1),
            'bgcfn':  dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1, imgfilter=1, noise=1),
            'bgcfnc': dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1, imgfilter=1, noise=1, cutout=1),
        }

        assert augpipe in augpipe_specs
        if aug != 'noaug':
            # args.augment_kwargs = dnnlib.EasyDict(class_name='training.augment.AugmentPipe', **augpipe_specs[augpipe])
            args.augment_kwargs = dnnlib.EasyDict(class_name='utils.stylegan2ada.training.augment.AugmentPipe', **augpipe_specs[augpipe])

        # ----------------------------------
        # Transfer learning: resume, freezed
        # ----------------------------------

        resume_specs = {
            'ffhq256':     'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/transfer-learning-source-nets/ffhq-res256-mirror-paper256-noaug.pkl',
            'ffhq512':     'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/transfer-learning-source-nets/ffhq-res512-mirror-stylegan2-noaug.pkl',
            'ffhq1024':    'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/transfer-learning-source-nets/ffhq-res1024-mirror-stylegan2-noaug.pkl',
            'celebahq256': 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/transfer-learning-source-nets/celebahq-res256-mirror-paper256-kimg100000-ada-target0.5.pkl',
            'lsundog256':  'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/transfer-learning-source-nets/lsundog-res256-paper256-kimg100000-noaug.pkl',
        }

        assert resume is None or isinstance(resume, str)
        if resume is None:
            resume = 'noresume'
        elif resume == 'noresume':
            desc += '-noresume'                                                                                                 #   desc=cifar10-subset-auto1-batch32-ada-bgc-noresume                      
        elif resume in resume_specs:
            desc += f'-resume{resume}'
            args.resume_pkl = resume_specs[resume]                                                                              #   predefined url
        else:
            desc += '-resumecustom'
            args.resume_pkl = resume                                                                                            #   custom path or url

        # maggie-----------
        if pretrain_pkl_path is not None:
            args.resume_pkl = pretrain_pkl_path  
            print("args.resume_pkl:",args.resume_pkl)
        #---------------------
        
        if resume != 'noresume':
            args.ada_kimg = 100                                                                                                 #   make ADA react faster at the beginning
            args.ema_rampup = None                                                                                              #   disable EMA rampup

        if freezed is not None:
            assert isinstance(freezed, int)
            if not freezed >= 0:
                raise Exception('--freezed must be non-negative')
            desc += f'-freezed{freezed:d}'
            args.D_kwargs.block_kwargs.freeze_layers = freezed

        # -------------------------------------------------
        # Performance options: fp32, nhwc, nobench, workers
        # -------------------------------------------------

        if fp32 is None:
            fp32 = False
        assert isinstance(fp32, bool)
        if fp32:
            args.G_kwargs.synthesis_kwargs.num_fp16_res = args.D_kwargs.num_fp16_res = 0
            args.G_kwargs.synthesis_kwargs.conv_clamp = args.D_kwargs.conv_clamp = None

        if nhwc is None:
            nhwc = False
        assert isinstance(nhwc, bool)
        if nhwc:
            args.G_kwargs.synthesis_kwargs.fp16_channels_last = args.D_kwargs.block_kwargs.fp16_channels_last = True

        if nobench is None:
            nobench = False
        assert isinstance(nobench, bool)
        if nobench:
            args.cudnn_benchmark = False

        if allow_tf32 is None:
            allow_tf32 = False
        assert isinstance(allow_tf32, bool)
        if allow_tf32:
            args.allow_tf32 = True

        if workers is not None:
            assert isinstance(workers, int)
            if not workers >= 1:
                raise Exception('--workers must be at least 1')
            args.data_loader_kwargs.num_workers = workers

        return desc, args                                                                                                       #   des是存储目录， args是训练参数
    
    def __subprocess_fn__(self,rank, args, temp_dir):
        dnnlib.util.Logger(file_name=os.path.join(args.run_dir, 'log.txt'), file_mode='a', should_flush=True)

        # Init torch.distributed.
        if args.num_gpus > 1:
            init_file = os.path.abspath(os.path.join(temp_dir, '.torch_distributed_init'))
            if os.name == 'nt':
                init_method = 'file:///' + init_file.replace('\\', '/')
                torch.distributed.init_process_group(backend='gloo', init_method=init_method, rank=rank, world_size=args.num_gpus)
            else:
                init_method = f'file://{init_file}'
                torch.distributed.init_process_group(backend='nccl', init_method=init_method, rank=rank, world_size=args.num_gpus)

        # Init torch_utils.
        sync_device = torch.device('cuda', rank) if args.num_gpus > 1 else None
        training_stats.init_multiprocessing(rank=rank, sync_device=sync_device)
        if rank != 0:
            custom_ops.verbosity = 'none'

        # Execute training loop.
        # training_loop.training_loop(rank=rank, **args)                                                                        #   调用utils/training/training_loop.py中定义爹函数开始逐轮训练

        #--------maggie add----------
        snapshot_network_pkls = training_loop.training_loop(rank=rank, **args)                                                  #   调用utils/training/training_loop.py中定义爹函数开始逐轮训练
        #----------------------------
    
        return snapshot_network_pkls

#-----------------投影--------------------
    def wyset(self):
        return self.projected_w_set,self.projected_y_set

    def project(self,exp_result_dir, ori_x_set = None, ori_y_set = None,batch_index=None):
        self._exp_result_dir = exp_result_dir
        self._batch_index = batch_index
        projected_w_set, projected_y_set = self.__projectmain__(self._args, self._exp_result_dir,ori_x_set, ori_y_set)
        self.projected_w_set = projected_w_set
        self.projected_y_set = projected_y_set

    def __projectmain__(self, opt, exp_result_dir,ori_x_set, ori_y_set):
        print("running projecting main()..............")

        if ori_x_set is not None :      # 实验都是没有view dataset的，都由此进入投影
            self.ori_x_set = ori_x_set
            self.ori_y_set = ori_y_set
            print("Project original images from images tensor set !")
            projected_w_set, projected_y_set = self.__ramxyproject__()

        else:
            print("Project original images from view dataset path !")
            if opt.target_fname == None:
                print(f'Project samples of the *{self._args.dataset}* dataset !')
                projected_w_set, projected_y_set = self.__run_projection_dataset_fromviewfolder(opt,exp_result_dir)

            elif opt.target_fname != None:
                print('Project single sample of the *{self._args.dataset}* dataset')
                projected_w_set, projected_y_set = self.__run_projection__(
                    network_pkl = opt.gen_network_pkl,
                    target_fname = opt.target_fname,
                    outdir = exp_result_dir,
                    save_video = opt.save_video,
                    seed = opt.seed,
                    num_steps = opt.num_steps,    
                    image_name = 'test'
                )

        return projected_w_set, projected_y_set         

    def __labelnames__(self):
        opt = self._args
        # print("opt.dataset:",opt.dataset)
        
        label_names = []
        
        if opt.dataset == 'cifar10':
            label_names = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
            #   label_names = ['飞机'，'汽车'，'鸟'，'猫'，'鹿'，'狗'，'青蛙'，'马'，'船'，'卡车']

        elif opt.dataset == 'cifar100': # = cle_train_dataloader.dataset.classes
            label_names = ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm']
        
        elif opt.dataset =='svhn':
            label_names = ['0','1','2','3','4','5','6','7','8','9']

        elif opt.dataset =='kmnist':
            label_names = ['0','1','2','3','4','5','6','7','8','9']
        
        elif opt.dataset =='stl10': # cle_train_dataloader.dataset.classes 标签序号是0-9, dataloader 已调整数字0的标签为0
            label_names = ['airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck']
            #   label_names = ['飞机'，'鸟'，'汽车'，'猫'，'鹿'，'狗'，'马'，'猴子'，'船'，'卡车'] 
        
        elif opt.dataset =='imagenetmixed10':
            label_names = ['dog,','bird','insect','monkey','car','feline','truck','fruit','fungus','boat']        
            #   label_names = ['狗，'，'鸟'，'昆虫'，'猴子'，'汽车'，'猫'，'卡车'，'水果'，'真菌'，'船']
        else:
            raise error            
        
        return label_names

    def __ramxyproject__(self):
        opt = self._args
        exp_result_dir = self._exp_result_dir
        exp_result_dir = os.path.join(exp_result_dir,f'project-{opt.dataset}-trainset')
        # exp_result_dir = os.path.join(exp_result_dir,f'project-{opt.dataset}-testset')

        os.makedirs(exp_result_dir,exist_ok=True)    

        target_x_set = self.ori_x_set
        target_y_set = self.ori_y_set

        projected_x_set = []
        projected_y_set = []

        for index in range(len(target_x_set)):                                                                            
            
            if  self._args.project_target_num != None:
                if index < self._args.project_target_num:                                                                          
                    projected_w, projected_y = self.__xyproject__(
                        network_pkl = opt.gen_network_pkl,
                        target_pil = target_x_set[index],
                        outdir = exp_result_dir,
                        save_video = opt.save_video,
                        seed = opt.seed,
                        num_steps = opt.num_steps,
                        projected_img_index = index + opt.batch_size * self._batch_index,
                        laber_index = target_y_set[index]
                    )
                    projected_x_set.append(projected_w)
                    projected_y_set.append(projected_y)         
            else:
                projected_w, projected_y = self.__xyproject__(
                    network_pkl = opt.gen_network_pkl,
                    target_pil = target_x_set[index],
                    outdir = exp_result_dir,
                    save_video = opt.save_video,
                    seed = opt.seed,
                    num_steps = opt.num_steps,
                    projected_img_index = index + opt.batch_size * self._batch_index,
                    laber_index = target_y_set[index]
                )
                projected_x_set.append(projected_w)
                projected_y_set.append(projected_y)         
                     
        print('Finished dataset projecting !')
        return projected_x_set,projected_y_set         

    def __xyproject__(self,                                            
        network_pkl: str,               #   使用的styleGAN2-ada网络pkl
        target_pil,                     #   target_pil是待投影的图片
        outdir: str,                    #   输出路径
        save_video: bool,               #   是否要存储视频
        seed: int,                      #   种子设定
        num_steps: int,                  #   默认是1000，指的是投影优化器参数更新的步长
        projected_img_index:int,         #   
        laber_index: int
    ):

        print(f"projecting {projected_img_index:08d} image:")

        np.random.seed(seed)                                                                                                    
        torch.manual_seed(seed)

        # Load networks.
        device = torch.device('cuda')
        with dnnlib.util.open_url(network_pkl) as fp:
            G = legacy.load_network_pkl(fp)['G_ema'].requires_grad_(False).to(device)                         #   load_network_pkl（）会调用到persistency中的class解释器。
        
        if self._args.dataset =='cifar10' or self._args.dataset =='cifar100' or self._args.dataset =='svhn' or self._args.dataset =='stl10' or self._args.dataset =='imagenetmixed10':
            if self._args.dataset =='svhn' or self._args.dataset =='stl10':
                # print("target_pil.shape:",target_pil.shape)           #   target_pil.shape: (3, 32, 32)
                target_pil = target_pil.transpose([1, 2, 0])
                # print("target_pil.shape:",target_pil.shape)           #  target_pil.shape: (32, 32, 3)

            target_pil = PIL.Image.fromarray(target_pil, 'RGB')     #   fromarray接收的是WHC格式或WH格式
        
            w, h = target_pil.size
            s = min(w, h)
            target_pil = target_pil.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
            target_pil = target_pil.resize((G.img_resolution, G.img_resolution), PIL.Image.LANCZOS)

            target_uint8 = np.array(target_pil, dtype=np.uint8)
            target_uint8 = target_uint8.transpose([2, 0, 1])
            # print("target_uint8.shape:",target_uint8.shape)                                 #         target_uint8.shape: (3, 64, 64)     

        elif self._args.dataset == 'kmnist' or self._args.dataset == 'mnist':
            target_pil = target_pil.numpy()                         
            target_pil = PIL.Image.fromarray(target_pil, 'L')     #   fromarray接收的是WHC格式或WH格式 28,28
            
            w, h = target_pil.size
            s = min(w, h)
            target_pil = target_pil.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
            target_pil = target_pil.resize((G.img_resolution, G.img_resolution), PIL.Image.LANCZOS) #32,32

            target_uint8 = np.array(target_pil, dtype=np.uint8)                                 #   32,32
            # print("target_uint8.shape:",target_uint8.shape)                                 #   target_uint8.shape: (32, 32)
            target_uint8 = torch.tensor(target_uint8).unsqueeze(0)
            target_uint8 = target_uint8.numpy()


        projected_w_steps = self.__project__(
            G,
            target=torch.tensor(target_uint8, device=device),                                              
            num_steps=num_steps,
            device=device,
            verbose=True
        )        

        # #------------maggie---------
        # print("projected_w_steps: ",projected_w_steps)                                                                        #   projected_w_steps.shape:  torch.Size([1000, 8, 512])        8是指复制的八份512向量，因为要送到stylegan2ada网络的mapping模块
        # print("projected_w_steps.shape: ",projected_w_steps.shape)                                                            #   projected_w_steps.shape:  torch.Size([1000, 8, 512])
        # #---------------------------

        # Render debug output: optional video and projected image and W vector.
        os.makedirs(outdir, exist_ok=True)

        classification = self.__labelnames__() 
        print("label_names:",classification)        #   label_names: ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0']

        label_name = classification[int(laber_index)]
        print(f"label = {laber_index:04d}-{classification[int(laber_index)]}")

        # 存原图
        # target_pil.save(f'{outdir}/original-{projected_img_index:08d}-{int(laber_index)}-{label_name}.png')                   

        # 存投影生成图
        projected_w = projected_w_steps[-1]                                                #   projected_w.shape:  torch.Size([8, 512])    
        synth_image = G.synthesis(projected_w.unsqueeze(0), noise_mode='const')            #   projected_w.unsqueeze(0).shape:  torch.Size([1, 8, 512])
        # print("synth_image.shape:",synth_image.shape)                       # synth_image.shape: torch.Size([1, 1, 32, 32])  
        synth_image = (synth_image + 1) * (255/2)
        synth_image = synth_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
        # print("synth_image.shape:",synth_image.shape)                                                                           #   synth_image.shape: (32, 32, 1)
        # print("synth_image.dtype:",synth_image.dtype)                                                                           #   synth_image.dtype: uint8

        if self._args.dataset != 'kmnist' and self._args.dataset != 'mnist':
            synth_image = PIL.Image.fromarray(synth_image, 'RGB')
        elif self._args.dataset == 'kmnist' or self._args.dataset == 'mnist':
            # print("synth_image.shape:",synth_image.shape)               #   synth_image.shape: (32, 32, 1) 现在要转变成（32,32）
            synth_image = synth_image.transpose([2, 0, 1])
            synth_image = synth_image[0]
            # print("synth_image.shape:",synth_image.shape)               #   synth_image.shape: (32, 32)
            synth_image = PIL.Image.fromarray(synth_image, 'L')

        #-----maggie注释 不存投影 20210909
        # synth_image.save(f'{outdir}/projected-{projected_img_index:08d}-{int(laber_index)}-{label_name}.png')

        #------------写成npz文件-------------------
        np.savez(f'{outdir}/{projected_img_index:08d}-{int(laber_index)}-{label_name}-projected_w.npz', w=projected_w.unsqueeze(0).cpu().numpy())      

        projected_w_y = int(laber_index) * torch.ones(projected_w.size(0), dtype = int) 
        # print("projected_w_y.shape:",projected_w_y.shape)   #   projected_w_y.shape: torch.Size([8])
        # print("projected_w_y:",projected_w_y)       #   projected_w_y: tensor([6, 6, 6, 6, 6, 6, 6, 6])
        np.savez(f'{outdir}/{projected_img_index:08d}-{int(laber_index)}-{label_name}-label.npz', w = projected_w_y.unsqueeze(0).cpu().numpy())      

        projected_w = projected_w                                                                                 #   projected_w.shape = torch.size[512]
        projected_y = int(laber_index)                                                                                              
        projected_y = projected_y * torch.ones(G.mapping.num_ws, dtype = int)                                     #   生成一个G.mapping.num_ws维整型张量
        return projected_w,projected_y
        

    def __run_projection_dataset_fromviewfolder(self,opt,exp_result_dir):

        peojected_w_set = []
        projected_y_set = []

        exp_result_dir = os.path.join(exp_result_dir,f'project-{opt.dataset}-trainset')
        os.makedirs(exp_result_dir,exist_ok=True)    
        
        file_dir=os.listdir(opt.viewdataset_path)
        file_dir.sort()
        filenames = [name for name in file_dir if os.path.splitext(name)[-1] == '.png']                                         #   选择指定目录下的.png图片

        for index, filename in enumerate(filenames):                                                                            #   从filenames池中迭代读取一个个路径filename
            # target_fname=None
            if  self._args.project_target_num != None:
                if index < self._args.project_target_num:                                                                           #   选100、1000个样本投影
                    print(f"projecting {self._args.project_target_num} cle samples !  ")
                    img_name = filename[:-4]
                    img_index = img_name[0:8]
                    label_number = img_name[9:10]
                    label = img_name[11:]
                    target_fname = os.path.join(opt.viewdataset_path,filename)                                                      #   f'{opt.viewdataset_path}/{filename}'   

                    projected_w,projected_y = self.__run_projection__(
                        network_pkl = opt.gen_network_pkl,
                        target_fname = target_fname,
                        # target_fname = None,
                        outdir = exp_result_dir,
                        save_video = opt.save_video,
                        seed = opt.seed,
                        num_steps = opt.num_steps,
                        image_name = img_name,
                        projected_img_index = index
                                
                    )

                    #-----------------maggie add-----------
                    peojected_w_set.append(projected_w)
                    projected_y_set.append(projected_y)         
                    #--------------------------------------   
            else:
                print(f"projecting the whole {len(filenames)} cle samples !  ")

                img_name = filename[:-4]
                img_index = img_name[0:8]
                label_number = img_name[9:10]
                label = img_name[11:]
                target_fname = os.path.join(opt.viewdataset_path,filename)                                                      #   f'{opt.viewdataset_path}/{filename}'   

                projected_w,projected_y = self.__run_projection__(
                    network_pkl = opt.gen_network_pkl,
                    target_fname = target_fname,
                    # target_fname = None,
                    outdir = exp_result_dir,
                    save_video = opt.save_video,
                    seed = opt.seed,
                    num_steps = opt.num_steps,
                    image_name = img_name,
                    projected_img_index = index
                            
                )

                #-----------------maggie add-----------
                peojected_w_set.append(projected_w)
                projected_y_set.append(projected_y)         
                #--------------------------------------                   
        print('Finished dataset projecting !')
        # raise error
        #----------------maggie add----------------
        return peojected_w_set, projected_y_set
        #------------------------------------------

    def __run_projection__(self,
        network_pkl: str,               #   使用的styleGAN2-ada网络pkl
        target_fname: str,              #   target_fname是待投影的图片
        outdir: str,                    #   输出路径
        save_video: bool,               #   是否要存储视频
        seed: int,                      #   种子设定
        num_steps: int,                  #   默认是1000，指的是投影优化器参数更新的步长
        image_name: str,                 #   00000000-6-frog
        projected_img_index:int         #   
    ):

        print(f"projecting {projected_img_index:08d} image:")
        
        np.random.seed(seed)                                                                                                    #   种子用于随机化np和torch对象
        torch.manual_seed(seed)

        # Load networks.
        # print('Loading networks from "%s"...' % network_pkl)
        device = torch.device('cuda')
        with dnnlib.util.open_url(network_pkl) as fp:
            G = legacy.load_network_pkl(fp)['G_ema'].requires_grad_(False).to(device)                                           #   load_network_pkl（）会调用到persistency中的class解释器。
            """persistency中的持久性class解释器在network.py中被调用。
            当project任务使用的是用原本stylegan2ada项目训练出来的模型时，
            由于torch_utils目录外不存在utils目录，因此存储下来的模型pkl文件在用persistency解释器解封时，
            会导致找不到/torch_utils/persistency目录，因为mmat项目中对应正确目录是/utils/persistency.
            因此，目前只能用mmat训练出来的模型pkl进行project实验，若要使用stylegan2ada原项目训练出来的模型，就必须将torch_utils目录放到与utils同级的路径下。"""

        # Load target image.
        target_pil = PIL.Image.open(target_fname).convert('RGB')
        # print(target_pil)
        
        w, h = target_pil.size
        # print('target_pil.size=%s' % target_pil)

        s = min(w, h)
        target_pil = target_pil.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
        target_pil = target_pil.resize((G.img_resolution, G.img_resolution), PIL.Image.LANCZOS)
        # print("target_pil:",target_pil)                                                                                         #   target_pil: <PIL.Image.Image image mode=RGB size=32x32 at 0x7FBFB4509C50>   
        # print("target_pil.type:",type(target_pil))                                                                              #   target_pil.type: <class 'PIL.Image.Image'>   
        # print("target_pil.size:",target_pil.size)                                                                               #   target_pil.size: (32, 32)
        
        target_uint8 = np.array(target_pil, dtype=np.uint8)
        # print("target_uint8:",target_uint8)                                                                                     #   target_uint8: [[[ 59  62  63] [ 43  46  45]
        # print("target_uint8.type:",type(target_uint8))                                                                          #   target_uint8.type: <class 'numpy.ndarray'>
        # print("target_uint8.shape:",target_uint8.shape)                                                                         #   target_uint8.shape: (32, 32, 3)
        # print("target_uint8.dtype:",target_uint8.dtype)                                                                         #   target_uint8.dtype: uint8

        # Optimize projection.
        start_time = perf_counter()
        projected_w_steps = self.__project__(
            G,
            target=torch.tensor(target_uint8.transpose([2, 0, 1]), device=device),                                              #   pylint: disable=not-callable
            num_steps=num_steps,
            device=device,
            verbose=True
        )
        # print (f'Elapsed: {(perf_counter()-start_time):.1f} s')

        # #------------maggie---------
        # print("projected_w_steps: ",projected_w_steps)                                                                        #   projected_w_steps.shape:  torch.Size([1000, 8, 512])        8是指复制的八份512向量，因为要送到stylegan2ada网络的mapping模块
        # print("projected_w_steps.shape: ",projected_w_steps.shape)                                                            #   projected_w_steps.shape:  torch.Size([1000, 8, 512])
        # raise error
        # #---------------------------

        # Render debug output: optional video and projected image and W vector.
        os.makedirs(outdir, exist_ok=True)
        if save_video:                                                                                                          #   如果save_video=True
            video = imageio.get_writer(f'{outdir}/proj.mp4', mode='I', fps=10, codec='libx264', bitrate='16M')                  #   将video存为outdir定义的输出路径下的proj.mp4文件
            print (f'Saving optimization progress video "{outdir}/proj.mp4"')
            for projected_w in projected_w_steps:
                synth_image = G.synthesis(projected_w.unsqueeze(0), noise_mode='const')                                         #   projected_w.unsqueeze(0).shape: torch.size[1,8,512]
                synth_image = (synth_image + 1) * (255/2)
                synth_image = synth_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
                video.append_data(np.concatenate([target_uint8, synth_image], axis=1))
            video.close()

        #----------------maggie add 读取target iamge的label---------------------
        #image_name = 00000000-6-frog
        img_index = image_name[0:8]
        label_number = image_name[9:10]                                                                                         #   此时 label_number是str
        label_number = int(label_number)                                                                                        #   str -> int
        label = image_name[11:]
        # print('img_index=%s'% img_index)
        # print('label_number=%s'% label_number)
        # print('label=%s'% label)
        #-----------------------------------------------------------------------
        
        #   存原图
        # Save final projected frame and W vector.

        target_pil.save(f'{outdir}/original-{img_index}-{label_number}-{label}.png')                                            #   指的是原图
        
        #   存投影
        projected_w = projected_w_steps[-1]                                                                                     #   projected_w.shape:  torch.Size([8, 512])                                                                          
        synth_image = G.synthesis(projected_w.unsqueeze(0), noise_mode='const')                                                 #   projected_w.unsqueeze(0).shape:  torch.Size([1, 8, 512])
        
        # print("synth_image:",synth_image)                                                                                       #   
        # print("synth_image.type:",type(synth_image))                                                                            #   
        # print("synth_image.shape:",synth_image.shape)                                                                           #   
        # print("synth_image.dtype:",synth_image.dtype)   
        
        synth_image = (synth_image + 1) * (255/2)
        synth_image = synth_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
        # print("synth_image:",synth_image)                                                                                       #   synth_image: [[[ 49  29   2] [ 48  34  10]
        # print("synth_image.type:",type(synth_image))                                                                            #   synth_image.type: <class 'numpy.ndarray'>
        # print("synth_image.shape:",synth_image.shape)                                                                           #   synth_image.shape: (32, 32, 3)
        # print("synth_image.dtype:",synth_image.dtype)                                                                           #   synth_image.dtype: uint8

        synth_image = PIL.Image.fromarray(synth_image, 'RGB')
        synth_image.save(f'{outdir}/projected-{img_index}-{label_number}-{label}.png')
        # PIL.Image.fromarray(synth_image, 'RGB').save(f'{outdir}/projected-{img_index}-{label_number}-{label}.png')              #   存投影png, synth_image.shape:  torch.Size([1, 3, 32, 32])
        np.savez(f'{outdir}/projected_w-{img_index}-{label_number}-{label}.npz', w=projected_w.unsqueeze(0).cpu().numpy())      #   存投影npz, projected_w.unsqueeze(0).shape:  torch.Size([1, 8, 512])

        # #--------------maggie add-----------
        # print("projected_w_steps.shape: ",projected_w_steps.shape)                                                            #   projected_w_steps.shape:  torch.Size([1000, 8, 512])
        # print("projected_w.shape: ",projected_w.shape)                                                                        #   projected_w.shape:  torch.Size([8, 512])
        # print("projected_w[-1].shape: ",projected_w[-1].shape)                                                                #   projected_w[-1].shape:  torch.Size([512])

        # projected_w = projected_w[-1]                                                                                         #   projected_w.shape = torch.size[8,512]
        projected_w = projected_w                                                                                               #   projected_w.shape = torch.size[512]

        projected_y = label_number                                                                                              #   这时的label还是一个数字 projected_y.shape = int
        projected_y = projected_y * torch.ones(G.mapping.num_ws, dtype = int)                                                   #   生成一个G.mapping.num_ws维整型张量

        # print("projected_w.shape: ",projected_w.shape)                                                                        #  projected_w.shape:  torch.Size([8, 512])
        print("projected_y: ",projected_y)                                                                                      #  projected_y:  tensor([[3., 3., 3., 3., 3., 3., 3., 3.]])
        # print("projected_y.shape: ",projected_y.shape)                                                                        #  projected_y.shape:  torch.Size([1, 8])
        return projected_w,projected_y
        #-----------------------------------

    def __project__(self,
        G,
        target: torch.Tensor, # [C,H,W] and dynamic range [0,255], W & H must match G output resolution
        *,
        num_steps                  = 1000,
        w_avg_samples              = 10000,
        initial_learning_rate      = 0.1,
        initial_noise_factor       = 0.05,
        lr_rampdown_length         = 0.25,
        lr_rampup_length           = 0.05,
        noise_ramp_length          = 0.75,
        regularize_noise_weight    = 1e5,
        verbose                    = False,
        device: torch.device
    ):
        # print("G.img_channels:",G.img_channels)                 #   G.img_channels: 1
        # print("G.img_resolution:",G.img_resolution)             #   G.img_resolution: 32
        assert target.shape == (G.img_channels, G.img_resolution, G.img_resolution)

        def logprint(*args):
            if verbose:
                print(*args)

        G = copy.deepcopy(G).eval().requires_grad_(False).to(device) # type: ignore

        # Compute w stats.
        # logprint(f'Computing W midpoint and stddev using {w_avg_samples} samples...')
        z_samples = np.random.RandomState(123).randn(w_avg_samples, G.z_dim)
        w_samples = G.mapping(torch.from_numpy(z_samples).to(device), None)  # [N, L, C]
        w_samples = w_samples[:, :1, :].cpu().numpy().astype(np.float32)       # [N, 1, C]
        w_avg = np.mean(w_samples, axis=0, keepdims=True)      # [1, 1, C]
        w_std = (np.sum((w_samples - w_avg) ** 2) / w_avg_samples) ** 0.5

        # Setup noise inputs.
        noise_bufs = { name: buf for (name, buf) in G.synthesis.named_buffers() if 'noise_const' in name }

        # Load VGG16 feature detector.
        # if self._args.dataset != 'kmnist' and self._args.dataset != 'mnist':
        url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt'          #该VGG模型不支持单通道样本
        # elif self._args.dataset == 'kmnist' or self._args.dataset == 'mnist':
        #     # print("请准备预训练好的单通道VGG16模型")
        #     url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt' 

        with dnnlib.util.open_url(url) as f:
            vgg16 = torch.jit.load(f).eval().to(device)

        # Features for target image.
        target_images = target.unsqueeze(0).to(device).to(torch.float32)
        if target_images.shape[2] > 256:
            target_images = F.interpolate(target_images, size=(256, 256), mode='area')
        

        # # target_features = vgg16(target_images, resize_images=False, return_lpips=True)
        # if self._args.dataset != 'kmnist' and self._args.dataset != 'mnist':
        #     target_features = vgg16(target_images, resize_images=False, return_lpips=True)
        #     # print("target_features.shape:",target_features.shape)           #   target_features.shape: torch.Size([1, 124928])
        #     # print("target_features.dtype:",target_features.dtype)           #   target_features.dtype: torch.float32
        #     # print("target_features:",target_features)                       #   target_features: tensor([[0.0000, 0.0000, 0.0002,  ..., 0.0000, 0.0000, 0.0009]],

        # elif self._args.dataset == 'kmnist' or self._args.dataset == 'mnist':
        #     # print("target_images.shape:",target_images.shape)       #   target_images [1,1,32,32]
        #     target_images = target_images.expand(-1, 3, -1, -1).clone() 
        #     # print("target_images.shape:",target_images.shape)       #   target_images.shape: torch.Size([1, 3, 32, 32])
        #     target_features = vgg16(target_images, resize_images=False, return_lpips=True)
        #     # target_features = target_images #   因为预训练的vgg16只支持3通道

        if self._args.dataset == 'kmnist' or self._args.dataset == 'mnist':
            target_images = target_images.expand(-1, 3, -1, -1).clone() 
        target_features = vgg16(target_images, resize_images=False, return_lpips=True)

        w_opt = torch.tensor(w_avg, dtype=torch.float32, device=device, requires_grad=True) # pylint: disable=not-callable
        w_out = torch.zeros([num_steps] + list(w_opt.shape[1:]), dtype=torch.float32, device=device)
        optimizer = torch.optim.Adam([w_opt] + list(noise_bufs.values()), betas=(0.9, 0.999), lr=initial_learning_rate)

        # Init noise.
        for buf in noise_bufs.values():
            buf[:] = torch.randn_like(buf)
            buf.requires_grad = True

        for step in range(num_steps):
            # Learning rate schedule.
            t = step / num_steps
            w_noise_scale = w_std * initial_noise_factor * max(0.0, 1.0 - t / noise_ramp_length) ** 2
            lr_ramp = min(1.0, (1.0 - t) / lr_rampdown_length)
            lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
            lr_ramp = lr_ramp * min(1.0, t / lr_rampup_length)
            lr = initial_learning_rate * lr_ramp
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            # Synth images from opt_w.
            w_noise = torch.randn_like(w_opt) * w_noise_scale
            ws = (w_opt + w_noise).repeat([1, G.mapping.num_ws, 1])
            synth_images = G.synthesis(ws, noise_mode='const')

            # Downsample image to 256x256 if it's larger than that. VGG was built for 224x224 images.
            synth_images = (synth_images + 1) * (255/2)
            if synth_images.shape[2] > 256:
                synth_images = F.interpolate(synth_images, size=(256, 256), mode='area')

            # Features for synth images.
            # # synth_features = vgg16(synth_images, resize_images=False, return_lpips=True)
            # if self._args.dataset != 'kmnist' and self._args.dataset != 'mnist':
            #     synth_features = vgg16(synth_images, resize_images=False, return_lpips=True)
            #     # print("synth_features.shape:",synth_features.shape)             #   synth_features.shape: torch.Size([1, 124928])
            #     # print("synth_features.dtype:",synth_features.dtype)             #   synth_features.dtype: torch.float32
            #     # print("synth_features:",synth_features)                         #   synth_features: tensor([[0., 0., 0.,  ..., 0., 0., 0.]], device='cuda:0',
            #     # # raise error           
            # elif self._args.dataset == 'kmnist' or self._args.dataset == 'mnist':
            #     # print("synth_images.shape:",synth_images.shape)       #   synth_images.shape: torch.Size([1, 1, 32, 32])
            #     synth_images = synth_images.expand(-1, 3, -1, -1).clone()
            #     # print("synth_images.shape:",synth_images.shape)       #   synth_images.shape: torch.Size([1, 3, 32, 32])
            #     synth_features = vgg16(synth_images, resize_images=False, return_lpips=True)

            #     # synth_features = synth_images #   因为预训练的vgg16只支持3通道

            #     # print("target_features:",target_features)                           #   target_features: tensor([[[[0., 0., 0.,  ..., 0., 0., 0.], [0., 0., 0.,  ..., 1., 1., 1.],
            #     # print("target_features.shape:",target_features.shape)               #   target_features.shape: torch.Size([1, 1, 32, 32])

            #     # print("synth_features:",synth_features)                             #   synth_features: tensor([[[[-1.1421,  2.3936,  0.0993,  ...,  0.3999, -0.0457,  3.5651],
            #     # print("synth_features.shape:",synth_features.shape)                 #   synth_features.shape: torch.Size([1, 1, 32, 32])
           
            if self._args.dataset == 'kmnist' or self._args.dataset == 'mnist':
                synth_images = synth_images.expand(-1, 3, -1, -1).clone()
            synth_features = vgg16(synth_images, resize_images=False, return_lpips=True)

            #   计算VGG特征损失
            dist = (target_features - synth_features).square().sum()
            # print("dist=",dist)
            # raise error
            # Noise regularization.
            reg_loss = 0.0
            for v in noise_bufs.values():
                noise = v[None,None,:,:] # must be [1,1,H,W] for F.avg_pool2d()
                while True:
                    reg_loss += (noise*torch.roll(noise, shifts=1, dims=3)).mean()**2
                    reg_loss += (noise*torch.roll(noise, shifts=1, dims=2)).mean()**2
                    if noise.shape[2] <= 8:
                        break
                    noise = F.avg_pool2d(noise, kernel_size=2)
            loss = dist + reg_loss * regularize_noise_weight

            # Step
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            # logprint(f'step {step+1:>4d}/{num_steps}: dist {dist:<4.2f} loss {float(loss):<5.2f}')

            # Save projected W for each optimization step.

            #---------maggie----------
            # print("w_opt.shape: ", w_opt.shape)                     #   w_opt.shape:  torch.Size([1, 1, 512])
            #-------------------------    

            w_out[step] = w_opt.detach()[0]             
    
            # #---------maggie----------
            # print("w_out[step].shape: ", w_out[step].shape)         #   w_out[step].shape:  torch.Size([1, 512])
            # print("w_out[step][0][:10]: ", w_out[step][0][:10]) #   projected_w[0][:10]: tensor([ 0.3518,  0.8602,  0.6213,  0.1626,  1.1341,  1.2988,  1.2405,  1.2416,
            # #-------------------------

            # Normalize noise.
            with torch.no_grad():
                for buf in noise_bufs.values():
                    buf -= buf.mean()
                    buf *= buf.square().mean().rsqrt()

        #---------maggie----------
        # print("w_out.shape: ", w_out.shape)                         #   w_out.shape:  torch.Size([1000, 1, 512])
        # print("G.mapping.num_ws: ",G.mapping.num_ws)                #   G.mapping.num_ws = 8                # STL10 G.mapping.num_ws:  10
        # raise error
        #-------------------------
        return w_out.repeat([1, G.mapping.num_ws, 1])
    
#-----------------插值--------------------
    def mixwyset(self):
        return self.interpolated_w_set, self.interpolated_y_set

    def interpolate(self, exp_result_dir, projected_w_set = None, projected_y_set = None):
        self._exp_result_dir = exp_result_dir
        interpolated_w_set, interpolated_y_set = self.__interpolatemain__(self._args, self._exp_result_dir, projected_w_set, projected_y_set)
        self.interpolated_w_set = interpolated_w_set
        self.interpolated_y_set = interpolated_y_set

    def __interpolatemain__(self, opt, exp_result_dir, projected_w_set, projected_y_set):
        # print("running interpolate main()..............")

        if projected_w_set is not None :
            self.projected_w_set = projected_w_set
            self.projected_y_set = projected_y_set
            # print("Interpolate projectors from projectors numpy ndarray !")
            interpolated_w_set, interpolated_y_set = self.__ramwymixup__()          #   rmt从这里进入
        else:
            print("Interpolate projectors from projectors npz files !")
            if opt.projected_dataset != None:                                                                                   #   数据集混合 projected_dataset放的是npz的路径
                interpolated_w_set, interpolated_y_set = self.__DatasetMixup__(opt,exp_result_dir)
            elif opt.projected_dataset == None:  
                # print('混合单个样本')      
                # raise error                                                                         #   样本混合
                print("opt.projected_w3:",opt.projected_w3)
                if opt.projected_w3 == None:
                    interpolated_w_set, interpolated_y_set = self.__TwoMixup__(opt,exp_result_dir)
                elif opt.projected_w3 != None:       
                    interpolated_w_set, interpolated_y_set = self.__ThreeMixup__(opt,exp_result_dir)


        #------maggie add----------
        return interpolated_w_set, interpolated_y_set
        #--------------------------

    def __ramwymixup__(self):
        #--------------读取标签-------------------------------------------
        opt = self._args
        exp_result_dir = self._exp_result_dir

        device = torch.device('cuda')        

        # print("projected_w_set_x.shape: ",self.projected_w_set.shape)                                                              
        # print("projected_w_set_y.shape: ",self.projected_y_set.shape)                                                              

        if self._args.defense_mode == 'rmt':

            projected_w_set_x = self.projected_w_set.to(device)                                                       #   numpy -> tensor
            projected_w_set_y = self.projected_y_set.to(device)   

            # print("projected_w_set_x.shape: ",self.projected_w_set.shape)                                                              
            # print("projected_w_set_y.shape: ",self.projected_y_set.shape)                                                              
            # raise error
            projected_w_set_y = torch.nn.functional.one_hot(projected_w_set_y, opt.n_classes).float().to(device)
            interpolated_w_set, interpolated_y_set = self.__getmixedbatchwy__(opt, projected_w_set_x, projected_w_set_y)

        else:
            projected_w_set_x = torch.tensor(self.projected_w_set).to(device)                                                       #   numpy -> tensor
            projected_w_set_y = torch.tensor(self.projected_y_set).to(device)                                                       #   numpy数组里是int型label编号，要转成one hot

            # print("projected_w_set_x.shape: ",self.projected_w_set.shape)                                                              
            # print("projected_w_set_y.shape: ",self.projected_y_set.shape)                                                              

            projected_w_set_y = torch.nn.functional.one_hot(projected_w_set_y, opt.n_classes).float().to(device)                    #   y成了gpu tensor
            interpolated_w_set, interpolated_y_set = self.__getmixededwy__(opt, projected_w_set_x,projected_w_set_y,exp_result_dir)

        return interpolated_w_set, interpolated_y_set

    def __getmixedbatchwy__(self, opt, projected_w_set_x, projected_w_set_y):

        # print("projected_w_set_x: ",projected_w_set_x)
        # print("projected_w_set_y: ",projected_w_set_y)  
        # print("projected_w_set_x.shape: ",projected_w_set_x.shape)                                                                                          
        # print("projected_w_set_y.shape: ",projected_w_set_y.shape) 
        repeat_num = projected_w_set_x.size(1)

        if opt.mix_w_num == 2:
            batch_size = projected_w_set_x.size()[0]

            use_cuda = torch.cuda.is_available()
            if use_cuda:
                shuffle_index = torch.randperm(batch_size).cuda()   #表示从batch中随机选一个待混合的样本x[index, :]
            else:
                shuffle_index = torch.randperm(batch_size)

            # print("shuffle_index:",shuffle_index)
            # print("shuffle_index.size():",shuffle_index.size())

            # -----------一个batch整体混合
            shuffled_projected_w_set_x = projected_w_set_x[shuffle_index,:]
            shuffled_projected_w_set_y = projected_w_set_y[shuffle_index,:]

            # print("shuffled_projected_w_set_x: ",shuffled_projected_w_set_x)                                                                                          
            # print("shuffled_projected_w_set_y: ",shuffled_projected_w_set_y)             
            # print("shuffled_projected_w_set_x.shape: ",shuffled_projected_w_set_x.shape)                                                                                          
            # print("shuffled_projected_w_set_y.shape: ",shuffled_projected_w_set_y.shape) 

            
            projected_w_set_x = projected_w_set_x[:, 0, :].squeeze(1)                           #   [4,1,512] --> [4,512]
            shuffled_projected_w_set_x = shuffled_projected_w_set_x[:, 0, :].squeeze(1)         #   [4,1,512] --> [4,512]

            projected_w_set_y = projected_w_set_y[:, 0, :].squeeze(1)                           #   [4,1,10] --> [4,10]
            shuffled_projected_w_set_y = shuffled_projected_w_set_y[:, 0, :].squeeze(1)

            # print("projected_w_set_x.shape: ",projected_w_set_x.shape)                          #   projected_w_set_x.shape:  torch.Size([4, 512])
            # print("projected_w_set_x: ",projected_w_set_x)                                                                                                    

            # print("shuffled_projected_w_set_x.shape: ",shuffled_projected_w_set_x.shape)        #   shuffled_projected_w_set_x.shape:  torch.Size([4, 512])
            # print("shuffled_projected_w_set_x: ",shuffled_projected_w_set_x)       

            # print("projected_w_set_y.shape: ",projected_w_set_y.shape)                          #   projected_w_set_y.shape:  torch.Size([4, 10])
            # print("projected_w_set_y: ",projected_w_set_y) 

            # print("shuffled_projected_w_set_y.shape: ",shuffled_projected_w_set_y.shape)        #   shuffled_projected_w_set_y.shape:  torch.Size([4, 10])
            # print("shuffled_projected_w_set_y: ",shuffled_projected_w_set_y) 

            #------------执行混合算法------------------
            #   关键问题就是这里的alpha取值for beta(alpha,alpha)
            if opt.mix_mode == 'basemixup':
                interpolated_w_set, interpolated_y_set = self.__BaseMixup2__(projected_w_set_x, shuffled_projected_w_set_x, opt.sample_mode, projected_w_set_y, shuffled_projected_w_set_y)
            elif opt.mix_mode == 'maskmixup':
                interpolated_w_set, interpolated_y_set = self.__MaskMixup2__(projected_w_set_x, shuffled_projected_w_set_x, opt.sample_mode, projected_w_set_y, shuffled_projected_w_set_y)
            else:
                raise Exception('please input valid mix_mode')

            # print("interpolated_w_set.shape: ",interpolated_w_set.shape)                                                                                          
            # print("interpolated_y_set.shape: ",interpolated_y_set.shape) 
            # print("interpolated_w_set: ",interpolated_w_set)                                                                                          
            # print("interpolated_y_set: ",interpolated_y_set) 
            # """
            # interpolated_w_set.shape:  torch.Size([4, 512]) --> [4,8,512]
            # interpolated_y_set.shape:  torch.Size([4, 10])  --> [4,8,10]
            # """
            
            interpolated_w_set = interpolated_w_set.unsqueeze(1)
            interpolated_y_set = interpolated_y_set.unsqueeze(1)

            # print("interpolated_w_set.shape: ",interpolated_w_set.shape)                                                                                          
            # print("interpolated_y_set.shape: ",interpolated_y_set.shape) 
            # print("interpolated_w_set: ",interpolated_w_set)                                                                                          
            # print("interpolated_y_set: ",interpolated_y_set) 


            interpolated_w_set = interpolated_w_set.expand(interpolated_w_set.size()[0],repeat_num,interpolated_w_set.size()[2])
            interpolated_y_set = interpolated_y_set.expand(interpolated_y_set.size()[0],repeat_num,interpolated_y_set.size()[2])

            # print("interpolated_w_set.shape: ",interpolated_w_set.shape)                                                                                          
            # print("interpolated_y_set.shape: ",interpolated_y_set.shape) 
            # print("interpolated_w_set: ",interpolated_w_set)                                                                                          
            # print("interpolated_y_set: ",interpolated_y_set) 

                        
        return interpolated_w_set, interpolated_y_set


        #     #   一个一个混合
        #     #------maggie add----------
        #     # 注意空间使用
        #     interpolated_w_set = []
        #     interpolated_y_set = []
        #     #--------------------------
        #     for i in range(len(projected_w_set_x)): 
        #         w1 = projected_w_set_x[i][-1].unsqueeze(0)
        #         y1 = projected_w_set_y[i][-1].unsqueeze(0)

        #         w2 = projected_w_set_x[shuffle_index[i]][-1].unsqueeze(0)
        #         y2 = projected_w_set_y[shuffle_index[i]][-1].unsqueeze(0) 

        #         # print(f"({i:04d}-w1,{shuffle_index[i]:04d}-w2)")
 

        #         _, w1_label_index = torch.max(y1, 1)    
        #         _, w2_label_index = torch.max(y2, 1)  

        #         #   存储图片
        #         # w1_label_name = f"{classification[int(w1_label_index)]}"
        #         # w2_label_name = f"{classification[int(w2_label_index)]}"
        #         # print("w1_label_index.type:",type(w1_label_index)) 
        #         # print("w1_label_index:",w1_label_index)  
        #         # print("w2_label_index.type:",type(w2_label_index))  
        #         # print("w2_label_index:",w2_label_index)  

        #         # if w1_label_name == w2_label_name:
        #         #     print("mixup same class")
        #         # else:
        #         #     print("mixup different classes")

        #         # print("w1_label_name:",w1_label_name)
        #         # print("w2_label_name:",w2_label_name)
                

        #         #------------执行混合算法------------------
        #         if opt.mix_mode == 'basemixup':
        #             w_mixed, y_mixed = self.__BaseMixup2__(w1,w2,opt.sample_mode,y1,y2)
        #         elif opt.mix_mode == 'maskmixup':
        #             w_mixed, y_mixed = self.__MaskMixup2__(w1,w2,opt.sample_mode,y1,y2)
        #         else:
        #             raise Exception('please input valid mix_mode')
            
        #         # print("w_mixed.shape: ",w_mixed.shape)                                                                          #   w_mixed.shape:  torch.Size([1, 512]) 
        #         # print("y_mixed.shape: ",y_mixed.shape)                                                                          #   y_mixed.shape:  torch.Size([1, 10])
        #         # print("projected_w_set_x.size(1):",projected_w_set_x.size(1))       #   stl10, projected_w_set_x.size(1): 10   cifar10: projected_w_set_x.size(1): 8
        #         # print("projected_w_set_y.size(1):",projected_w_set_y.size(1))       #   projected_w_set_y.size(1): 10 projected_w_set_y.size(1): 8
        #         repeat_num = projected_w_set_x.size(1)
        #         # w_mixed = w_mixed.repeat([8,1])       
        #         # y_mixed = y_mixed.repeat([8,1])
        #         w_mixed = w_mixed.repeat([repeat_num,1])       
        #         y_mixed = y_mixed.repeat([repeat_num,1])                    
        #         # print("w_mixed: ",w_mixed)                             
        #         # print("w_mixed.shape: ",w_mixed.shape)                                                                      #   w_mixed.shape:  torch.Size([8,512])
        #         # print("y_mixed: ",y_mixed)                             
        #         # print("y_mixed.shape: ",y_mixed.shape)                                                                      #   y_mixed.shape:  torch.Size([8,10])

        #         interpolated_w_set.append(w_mixed)
        #         interpolated_y_set.append(y_mixed)


        # return interpolated_w_set, interpolated_y_set


    def __getmixededwy__(self,opt, projected_w_set_x,projected_w_set_y,exp_result_dir):

        exp_result_dir = os.path.join(exp_result_dir,f'interpolate-{opt.dataset}-trainset')
        os.makedirs(exp_result_dir,exist_ok=True)    

        classification = self.__labelnames__() 
        print("classification label name:",classification)
        # raise error

        #------maggie add----------
        # 注意空间使用
        interpolated_w_set = []
        interpolated_y_set = []
        #--------------------------
        print("projected_w_set_x.shape:",projected_w_set_x.shape)           #   projected_w_set_x.shape: torch.Size([38, 10, 512])
        print("projected_w_set_y.shape:",projected_w_set_y.shape)
        
        mix_num = 0
        if opt.mix_w_num == 2:      
            print("--------------------Dual mixup----------------------")
            for i in range(len(projected_w_set_x)):                                                                                 #   projected_w_set列表共有72个张量
                for j in range(len(projected_w_set_x)):
                    if j != i:
                        if mix_num < opt.mix_img_num:       #   mix_img_num指定混合多少样本
                            # print(f"projected_w_set_x[{i}]:{projected_w_set_x[i]}")
                            w1 = projected_w_set_x[i][-1].unsqueeze(0)
                            y1 = projected_w_set_y[i][-1].unsqueeze(0)
                            # print("w1.shape: ",w1.shape)                                                                                    #   w1.shape:  torch.Size([1, 512]
                            # print("y1.shape: ",y1.shape)                            
                            
                            # print(f"projected_w_set_x[{j}]:{projected_w_set_x[j]}")
                            w2 = projected_w_set_x[j][-1].unsqueeze(0)
                            y2 = projected_w_set_y[j][-1].unsqueeze(0) 
                            # print("w2.shape: ",w2.shape)                                                                                    #   w2.shape:  torch.Size([1, 512])
                            # print("y2.shape: ",y2.shape)      

                            _, w1_label_index = torch.max(y1, 1)    
                            _, w2_label_index = torch.max(y2, 1)  

                            #   存储图片
                            w1_label_name = f"{classification[int(w1_label_index)]}"
                            w2_label_name = f"{classification[int(w2_label_index)]}"
                            # print("w1_label_index.type:",type(w1_label_index)) 
                            # print("w1_label_index:",w1_label_index)  
                            # print("w2_label_index.type:",type(w2_label_index))  
                            # print("w2_label_index:",w2_label_index)  

                            if w1_label_name == w2_label_name:
                                print("mixup same class")
                            else:
                                print("mixup different classes")

                            print("w1_label_name:",w1_label_name)
                            print("w2_label_name:",w2_label_name)
                            

                            #------------执行混合算法------------------
                            if opt.mix_mode == 'basemixup':
                                w_mixed, y_mixed = self.__BaseMixup2__(w1,w2,opt.sample_mode,y1,y2)
                            elif opt.mix_mode == 'maskmixup':
                                w_mixed, y_mixed = self.__MaskMixup2__(w1,w2,opt.sample_mode,y1,y2)
                            elif opt.mix_mode == 'adversarialmixup':
                                w_mixed = self.__AdversarialMixup2__(w1,w2,opt.sample_mode)
                            else:
                                raise Exception('please input valid mix_mode')
                        
                            # print("w_mixed.shape: ",w_mixed.shape)                                                                          #   w_mixed.shape:  torch.Size([1, 512]) 
                            # print("y_mixed.shape: ",y_mixed.shape)                                                                          #   y_mixed.shape:  torch.Size([1, 10])
                            # print("projected_w_set_x.size(1):",projected_w_set_x.size(1))       #   stl10, projected_w_set_x.size(1): 10   cifar10: projected_w_set_x.size(1): 8
                            # print("projected_w_set_y.size(1):",projected_w_set_y.size(1))       #   projected_w_set_y.size(1): 10 projected_w_set_y.size(1): 8
                            repeat_num = projected_w_set_x.size(1)
                            # w_mixed = w_mixed.repeat([8,1])       
                            # y_mixed = y_mixed.repeat([8,1])
                            w_mixed = w_mixed.repeat([repeat_num,1])       
                            y_mixed = y_mixed.repeat([repeat_num,1])                    
                            # print("w_mixed: ",w_mixed)                             
                            # print("w_mixed.shape: ",w_mixed.shape)                                                                      #   w_mixed.shape:  torch.Size([8,512])
                            # print("y_mixed: ",y_mixed)                             
                            # print("y_mixed.shape: ",y_mixed.shape)                                                                      #   y_mixed.shape:  torch.Size([8,10])



                            #------------写成npz文件-------------------
                            # 当前样本编号：
                            np.savez(f'{exp_result_dir}/{i:08d}-{int(w1_label_index)}-{w1_label_name}+{j:08d}-{int(w2_label_index)}-{w2_label_name}-mixed_projected_w.npz', w=w_mixed.unsqueeze(0).cpu().numpy())                   #   将latent code w 存为 outdir定义的输出路径下的projected_w.npz文件
                            np.savez(f'{exp_result_dir}/{i:08d}-{int(w1_label_index)}-{w1_label_name}+{j:08d}-{int(w2_label_index)}-{w2_label_name}-mixed_label.npz', w = y_mixed.unsqueeze(0).cpu().numpy())               #   将latent code w 存为 outdir定义的输出路径下的projected_w.npz文件 存成了(1,8,10)

                            interpolated_w_set.append(w_mixed)
                            interpolated_y_set.append(y_mixed)

                            mix_num = mix_num + 1

        elif opt.mix_w_num == 3:
            print("-------------------Ternary mixup----------------------")
            for i in range(len(projected_w_set_x)):
                for j in range(len(projected_w_set_x)):
                    for k in range(len(projected_w_set_x)):
                        if k != j and j != i :
                            if mix_num < opt.mix_img_num:
                                # print(f"projected_w_set_x[{i}]:{projected_w_set_x[i]}")
                                w1 = projected_w_set_x[i][-1].unsqueeze(0)
                                y1 = projected_w_set_y[i][-1].unsqueeze(0)
                                # print("w1.shape: ",w1.shape)                                                               #   w1.shape:  torch.Size([1, 512]
                                # print("y1.shape: ",y1.shape)                                                               #   y1.shape:  torch.Size([1, 10])

                                # print(f"projected_w_set_x[{j}]:{projected_w_set_x[j]}")
                                w2 = projected_w_set_x[j][-1].unsqueeze(0)
                                y2 = projected_w_set_y[j][-1].unsqueeze(0) 
                                # print("w2.shape: ",w2.shape)                                                               #   w2.shape:  torch.Size([1, 512])
                                # print("y2.shape: ",y2.shape)                                                               #   y2.shape:  torch.Size([1, 10])

                                # print(f"projected_w_set_x[{k}]:{projected_w_set_x[k]}")
                                w3 = projected_w_set_x[k][-1].unsqueeze(0)
                                y3 = projected_w_set_y[k][-1].unsqueeze(0) 
                                # print("w3.shape: ",w3.shape)                                                               #   w3.shape:  torch.Size([1, 512])
                                # print("y3.shape: ",y3.shape)   
                            

                                #-----maggie------------
                                _, w1_label_index = torch.max(y1, 1)    
                                _, w2_label_index = torch.max(y2, 1)  
                                _, w3_label_index = torch.max(y3, 1)    
                                #   存储图片
                                w1_label_name = f"{classification[int(w1_label_index)]}"
                                w2_label_name = f"{classification[int(w2_label_index)]}"
                                w3_label_name = f"{classification[int(w3_label_index)]}"

                                # print("w1_label_index.type:",type(w1_label_index)) 
                                # print("w1_label_index:",w1_label_index)  
                                # print("w2_label_index.type:",type(w2_label_index))  
                                # print("w2_label_index:",w2_label_index)  
                                #-----------------------

                                if w1_label_name == w2_label_name and w2_label_name == w3_label_name:
                                    print("mixup same class")

                                else:
                                    print("mixup different classes")

                                print("w1_label_name:",w1_label_name)
                                print("w2_label_name:",w2_label_name)
                                print("w3_label_name:",w2_label_name)
                                
                                #------------执行混合算法------------------
                                # print("opt.mix_mode:",opt.mix_mode)
                                if opt.mix_mode == 'basemixup':
                                    w_mixed, y_mixed = self.__BaseMixup3__(w1,w2,w3,opt.sample_mode,y1,y2,y3)
                                elif opt.mix_mode == 'maskmixup':
                                    w_mixed, y_mixed = self.__MaskMixup3__(w1,w2,w3,opt.sample_mode,y1,y2,y3)
                                else:
                                    raise Exception('please input valid mix_mode')
                            
                                # print("w_mixed.shape: ",w_mixed.shape)                                                       #   w_mixed.shape:  torch.Size([1, 512]) 
                                # print("y_mixed.shape: ",y_mixed.shape)                                                       #   y_mixed.shape:  torch.Size([1, 10])

                                repeat_num = projected_w_set_x.size(1)
                                # raise error
                                # w_mixed = w_mixed.repeat([8,1])       
                                # y_mixed = y_mixed.repeat([8,1])
                                w_mixed = w_mixed.repeat([repeat_num,1])       
                                y_mixed = y_mixed.repeat([repeat_num,1]) 
                                # print("w_mixed: ",w_mixed)                             
                                # print("w_mixed.shape: ",w_mixed.shape)                                                        #   w_mixed.shape:  torch.Size([8,512])
                                # print("y_mixed: ",y_mixed)                             
                                # print("y_mixed.shape: ",y_mixed.shape)                                                        #   y_mixed.shape:  torch.Size([8,10])



                                #------------写成npz文件-------------------
                                np.savez(f'{exp_result_dir}/{i:08d}-{int(w1_label_index)}-{w1_label_name}+{j:08d}-{int(w2_label_index)}-{w2_label_name}+{k:08d}-{int(w3_label_index)}-{w3_label_name}-mixed_projected_w.npz', w=w_mixed.unsqueeze(0).cpu().numpy())      #   将latent code w村委outdir定义的输出路径下的projected_w.npz文件
                                np.savez(f'{exp_result_dir}/{i:08d}-{int(w1_label_index)}-{w1_label_name}+{j:08d}-{int(w2_label_index)}-{w2_label_name}+{k:08d}-{int(w3_label_index)}-{w3_label_name}-mixed_label.npz', w = y_mixed.unsqueeze(0).cpu().numpy())      #   将latent code w村委outdir定义的输出路径下的projected_w.npz文件 存成了(1,8,10)

                                interpolated_w_set.append(w_mixed)
                                interpolated_y_set.append(y_mixed)           

                                mix_num = mix_num + 1                       

        return interpolated_w_set, interpolated_y_set

    def __BaseMixup2__(self,w1,w2,sample_mode,y1,y2):                                                                           #   alpha*w1+(1-alpha)*w2   
        # print("flag: BaseMixup2...")
        
        # print("w1:",w1)                                                                                                       #   w1.type: <class 'torch.Tensor'>
        # print("w1.type:",type(w1))
        # print("w1.shape",w1.shape)                                                                                            #   w1.shape torch.Size([1, 512])
        # print("w1.size:",w1.size())                                                                                           #  w1.size: torch.Size([1,512])                     
       
        # print("w1.size(0): ",w1.size(0))                                                                                      #   w1.size(0):  1
        # print("w1.size(1): ",w1.size(1))                                                                                      #   w1.size(1):  512

        # print("y1:",y1)
        # print("y1.type:",type(y1))                                                                                            #   y1.type: <class 'torch.Tensor'>
        # print("y1.shape",y1.shape)                                                                                            #   y1.shape torch.Size([1, 10])     


        is_2d = True if len(w1.size()) == 2 else False                                                                          #   即w1的shape元组是二维时,is_2d = true
        # print("is_2d=%s" % is_2d)

        #   beta分布采样
        if sample_mode == 'uniformsampler':
            # print('sample_mode = uniformsampler, set the same alpha value for each dimension of the 512 dimensions values of projected w !')
            alpha = utils.sampler.UniformSampler(w1.size(0), w1.size(1), is_2d, p=None)                                         #  UniformSampler里的w1.size(0)应该填1 ，因为此处是batchsize大小
                                # UniformSampler(bs, f, is_2d, p=None)
        elif sample_mode == 'uniformsampler2':
            # print('sample_mode = uniformsampler2,set different alpha values for each dimension of the 512 dimensions values of projected w !')
            alpha = utils.sampler.UniformSampler2(w1.size(0), w1.size(1), is_2d, p=None)
        #   修改后的混合模式
        elif sample_mode == 'betasampler':
            alpha = utils.sampler.BetaSampler(w1.size(0), w1.size(1), is_2d, p=None, beta_alpha = self._args.beta_alpha)



        # print('alpha=%s' % alpha)                                                                                             #   alpha=tensor([[0.5488]], device='cuda:0')
        # print('alpha.shape:', alpha.shape)  (batchsize,1)                                                                                    #   torch.size[8]

        w_mixed = alpha*w1 + (1.-alpha)*w2
        y_mixed = alpha*y1 + (1.-alpha)*y2

        # print("w_mixed: ",w_mixed)
        # print("w_mixed.shape:",w_mixed.shape)                                                                                 #   w_mixed.shape: torch.Size([1, 512])
        # print("y_mixed: ",y_mixed)
        # print("y_mixed.shape:",y_mixed.shape)                                                                                 #   y_mixed.shape: torch.Size([1, 10])

        return w_mixed,y_mixed

    def __MaskMixup2__(self,w1,w2,sample_mode,y1,y2):                                                                           #   m1*w1+(1-m1)*w2
        print("flag: MaskMixup2")
        # print("w1:",w1)                                                                                                       #   w1.type: <class 'torch.Tensor'>
        # print("w1.type:",type(w1))
        # print("w1.shape",w1.shape)                                                                                            #   w1.shape torch.Size([1, 512])
        # print("w1.size:",w1.size())                                                                                           #  w1.size: torch.Size([1,512])                     
       
        # print("w1.size(0): ",w1.size(0))                                                                                      #   w1.size(0):  1
        # print("w1.size(1): ",w1.size(1))                                                                                      #   w1.size(1):  512

        # print("y1:",y1)
        # print("y1.type:",type(y1))                                                                                            #   y1.type: <class 'torch.Tensor'>
        # print("y1.shape",y1.shape)        #   y1.shape torch.Size([1, 10])     


        is_2d = True if len(w1.size()) == 2 else False
        if sample_mode == 'bernoullisampler':
            print('sample_mode = bernoullisampler, samll variance !')
            m = utils.sampler.BernoulliSampler(w1.size(0), w1.size(1), is_2d, p=None)
        elif sample_mode == 'bernoullisampler2':
            print('sample_mode = bernoullisampler2, big variance !')
            m = utils.sampler.BernoulliSampler2(w1.size(0), w1.size(1), is_2d, p=None)

        # print('m.shape:', m.shape)          #   m.shape: torch.Size([1, 512])
        # print("m.size(0):",m.size(0))       #   m.size(0): 1
        # print("m.size(1):",m.size(1))       #   m.size(1): 512

        # print("torch.nonzero(m[0]).shape:",torch.nonzero(m[0]).shape)             # torch.nonzero(m[0]).shape: torch.Size([250, 1]) 找出tensor中非零的元素的索引
        # print("torch.nonzero(m[0]).size(0):",torch.nonzero(m[0]).size(0))         # torch.nonzero(m[0]).size(0): 250
        
        lam = (torch.nonzero(m[0]).size(0)) / m.size(1)
        # print("lam:",lam)   #   lam: 0.513671875

        w_mixed = m*w1 + (1.-m)*w2
        y_mixed = lam*y1 + (1.-lam)*y2
        print("w_mixed.shape:",w_mixed.shape)
        print("y_mixed.shape:",y_mixed.shape)

        # raise error
        return w_mixed,y_mixed

    def __BaseMixup3__(self,w1,w2,w3,sample_mode,y1,y2,y3):
        # print("flag: BaseMixup3")
        # print(w1.size())                                                                          #   返回的是当前张量w1的形状 , 输出torch.Size([1, 14, 512])
        # print('bs=w1.size(0)=%s' % w1.size(0))             
        # print('f=w1.size(1)=%s' % w1.size(1))             
        # print("len w1.size()=%s" % len(w1.size()))

        # is_2d = True if len(w1.size()) == 2 else False                                                                          #   即w1的shape元组是二维时,is_2d = true
        # print("is_2d=%s" % is_2d)
        # print('sample_mode = dirichletsampler')
        # alpha = utils.sampler.DirichletSampler(w1.size(0), w1.size(1), is_2d)
        # print('alpha=%s' % alpha)

        # w_mixed = alpha[:, 0:1]*w1 + alpha[:, 1:2]*w2 + alpha[:, 2:3]*w3
        # # return w_mixed


        print("flag: BaseMixup3...")
        
        # print("w1:",w1)                                                                                                       #   w1.type: <class 'torch.Tensor'>
        # print("w1.type:",type(w1))
        # print("w1.shape",w1.shape)                                                                                            #  w1.shape torch.Size([1, 512])
        # print("w1.size:",w1.size())                                                                                           #  w1.size: torch.Size([1,512])                     
       
        # print("w1.size(0): ",w1.size(0))                                                                                      #   w1.size(0):  1
        # print("w1.size(1): ",w1.size(1))                                                                                      #   w1.size(1):  512

        # print("y1:",y1)
        # print("y1.type:",type(y1))                                                                                            #   y1.type: <class 'torch.Tensor'>
        # print("y1.shape",y1.shape)                                                                                            #   y1.shape torch.Size([1, 10])  


        is_2d = True if len(w1.size()) == 2 else False                                                                          #   即w1的shape元组是二维时,is_2d = true
        # print("is_2d=%s" % is_2d)

        if sample_mode == 'uniformsampler' or sample_mode == 'uniformsampler2' or sample_mode =='dirichletsampler':
            # print('sample_mode = uniformsampler, set the same alpha value for each dimension of the 512 dimensions values of projected w !')
            # alpha = utils.sampler.UniformSampler(w1.size(0), w1.size(1), is_2d, p=None)                                         #  UniformSampler里的w1.size(0)应该填1 ，因为此处是batchsize大小
            #                     # UniformSampler(bs, f, is_2d, p=None)

            alpha = utils.sampler.DirichletSampler(w1.size(0), w1.size(1), is_2d, dirichlet_gama = self._args.dirichlet_gama)


        # elif sample_mode == 'uniformsampler2':
        #     # print('sample_mode = uniformsampler2,set different alpha values for each dimension of the 512 dimensions values of projected w !')
        #     # alpha = utils.sampler.UniformSampler2(w1.size(0), w1.size(1), is_2d, p=None)
        #     alpha = utils.sampler.DirichletSampler(w1.size(0), w1.size(1), is_2d)

        # print('alpha=',alpha)                                                                                           #   alpha= tensor([[0.3399, 0.2961, 0.3639]], device='cuda:0')
        # print('alpha[:, 0:1]=',alpha[:, 0:1])                                                                           #   alpha[:, 0:1]= tensor([[0.3399]], device='cuda:0')
        # print('alpha[:, 1:2]=',alpha[:, 1:2])                                                                           #   alpha[:, 1:2]= tensor([[0.2961]], device='cuda:0')
        # print('alpha[:, 2:3]=',alpha[:, 2:3])                                                                           #   alpha[:, 2:3]= tensor([[0.3639]], device='cuda:0')
        
        
        # print('alpha.shape:', alpha.shape)                                                                                 # alpha.shape: torch.Size([1, 3])
        # # raise error

        # w_mixed = alpha*w1 + (1.-alpha)*w2
        # y_mixed = alpha*y1 + (1.-alpha)*y2
        
        w_mixed = alpha[:, 0:1]*w1 + alpha[:, 1:2]*w2 + alpha[:, 2:3]*w3
        y_mixed = alpha[:, 0:1]*y1 + alpha[:, 1:2]*y2 + alpha[:, 2:3]*y3

        # print("w_mixed: ",w_mixed)
        # print("w_mixed.shape:",w_mixed.shape)                                                                                 #   w_mixed.shape: torch.Size([1, 512])
        # print("y_mixed: ",y_mixed)
        # print("y_mixed.shape:",y_mixed.shape)                                                                                 #   y_mixed.shape: torch.Size([1, 10])

        return w_mixed,y_mixed

    def __MaskMixup3__(self,w1,w2,w3,sample_mode,y1,y2,y3):
        # print("flag: MaskMixup3")
        # print(w1.size())                                                                                                        #   返回的是当前张量w1的形状 , 输出torch.Size([1, 14, 512])
        # print('bs=w1.size(0)=%s' % w1.size(0))             
        # print('f=w1.size(1)=%s' % w1.size(1))             
        # print("len w1.size()=%s" % len(w1.size()))

        # is_2d = True if len(w1.size()) == 2 else False
        # print('sample_mode = bernoullisampler3')
        # alpha = utils.sampler.BernoulliSampler3(w1.size(0), w1.size(1), is_2d)
        # print('alpha=%s' % alpha)

        # w_mixed = alpha[:, 0]*w1 + alpha[:, 1]*w2 + alpha[:, 2]*w3
        # return w_mixed

        print("flag: MaskMixup3")
        # print("w1:",w1)                                                                                                       #   w1.type: <class 'torch.Tensor'>
        # print("w1.type:",type(w1))
        # print("w1.shape",w1.shape)                                                                                            #   w1.shape torch.Size([1, 512])
        # print("w1.size:",w1.size())                                                                                           #  w1.size: torch.Size([1,512])                     
       
        # print("w1.size(0): ",w1.size(0))                                                                                      #   w1.size(0):  1
        # print("w1.size(1): ",w1.size(1))                                                                                      #   w1.size(1):  512

        # print("y1:",y1)
        # print("y1.type:",type(y1))                                                                                            #   y1.type: <class 'torch.Tensor'>
        # print("y1.shape",y1.shape)        #   y1.shape torch.Size([1, 10])     


        is_2d = True if len(w1.size()) == 2 else False
        if sample_mode == 'bernoullisampler' or sample_mode == 'bernoullisampler2':
            m = utils.sampler.BernoulliSampler3(w1.size(0), w1.size(1), is_2d)

        # print('m.shape:', m.shape)          #   m.shape: torch.Size([1, 3, 512])
        # print("m.size(0):",m.size(0))       #   m.size(0): 1
        # print("m.size(1):",m.size(1))       #   m.size(1): 3
        # print("m.size(2):",m.size(2))       #   m.size(2): 512
        # print("m[0][0]:",m[0][0])
        # print("m[0][1]:",m[0][1])
        # print("m[0][2]:",m[0][2])
        # m_syn = m[0][0]+m[0][1]+m[0][2]
        # print( torch.nonzero(m_syn).shape )     #torch.Size([512, 1]) 说明三个mask分量合为1矩阵
        # raise error 

        m1 = m[0][0].unsqueeze(0)
        m2 = m[0][1].unsqueeze(0)
        m3 = m[0][2].unsqueeze(0)
        # print("m1.shape:",m1.shape) #   m1.shape: torch.Size([1, 512])
        # print("m2.shape:",m2.shape)
        # print("m3.shape:",m3.shape)
        # print("m1:",m1)
        # print("m2:",m2)
        # print("m3:",m3)


        lam_1 = (torch.nonzero(m[0][0]).size(0)) / m.size(2)
        lam_2 = (torch.nonzero(m[0][1]).size(0)) / m.size(2)
        lam_3 = (torch.nonzero(m[0][1]).size(0)) / m.size(2)
        # print("torch.nonzero(m[0][0]).size(0):",torch.nonzero(m[0][0]).size(0))         #   torch.nonzero(m[0][0]).size(0): 159
        # print("torch.nonzero(m[0][1]).size(0):",torch.nonzero(m[0][1]).size(0))         #   torch.nonzero(m[0][1]).size(0): 175
        # print("torch.nonzero(m[0][2]).size(0):",torch.nonzero(m[0][2]).size(0))         #   torch.nonzero(m[0][2]).size(0): 178

        # print("lam_1:",lam_1)       #   lam_1: 0.310546875
        # print("lam_2:",lam_2)       #   lam_2: 0.341796875
        # print("lam_3:",lam_3)       #   lam_3: 0.34765625
        # raise error 

        w_mixed = m1*w1 + m2*w2 +m3*w3
        y_mixed = lam_1*y1 + lam_2*y2 +lam_3*y3

        # print("w_mixed.shape:",w_mixed.shape)       #   w_mixed.shape: torch.Size([1, 512])
        # print("y_mixed.shape:",y_mixed.shape)       #   y_mixed.shape: torch.Size([1, 10])
        # raise error
        return w_mixed,y_mixed

    def __AdversarialMixup2__(self,ws1,ws2,sample_mode):
        print('AdversarialMixup2')

    def __TwoMixup__(self,opt, exp_result_dir):

        device = torch.device('cuda')
        projected_w1_x = np.load(opt.projected_w1)['w']
        projected_w1_x = torch.tensor(projected_w1_x, device=device)      
        projected_w2_x = np.load(opt.projected_w2)['w']
        projected_w2_x = torch.tensor(projected_w2_x, device=device)     
        # print("projected_w1_x.shape:",projected_w1_x.shape)                         #   projected_w1_x.shape: torch.Size([1, 8, 512])

        projected_w_set_x = torch.cat((projected_w1_x,projected_w2_x),dim=0)
        # print("projected_w_set_x.shape：",projected_w_set_x.shape)                  #   projected_w_set_x.shape： torch.Size([2, 8, 512])

        # w1_npz_name = os.path.basename(opt.projected_w1)
        # w2_npz_name = os.path.basename(opt.projected_w2)
        # print("w1_npz_name:",w1_npz_name)
        # print("w2_npz_name:",w2_npz_name)

        # projected_w1_y = int(w1_npz_name[21:22])
        # projected_w1_y = projected_w1_y * torch.ones(projected_w_set_x.size(1), dtype = int)                                        
        # projected_w2_y = int(w2_npz_name[21:22])
        # projected_w2_y = projected_w2_y * torch.ones(projected_w_set_x.size(1), dtype = int)     

        projected_w1_y = np.load(opt.projected_w1_label)['w']
        projected_w1_y = torch.tensor(projected_w1_y, device=device)      
        projected_w2_y = np.load(opt.projected_w2_label)['w']
        projected_w2_y = torch.tensor(projected_w2_y, device=device)     

        # print("projected_w1_y.shape：",projected_w1_y.shape)                        #   projected_w1_y.shape： torch.Size([1, 8])
        # print("projected_w2_y.shape：",projected_w2_y.shape)                        #   projected_w2_y.shape： torch.Size([1, 8])
        # print("projected_w1_y:",projected_w1_y)                                       #     projected_w1_y: tensor([[6, 6, 6, 6, 6, 6, 6, 6]], device='cuda:0')
        # print("projected_w2_y:",projected_w2_y)                                       # projected_w2_y: tensor([[9, 9, 9, 9, 9, 9, 9, 9]], device='cuda:0')

        projected_w_set_y = torch.cat((projected_w1_y,projected_w2_y),dim=0)
        # print("projected_w_set_y.shape：",projected_w_set_y.shape)                  #   projected_w_set_y.shape： torch.Size([2, 8])
        projected_w_set_y = torch.nn.functional.one_hot(projected_w_set_y, opt.n_classes).float().to(device)                            
        # print("projected_w_set_y.shape：",projected_w_set_y.shape)                  #   projected_w_set_y.shape： torch.Size([2, 8, 10])
        # raise error
        interpolated_w_set, interpolated_y_set = self.__getmixededwy__(opt, projected_w_set_x,projected_w_set_y,exp_result_dir)

        return interpolated_w_set, interpolated_y_set

    def __ThreeMixup__(self,opt, exp_result_dir):
        print("flag: ThreeMixup")

        device = torch.device('cuda')
        projected_w1_x = np.load(opt.projected_w1)['w']
        projected_w1_x = torch.tensor(projected_w1_x, device=device)      
        projected_w2_x = np.load(opt.projected_w2)['w']
        projected_w2_x = torch.tensor(projected_w2_x, device=device)   
        projected_w3_x = np.load(opt.projected_w3)['w']
        projected_w3_x = torch.tensor(projected_w3_x, device=device)            
        # print("projected_w1_x.shape:",projected_w1_x.shape)                         #   projected_w1_x.shape: torch.Size([1, 8, 512])

        projected_w_set_x = torch.cat((projected_w1_x,projected_w2_x,projected_w3_x),dim=0)
        print("projected_w_set_x.shape：",projected_w_set_x.shape)                  #   projected_w_set_x.shape： torch.Size([2, 8, 512])

        # w1_npz_name = os.path.basename(opt.projected_w1)
        # w2_npz_name = os.path.basename(opt.projected_w2)
        # print("w1_npz_name:",w1_npz_name)
        # print("w2_npz_name:",w2_npz_name)

        # projected_w1_y = int(w1_npz_name[21:22])
        # projected_w1_y = projected_w1_y * torch.ones(projected_w_set_x.size(1), dtype = int)                                        
        # projected_w2_y = int(w2_npz_name[21:22])
        # projected_w2_y = projected_w2_y * torch.ones(projected_w_set_x.size(1), dtype = int)     

        projected_w1_y = np.load(opt.projected_w1_label)['w']
        projected_w1_y = torch.tensor(projected_w1_y, device=device)      
        projected_w2_y = np.load(opt.projected_w2_label)['w']
        projected_w2_y = torch.tensor(projected_w2_y, device=device)     
        projected_w3_y = np.load(opt.projected_w3_label)['w']
        projected_w3_y = torch.tensor(projected_w3_y, device=device)     

        # print("projected_w1_y.shape：",projected_w1_y.shape)                        #   projected_w1_y.shape： torch.Size([1, 8])
        # print("projected_w2_y.shape：",projected_w2_y.shape)                        #   projected_w2_y.shape： torch.Size([1, 8])
        # print("projected_w1_y:",projected_w1_y)                                       #     projected_w1_y: tensor([[6, 6, 6, 6, 6, 6, 6, 6]], device='cuda:0')
        # print("projected_w2_y:",projected_w2_y)                                       # projected_w2_y: tensor([[9, 9, 9, 9, 9, 9, 9, 9]], device='cuda:0')

        projected_w_set_y = torch.cat((projected_w1_y,projected_w2_y,projected_w3_y),dim=0)
        print("projected_w_set_y.shape：",projected_w_set_y.shape)                  #   projected_w_set_y.shape： torch.Size([2, 8])
        projected_w_set_y = torch.nn.functional.one_hot(projected_w_set_y, opt.n_classes).float().to(device)                            
        # print("projected_w_set_y.shape：",projected_w_set_y.shape)                  #   projected_w_set_y.shape： torch.Size([2, 8, 10])
        # raise error
        interpolated_w_set, interpolated_y_set = self.__getmixededwy__(opt, projected_w_set_x,projected_w_set_y,exp_result_dir)

        return interpolated_w_set, interpolated_y_set

    def __DatasetMixup__(self,opt,exp_result_dir):
        file_dir=os.listdir(opt.projected_dataset)                                                                              #   提取npz文件所在目录下所有文件
        file_dir.sort()                                                                                                         #   排序

        npzfile_name = []
        for name in file_dir:                                                                                                   #   选择指定目录下的.npz文件名
            if os.path.splitext(name)[-1] == '.npz':
                npzfile_name.append(name)                                                                                       #   name指代了list中的object 
                # 00000000-1-1-projected_w.npz
                # 00000000-1-1-label.npz
        projected_w_npz_paths =[]
        label_npz_paths = []
        for name in npzfile_name:
            if name[-15:-4] == 'projected_w':   #   倒数几位
                projected_w_npz_paths.append(f'{opt.projected_dataset}/{name}')

            elif name[-9:-4] == 'label':
                label_npz_paths.append(f'{opt.projected_dataset}/{name}')

        if opt.mix_w_num == 2:
            print("flag: DatasetTwoMixup")
        #     interpolated_w_set, interpolated_y_set = self.__Dataset2Mixup__(opt,exp_result_dir,projected_w_npz_paths,label_npz_paths)
        
        elif opt.mix_w_num == 3:
            print("flag: DatasetThreeMixup")
        #     interpolated_w_set, interpolated_y_set = self.__Dataset3Mixup__(opt,exp_result_dir,projected_w_npz_paths,label_npz_paths)        
        
        else:
            raise Exception('please input valid w_num: 2 or 3')

        device = torch.device('cuda')

        #   注意空间使用
        projected_w_set_x = []       
        for projected_w_path in projected_w_npz_paths:                                                                                   
            w = np.load(projected_w_path)['w']
            w = torch.tensor(w, device=device)                                                                                 
            w = w[-1]                                                                                       #   w.shape: torch.Size([1, 8,512]))         
            projected_w_set_x.append(w)                                                                                         
        projected_w_set_x = torch.stack(projected_w_set_x)           
        # print("projected_w_set_x.shape:",projected_w_set_x.shape)                                         #   projected_w_set_x.shape: torch.Size([37, 8, 512])
                                                                                                            #   stl10 projected_w_set_x.shape: torch.Size([38, 10, 512])

        projected_w_set_y = []       
        for label_npz_path in label_npz_paths:                                                                                  
            y = np.load(label_npz_path)['w']
            y = torch.tensor(y, device=device)                                                                                 
            y = y[-1]                                                                                       #   y.shape: torch.Size([1, 8]))
            projected_w_set_y.append(y)
        projected_w_set_y = torch.stack(projected_w_set_y)           
        # print("projected_w_set_y.shape:",projected_w_set_y.shape)                                         #   projected_w_set_y.shape: torch.Size([37, 8])  
                                                                                                            #   projected_w_set_y.shape: torch.Size([38, 10])
        projected_w_set_y = torch.nn.functional.one_hot(projected_w_set_y, opt.n_classes).float().to(device)                           
        # print("projected_w_set_y.shape:",projected_w_set_y.shape)                                         #   projected_w_set_y.shape: torch.Size([37, 8, 10])
                                                                                                            #   projected_w_set_y.shape: torch.Size([38, 10, 10])

        interpolated_w_set, interpolated_y_set = self.__getmixededwy__(opt, projected_w_set_x,projected_w_set_y,exp_result_dir)

        return interpolated_w_set, interpolated_y_set

#-----------------生成--------------------
    def genxyset(self):
        return self.generated_x_set, self.generated_y_set
        
    def generate(self, exp_result_dir, interpolated_w_set = None, interpolated_y_set = None):
        self._exp_result_dir = exp_result_dir
        generated_x_set, generated_y_set = self.__generatemain__(self._args, self._exp_result_dir, interpolated_w_set, interpolated_y_set)
        self.generated_x_set = generated_x_set
        self.generated_y_set = generated_y_set

    def __generatemain__(self, opt, exp_result_dir, interpolated_w_set, interpolated_y_set):
        # print("running generate main()..............")

        if interpolated_w_set is not None:
            # print("Generate mixed samples from mixed projectors numpy ndarray !")
            self.interpolated_w_set = interpolated_w_set
            self.interpolated_y_set = interpolated_y_set
            generated_x_set, generated_y_set = self.__generatefromntensor__()

        else:
            print("Generate mixed samples from mixed projectors npz files !")

            if opt.mixed_dataset != None:
                generated_x_set, generated_y_set = self.__generate_dataset__(opt, exp_result_dir)
            
            elif opt.mixed_dataset == None:
                if opt.projected_w is not None:
                    print("根据projected_w生成图像")
                    generated_x_set, generated_y_set = self.__generate_images__(
                        ctx = click.Context,                                                                                        #   没调试好
                        network_pkl = opt.gen_network_pkl,
                        # seeds = opt.seeds,
                        # seeds = [600, 601, 602, 603, 604, 605],
                        seeds = [500, 501, 502, 503, 504, 505],
                        truncation_psi = opt.truncation_psi,
                        noise_mode = opt.noise_mode,
                        outdir = exp_result_dir,
                        class_idx = opt.class_idx,
                        projected_w = opt.projected_w,
                        mixed_label_path = None
                        # mixed_label_path = opt.projected_w_label
                    )
                elif opt.projected_w is None:
                    print("根据seed生成图像")
                    print("opt.generate_seeds:",opt.generate_seeds)
                    generated_x_set, generated_y_set = self.__generate_images__(
                        ctx = click.Context,                                                                                        #   没调试好
                        network_pkl = opt.gen_network_pkl,
                        seeds = opt.generate_seeds,
                        truncation_psi = opt.truncation_psi,
                        noise_mode = opt.noise_mode,
                        outdir = exp_result_dir,
                        class_idx = opt.class_idx,
                        projected_w = opt.projected_w,
                        mixed_label_path = None
                        # mixed_label_path = opt.projected_w_label
                    )              
                    raise error      
            
        return generated_x_set, generated_y_set

    def __generatefromntensor__(self):
        exp_result_dir = self._exp_result_dir

        opt = self._args

        exp_result_dir = os.path.join(exp_result_dir,f'generate-{opt.dataset}-trainset')
        os.makedirs(exp_result_dir,exist_ok=True)    

        # print("self.interpolated_w_set.shape:",self.interpolated_w_set.shape)
        # print("self.interpolated_y_set.shape:",self.interpolated_y_set.shape)

        """
        self.interpolated_w_set.shape: torch.Size([4, 8, 512])
        self.interpolated_y_set.shape: torch.Size([4, 8, 10])
        """
        interpolated_w_set = self.interpolated_w_set
        interpolated_y_set = self.interpolated_y_set
        #----------maggie add----------
        generated_x_set = []
        generated_y_set = []
        #------------------------------

        for i in range(len(interpolated_w_set)):
            generated_x, generated_y = self.__imagegeneratefromwset__(
                ctx = click.Context, #  没调试好
                network_pkl = opt.gen_network_pkl,
                seeds = [500, 501, 502, 503, 504, 505],
                truncation_psi = opt.truncation_psi,
                noise_mode = opt.noise_mode,
                outdir = exp_result_dir,
                class_idx = opt.class_idx,
                interpolated_w = interpolated_w_set[i],
                interpolated_y = interpolated_y_set[i],
                interpolated_w_index = i
            )

            generated_x_set.append(generated_x)
            generated_y_set.append(generated_y)

        return generated_x_set, generated_y_set

    def __imagegeneratefromwset__(self,                                 #   和self.__generate_images__()的区别在于输入不是w的文件路径而是w的张量
        ctx: click.Context,
        network_pkl: str,
        seeds: Optional[List[int]],
        truncation_psi: float,
        noise_mode: str,
        outdir: str,
        class_idx: Optional[int],
        interpolated_w: torch.tensor,
        interpolated_y: torch.tensor,      
        interpolated_w_index: int                                     
    ):

        # print(f"generating {interpolated_w_index:08d} mixed imgae")

        device = torch.device('cuda')
        with dnnlib.util.open_url(network_pkl) as f:
            G = legacy.load_network_pkl(f)['G_ema'].to(device)                                                                  #   type: ignore

        # Synthesize the result of a W projection.                                #   投影合成,如果投影向量不为空，就根据投影向量生成样本；否则根据随机数种子生成样本
        if interpolated_w is not None:
            
            ws = interpolated_w.unsqueeze(0)                                                                                    #   ws.shape: torch.Size([1, 8, 512]) ws里本来可能有一个batch的w,现在只有1个w
            # print("ws:",ws)
            # print("ws.shape: ",ws.shape)    #   ws.shape:  torch.Size([1, 8, 512])      #   ws.shape:  torch.Size([1, 8, 512])

            mixed_label = interpolated_y.unsqueeze(0)                                                   
            # print("flag A mixed_label.shape:",mixed_label.shape)                                                              #   mixed_label.shape: torch.Size([1, 8, 10])
            mixed_label = mixed_label[-1]
            # print("mixed_label:",mixed_label)
            # print("flag B mixed_label.shape:",mixed_label.shape)                                                              #   mixed_label.shape: torch.Size([8, 10])
            #------------------------------------------------------

            #------maggie add----------
            # print("计算混合label")
            mixed_label = mixed_label[-1].unsqueeze(0)                                                                          #   mixed_label.shape: torch.Size([1, 10]) [[1,0,0,...,0]
            
            # print("mixed_label:",mixed_label)   
            # print("mixed_label.shape:",mixed_label.shape)                                                                       #   mixed_label.shape: torch.Size([1, 10])

            #   求第一大概率标签
            _, w1_label_index = torch.max(mixed_label, 1)       
            # print(f'w1_label_index = {int(w1_label_index)}')
            

            if self._args.mix_w_num == 2:
            
                # #------------maggie------------------------------------ 
                # ws = interpolated_w.unsqueeze(0)                                                                                    #   ws.shape: torch.Size([1, 8, 512]) ws里本来可能有一个batch的w,现在只有1个w
                # # print("ws:",ws)
                # # print("ws.shape: ",ws.shape)

                # mixed_label = interpolated_y.unsqueeze(0)                                                   
                # # print("flag A mixed_label.shape:",mixed_label.shape)                                                              #   mixed_label.shape: torch.Size([1, 8, 10])
                # mixed_label = mixed_label[-1]
                # # print("mixed_label:",mixed_label)
                # # print("flag B mixed_label.shape:",mixed_label.shape)                                                              #   mixed_label.shape: torch.Size([8, 10])
                # #------------------------------------------------------

                # #------maggie add----------
                # # print("计算混合label")
                # mixed_label = mixed_label[-1].unsqueeze(0)                                                                          #   mixed_label.shape: torch.Size([1, 10]) [[1,0,0,...,0]
                
                # # print("mixed_label:",mixed_label)   
                # # print("mixed_label.shape:",mixed_label.shape)                                                                       #   mixed_label.shape: torch.Size([1, 10])

                # #   求第一大概率标签
                # _, w1_label_index = torch.max(mixed_label, 1)       
                # # print(f'w1_label_index = {int(w1_label_index)}')
                
                #   求第二大概率标签
                modified_mixed_label = copy.deepcopy(mixed_label)
                modified_mixed_label[0][w1_label_index] = 0                             
                # print("modified_mixed_label:",modified_mixed_label)                                                                 #   [[0,0,0,...,0]] modified_mixed_label.shape=[1,10]
                
                # 当两个样本是同一类时,将最大置零后，会使得标签2被随机分配为label 0，例如[0,0,0,1,0,0]
    
                # print("torch.nonzero(modified_mixed_label[0]): ",torch.nonzero(modified_mixed_label[0]))                            #   torch.nonzero([0,0,0,...,0]) = tensor[] 其中size(0,1)
                # print("torch.nonzero(modified_mixed_label[0]).size(0): ",torch.nonzero(modified_mixed_label[0]).size(0))            #   torch.size(0,1)
                if torch.nonzero(modified_mixed_label[0]).size(0) == 0:
                    # print("混合label的最大值维度置零后，其他全为0！")
                    w2_label_index = w1_label_index
                    # raise Exception("maggie stop here")
                else:
                    _, w2_label_index = torch.max(modified_mixed_label, 1)
                    # print(f'w2_label_index = {int(w2_label_index)}')

                
                # classification = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
                classification = self.__labelnames__()

                #--------------------------
                # print("G.num_ws: ", G.num_ws)                                                                                     #   G.num_ws:  8          #   G.num_ws:  10
                # print("G.w_dim: ", G.w_dim)                                                                                       #   G.w_dim:  512         #   G.w_dim:  512
                # print("ws.shape[1:]:",ws.shape[1:])                                                                                 #   ws.shape[1:]: torch.Size([8, 512])
                # # raise error
                assert ws.shape[1:] == (G.num_ws, G.w_dim)                                                                          #   断言的功能是，在满足条件时，程序go on，在不满足条件时，报错
                for _, w in enumerate(ws):

                    # print("flag 0: mixed projecte w : ",w)                                                                        #   normalized
                    # print("flag 0: mixed projecte w.type: ",type(w))                     
                    # print("flag 0: mixed projecte w.shape: ",w.shape)                                                             #   mixed projecte w.shape:  torch.Size([8, 512])

                    img = G.synthesis(w.unsqueeze(0), noise_mode=noise_mode)

                    # print("flag 1: generated img: ",img)                                                                          #   normalized [0,1] float
                    # print("flag 1: generated img.type: ",type(img))                                                               #   torch.tensor
                    # print("flag 1: generated img.shape: ",img.shape)                                     #   flag 1: generated img.shape:  torch.Size([1, 1, 32, 32])cifar10数据集
                    #   flag 1: generated img.shape:  torch.Size([1, 3, 32, 32])
                    # raise error
                    #-----------------maggie add-----------
                    generated_x = img[-1]            
                    # print("generated_x[0]: ",generated_x[0])                                                                        #   generated_x[0]:  tensor([[ 0.0170, -0.2638, -0.4614,  ...,  0.6033,  0.4530, -0.0071]
                    # print("generated_x.type: ",type(generated_x))                                                                   #   generated_x.type:  <class 'torch.Tensor'>
                    # print("generated_x.shape: ",generated_x.shape)                                        #   generated_x.shape:  torch.Size([1, 32, 32]) generated_x.shape:  torch.Size([3, 32, 32])

                    generated_y = mixed_label[-1]
                    # print("generated_y: ",generated_y)                                                                              #  generated_y:  tensor([0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.5298, 0.0000, 0.0000,0.4702],
                    # print("generated_y.type: ",type(generated_y))                                                                   #   torch.tensor
                    # print("generated_y.shape: ",generated_y.shape)                                        #   generated_y.shape:  torch.Size([10])
                    # #--------------------------------------


                    # synth_image = (synth_image + 1) * (255/2)
                    # synth_image = synth_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()

                    #   为存储图片进行格式转换
                    img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
                    # print("flag 2: modified generated img: ",img)                                                                 #   [0,255] uint8
                    # print("flag 2: modified generated img.type: ",type(img))                                                      #   torch.tensor
                    # print("flag 2: modified generated img.shape: ",img.shape)                                                     #   flag 2: modified generated img.shape:  (32, 32, 1)    flag 2: modified generated img.shape:  (32, 32, 3)

                    w1_label_name = f"{classification[int(w1_label_index)]}"
                    w2_label_name = f"{classification[int(w2_label_index)]}"
    
                    # print("img.size():",img.size())             #   img.size(): torch.Size([1, 32, 32, 1])
                    # print("img.size(0):",img.size(0))
                    # print("img.size(1):",img.size(1))
                    # print("img.size(2):",img.size(2))
                    # print("img.size(3):",img.size(3))           #   img.size(3): 1

                    if self._args.dataset != 'kmnist' and self._args.dataset != 'mnist':
                        # img_pil = PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB')
                        img_pil = PIL.Image.fromarray(img, 'RGB')

                        # label_path = f'{outdir}/{interpolated_w_index:08d}-{int(w1_label_index)}-{w1_label_name}+{int(w2_label_index)}-{w2_label_name}-mixed_label.npz'
                        # np.savez(label_path, w = mixed_label.unsqueeze(0).cpu().numpy())                                                #   将latent code w村委outdir定义的输出路径下的projected_w.npz文件
                    elif self._args.dataset == 'kmnist' or self._args.dataset == 'mnist':
                        
                        # print("img.shape:",img.shape)           #   img.shape: (32, 32, 1)

                        # img = img[0]
                        # print("img.shape:",img.shape)          

                        img = img.transpose([2, 0, 1])
                        # print("img.shape:",img.shape)           #   img.shape: (1, 32, 32)

                        img = img[0]
                        # print("img.shape:",img.shape)           #   img.shape: (32, 32)

                        img_pil = PIL.Image.fromarray(img, 'L')

                    #   关闭存储
                    # img = img_pil.save(f'{outdir}/{interpolated_w_index:08d}-{int(w1_label_index)}-{w1_label_name}+{int(w2_label_index)}-{w2_label_name}-mixed-image.png')       #   idx实则一直为0，因为ws中只有一个w，该函数是处理单张投影向量的

                    if self._args.defense_mode != 'rmt':
                        np.savez(f'{outdir}/{interpolated_w_index:08d}-{int(w1_label_index)}-{w1_label_name}+{int(w2_label_index)}-{w2_label_name}-mixed-image.npz', w = generated_x.cpu().numpy())                                                #   将latent code w村委outdir定义的输出路径下的projected_w.npz文件
                        np.savez(f'{outdir}/{interpolated_w_index:08d}-{int(w1_label_index)}-{w1_label_name}+{int(w2_label_index)}-{w2_label_name}-mixed-label.npz', w = generated_y.cpu().numpy())                                                #   将latent code w村委outdir定义的输出路径下的projected_w.npz文件
                
                    # print(f'{outdir}/{interpolated_w_index:08d}-{int(w1_label_index)}-{w1_label_name}+{int(w2_label_index)}-{w2_label_name}')
                    #   result/interpolate/basemixup/uniformsampler/stylegan2ada-cifar10/20210709/00004/generate-cifar10-trainset/00000000-9-truck+6-frog
                    # raise Exception("maggie stop")

            elif self._args.mix_w_num == 3:
            
                # ws = interpolated_w.unsqueeze(0)                                                                                    #   ws.shape: torch.Size([1, 8, 512]) ws里本来可能有一个batch的w,现在只有1个w
                # # print("ws:",ws)
                # # print("ws.shape: ",ws.shape)

                # mixed_label = interpolated_y.unsqueeze(0)                                                   
                # # print("flag A mixed_label.shape:",mixed_label.shape)                                                              #   mixed_label.shape: torch.Size([1, 8, 10])
                # mixed_label = mixed_label[-1]
                # # print("mixed_label:",mixed_label)
                # # print("flag B mixed_label.shape:",mixed_label.shape)                                                              #   mixed_label.shape: torch.Size([8, 10])
                # #------------------------------------------------------

                # #------maggie add----------
                # # print("计算混合label")
                # mixed_label = mixed_label[-1].unsqueeze(0)                                                                          #   mixed_label.shape: torch.Size([1, 10]) [[1,0,0,...,0]
                
                # # print("mixed_label:",mixed_label)   
                # # print("mixed_label.shape:",mixed_label.shape)                                                                       #   mixed_label.shape: torch.Size([1, 10])

                # #   求第一大概率标签
                # _, w1_label_index = torch.max(mixed_label, 1)       
                # # print(f'w1_label_index = {int(w1_label_index)}')
                
                #   求第二大概率标签
                modified_mixed_label = copy.deepcopy(mixed_label)
                modified_mixed_label[0][w1_label_index] = 0                             
                # print("modified_mixed_label:",modified_mixed_label)                                                                 #   [[0,0,0,...,0]] modified_mixed_label.shape=[1,10]
                
                # 当两个样本是同一类时,将最大置零后，会使得标签2被随机分配为label 0，例如[0,0,0,1,0,0]
    
                # print("torch.nonzero(modified_mixed_label[0]): ",torch.nonzero(modified_mixed_label[0]))                            #   torch.nonzero([0,0,0,...,0]) = tensor[] 其中size(0,1)
                # print("torch.nonzero(modified_mixed_label[0]).size(0): ",torch.nonzero(modified_mixed_label[0]).size(0))            #   torch.size(0,1)
                
                if torch.nonzero(modified_mixed_label[0]).size(0) == 0:
                    # print("混合label的最大值维度置零后，其他全为0！")
                    w2_label_index = w1_label_index
                    w3_label_index = w1_label_index
                    # print(f'w2_label_index = {int(w2_label_index)}')
                    # print(f'w3_label_index = {int(w3_label_index)}')

                    # raise Exception("maggie stop here")
                else:
                    _, w2_label_index = torch.max(modified_mixed_label, 1)
                    # print(f'w2_label_index = {int(w2_label_index)}')

                    #   求第三大概率标签
                    modified2_mixed_label = copy.deepcopy(modified_mixed_label)
                    modified2_mixed_label[0][w2_label_index] = 0
                    
                    if torch.nonzero(modified2_mixed_label[0]).size(0) == 0:
                        # print("混合label的最大值维度置零后，其他全为0！")
                        w3_label_index = w2_label_index
                        # print(f'w3_label_index = {int(w3_label_index)}')
                        # raise Exception("maggie stop here")
                    else:
                        _, w3_label_index = torch.max(modified2_mixed_label, 1)
                        # print(f'w3_label_index = {int(w3_label_index)}')                

                # raise error
                # classification = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
                classification = self.__labelnames__()

                #--------------------------
                # print("G.num_ws: ", G.num_ws)                                                                                     #   8
                # print("G.w_dim: ", G.w_dim)                                                                                       #   512

                assert ws.shape[1:] == (G.num_ws, G.w_dim)                                                                          #   断言的功能是，在满足条件时，程序go on，在不满足条件时，报错
                for _, w in enumerate(ws):

                    # print("flag 0: mixed projecte w : ",w)                                                                        #   normalized
                    # print("flag 0: mixed projecte w.type: ",type(w))                     
                    # print("flag 0: mixed projecte w.shape: ",w.shape)                                                             #   mixed projecte w.shape:  torch.Size([8, 512])

                    img = G.synthesis(w.unsqueeze(0), noise_mode=noise_mode)

                    # print("flag 1: generated img: ",img)                                                                          #   normalized [0,1] float
                    # print("flag 1: generated img.type: ",type(img))                                                               #   torch.tensor
                    # print("flag 1: generated img.shape: ",img.shape)                                                              #   generated img.shape:  torch.Size([1, 3, 32, 32]) cifar10数据集

                    #-----------------maggie add-----------
                    generated_x = img[-1]            
                    # print("generated_x[0]: ",generated_x[0])                                                                        #   generated_x[0]:  tensor([[ 0.0170, -0.2638, -0.4614,  ...,  0.6033,  0.4530, -0.0071]
                    # print("generated_x.type: ",type(generated_x))                                                                   #   generated_x.type:  <class 'torch.Tensor'>
                    # print("generated_x.shape: ",generated_x.shape)                                                                  #   generated_x.shape:  torch.Size([3, 32, 32])
                    
                    generated_y = mixed_label[-1]
                    # print("generated_y: ",generated_y)                                                                              #  generated_y:  tensor([0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.5298, 0.0000, 0.0000,0.4702],
                    # print("generated_y.type: ",type(generated_y))                                                                   #   torch.tensor
                    # print("generated_y.shape: ",generated_y.shape)                                                                  #   generated_y.shape:  torch.Size([10])
                    # #--------------------------------------

                    #   为存储图片进行格式转换
                    img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
                    # print("flag 2: modified generated img: ",img)                                                                 #   [0,255] uint8
                    # print("flag 2: modified generated img.type: ",type(img))                                                      #   torch.tensor
                    # print("flag 2: modified generated img.shape: ",img.shape)                                                     #   img.shape:  torch.Size([1, 32, 32, 3])

                    w1_label_name = f"{classification[int(w1_label_index)]}"
                    w2_label_name = f"{classification[int(w2_label_index)]}"
                    w3_label_name = f"{classification[int(w3_label_index)]}"

                    # img_pil = PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB')
                    if self._args.dataset != 'kmnist' and self._args.dataset != 'mnist':
                        # img_pil = PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB')
                        img_pil = PIL.Image.fromarray(img, 'RGB')

                        # label_path = f'{outdir}/{interpolated_w_index:08d}-{int(w1_label_index)}-{w1_label_name}+{int(w2_label_index)}-{w2_label_name}-mixed_label.npz'
                        # np.savez(label_path, w = mixed_label.unsqueeze(0).cpu().numpy())                                                #   将latent code w村委outdir定义的输出路径下的projected_w.npz文件
                    elif self._args.dataset == 'kmnist' or self._args.dataset == 'mnist':
                        
                        # print("img.shape:",img.shape)           #   img.shape: (32, 32, 1)

                        # img = img[0]
                        # print("img.shape:",img.shape)          

                        img = img.transpose([2, 0, 1])
                        # print("img.shape:",img.shape)           #   img.shape: (1, 32, 32)

                        img = img[0]
                        # print("img.shape:",img.shape)           #   img.shape: (32, 32)

                        img_pil = PIL.Image.fromarray(img, 'L')                                      
                    
                    img = img_pil.save(f'{outdir}/{interpolated_w_index:08d}-{int(w1_label_index)}-{w1_label_name}+{int(w2_label_index)}-{w2_label_name}+{int(w3_label_index)}-{w3_label_name}-mixed-image.png')                                         #   idx实则一直为0，因为ws中只有一个w，该函数是处理单张投影向量的
                    
                    # label_path = f'{outdir}/{interpolated_w_index:08d}-{int(w1_label_index)}-{w1_label_name}+{int(w2_label_index)}-{w2_label_name}-mixed_label.npz'
                    # np.savez(label_path, w = mixed_label.unsqueeze(0).cpu().numpy())                                                #   将latent code w村委outdir定义的输出路径下的projected_w.npz文件
                
                    if self._args.defense_mode != 'rmt':

                        np.savez(f'{outdir}/{interpolated_w_index:08d}-{int(w1_label_index)}-{w1_label_name}+{int(w2_label_index)}-{w2_label_name}+{int(w3_label_index)}-{w3_label_name}-mixed-image.npz', w = generated_x.cpu().numpy())                                                #   将latent code w村委outdir定义的输出路径下的projected_w.npz文件
                        np.savez(f'{outdir}/{interpolated_w_index:08d}-{int(w1_label_index)}-{w1_label_name}+{int(w2_label_index)}-{w2_label_name}+{int(w3_label_index)}-{w3_label_name}-mixed-label.npz', w = generated_y.cpu().numpy())                                                #   将latent code w村委outdir定义的输出路径下的projected_w.npz文件
                
                    # print(f'{outdir}/{interpolated_w_index:08d}-{int(w1_label_index)}-{w1_label_name}+{int(w2_label_index)}-{w2_label_name}')
                    #   result/interpolate/basemixup/uniformsampler/stylegan2ada-cifar10/20210709/00004/generate-cifar10-trainset/00000000-9-truck+6-frog
                    # raise Exception("maggie stop")

            #------------maggie add----------------
            return generated_x, generated_y
            #--------------------------------------

        if seeds is None:
            ctx.fail('--seeds option is required when not using --projected_w')

        # Labels.
        label = torch.zeros([1, G.c_dim], device=device)
        if G.c_dim != 0:
            if class_idx is None:
                ctx.fail('Must specify class label with --class when using a conditional network')
            label[:, class_idx] = 1
        else:
            if class_idx is not None:
                print ('warn: --class=lbl ignored when running on an unconditional network')

        # Generate images.
        for seed_idx, seed in enumerate(seeds):
            print('Generating image for seed %d (%d/%d) ...' % (seed, seed_idx, len(seeds)))
            z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)
            img = G(z, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
            img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'{outdir}/seed{seed:04d}.png') 

        #------------maggie add---------------
        generated_x = img[0]
        generated_y = label
        # return generated_x, generated_y
        #------------------------------------
        return generated_x, generated_y

    def __generate_dataset__(self, opt, exp_result_dir):
        exp_result_dir = os.path.join(exp_result_dir,f'generate-{opt.dataset}-trainset')
        os.makedirs(exp_result_dir,exist_ok=True)   

        file_dir=os.listdir(opt.mixed_dataset)
        file_dir.sort()

        #-----------maggie add-------------
        mixed_projector_filenames=[]                                                                                            #   w的地址     00000991-1-class+00000992-3-class-mixed_projected_w.npz
        mixed_label_filenames=[]                                                                                                #   w的标签地址 00000995-3-class+00000996-5-class-mixed_label.npz
        #----------------------------------
        
        for name in file_dir:
            # print(name)
            if name[-15:-4] == 'projected_w':
                # print(name[-15:-4])
                mixed_projector_filenames.append(name)
            elif name[-9:-4] == 'label':
                # print(name[-9:-4])
                mixed_label_filenames.append(name)

        mixed_projector_path = []
        mixed_label_path = []
        for name in mixed_projector_filenames:
            mixed_projector_path.append(f'{opt.mixed_dataset}/{name}')

        for name in mixed_label_filenames:
            mixed_label_path.append(f'{opt.mixed_dataset}/{name}')

        # for i in range(len(mixed_projector_path)):
        #     print(mixed_projector_filenames[i])
        #     print(mixed_label_filenames[i])
        #     print(mixed_projector_path[i])
        #     print(mixed_label_path[i])

        #----------maggie add----------
        generated_x_set = []
        generated_y_set = []
        #------------------------------

        for i in range(len(mixed_projector_path)):
            # if i<3:
            # print("mixed_projector_path[i]:",mixed_projector_path[i])
            # print("mixed_label_path[i]:",mixed_label_path[i])
            generated_x, generated_y = self.__generate_images__(
                ctx = click.Context, #  没调试好
                network_pkl = opt.gen_network_pkl,
                # seeds = opt.seeds,
                # seeds = [600, 601, 602, 603, 604, 605],
                seeds = [500, 501, 502, 503, 504, 505],
                truncation_psi = opt.truncation_psi,
                noise_mode = opt.noise_mode,
                outdir = exp_result_dir,
                class_idx = opt.class_idx,
                projected_w = mixed_projector_path[i],
                mixed_label_path = mixed_label_path[i]
            )

            generated_x_set.append(generated_x)
            generated_y_set.append(generated_y)

        return generated_x_set, generated_y_set
    
    def __generate_images__(self,
        ctx: click.Context,
        network_pkl: str,
        seeds: Optional[List[int]],
        truncation_psi: float,
        noise_mode: str,
        outdir: str,
        class_idx: Optional[int],
        projected_w: Optional[str],
        mixed_label_path:Optional[str]                                                                                          #   maggie add
    ):

        print('Loading networks from "%s"...' % network_pkl)
        device = torch.device('cuda')
        with dnnlib.util.open_url(network_pkl) as f:
            G = legacy.load_network_pkl(f)['G_ema'].to(device)                                                                  #   type: ignore

        # Synthesize the result of a W projection.                                                                              #   投影合成,如果投影向量不为空，就根据投影向量生成样本；否则根据随机数种子生成样本
        if projected_w is not None:
            # if seeds is not None:
            #     #print ('warn: --seeds is ignored when using --projected-w')
            #     print ('warn: --seeds is ignored when using --projected_w')
            print(f'Generating images from projected W "{projected_w}"')        
            ws = np.load(projected_w)['w']
            ws = torch.tensor(ws, device=device) # pylint: disable=not-callable

            #------------maggie------------------------------------
            mixed_label = np.load(mixed_label_path)['w']
            mixed_label = torch.tensor(mixed_label, device=device)                                                              #   pylint: disable=not-callable
            mixed_label = mixed_label[-1]
            # print(mixed_label)
            # print(mixed_label.shape)        #   torch.Size([8, 10])
            #------------------------------------------------------
            #-----------maggie-------------------------------------
            # print('ws=%s' % ws)#  maggie
            # print(ws.shape[1:])
            # print('(G.num_ws=%s, G.w_dim=%s)' % (G.num_ws, G.w_dim))
            #------------------------------------------------------

            assert ws.shape[1:] == (G.num_ws, G.w_dim)                                                                          #   断言的功能是，在满足条件时，程序go on，在不满足条件时，报错
            for idx, w in enumerate(ws):

                # print("flag 0: mixed projecte w : ",w)                                                                        #   normalized
                # print("flag 0: mixed projecte w.type: ",type(w))                                                              #   torch.tensor
                # print("flag 0: mixed projecte w.shape: ",w.shape)                                                             #   [8,512] 

                img = G.synthesis(w.unsqueeze(0), noise_mode=noise_mode)
                # np.savez(f'{outdir}/{w1_name}+{w2_name}-generated-image.npz', w=img.cpu().numpy())                              #   存图片npz, projected_w.unsqueeze(0).shape:  torch.Size([1, 8, 512])



                # print("flag 1: generated img: ",img)                                                                          #   normalized [0,1] float
                # print("flag 1: generated img.type: ",type(img))                                                               #   torch.tensor
                # print("flag 1: generated img.shape: ",img.shape)                                                              #   [1,3,32,32] cifar10数据集

               #-----------------maggie add-----------
                generated_x = img[-1]            
                # print("generated_x[0]: ",generated_x[0])                                                                        #   generated_x[0]:  tensor([[ 0.9688,  0.9599,  0.9377,  ..., -0.0858, -0.1952, -0.3162],,
                # print("generated_x.type: ",type(generated_x))                                                                   #   generated_x.type:  <class 'torch.Tensor'>
                # print("generated_x.shape: ",generated_x.shape)                                                                  #   generated_x.shape:  torch.Size([3, 32, 32])
                
                generated_y = mixed_label[-1]
                # print("generated_y: ",generated_y)                                                                              #   generated_y:  tensor([0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.1311, 0.0000, 0.0000,0.8689],
                # print("generated_y.type: ",type(generated_y))                                                                   #   torch.tensor
                # print("generated_y.shape: ",generated_y.shape)                                                                  #   generated_y.shape:  torch.Size([10])

                img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)

                # print("flag 2: modified generated img: ",img)                                                                 #   [0,255] uint8
                # print("flag 2: modified generated img.type: ",type(img))                                                      #   torch.tensor
                # print("flag 2: modified generated img.shape: ",img.shape)                                                     #   [1,32,32,3]

                # #-----------------maggie add-----------
                # generated_x = img[0]            
                # # print("flag 3: modified generated img[0]: ",img[0])                                                           #   [0,255] uint8
                # # print("flag 3: modified generated img[0].type: ",type(img[0]))                                                #   torch.tensor
                # # print("flag 3: modified generated img[0].shape: ",img[0].shape)                                               #   [32,32,3]
            
                # generated_y = mixed_label
                # # print("flag 4: generated img label: ",mixed_label)                                                            #   [0,1] float
                # # print("flag 4: generated img label.type: ",type(mixed_label))                                                 #   torch.tensor
                # # print("flag 4: generated img label.shape: ",mixed_label.shape)                                                #   [8,10]
                # #--------------------------------------


                #   存储图片
                # img = PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'{outdir}/proj{idx:02d}.png')                    #   合成图像命名为proj+2位数的idx+.png
                #---------------------------maggie--------------------------------------
                # print(projected_w)  #/home/maggie/mmat/result/interpolate/stylegan2ada-cifar10/ID20210525/00002/00000000-6-frog+00000001-9-truck-mixed_projected_w.npz
                # print(mixed_label)  #/home/maggie/mmat/result/interpolate/stylegan2ada-cifar10/ID20210525/00002/00000000-6-frog+00000001-9-truck-mixed_label.npz

                w_name= re.findall(r'/home/maggie/mmat/.*/(.*?)\+(.*?)-mixed_projected_w.npz',projected_w)                      #   +是正则表达式中的规定字符，因此匹配加号时要在前面使用转义字符\
                # print(w_name)            
                # print(w_name[0][0])
                # print(w_name[0][1])
                w1_name =str(w_name[0][0])
                w2_name = str(w_name[0][1])

                img_pil = PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB')
                
                # img = img_pil.save(f'{outdir}/{idx:04d}-{w1_name}+{w2_name}-generated-image.png')             
                # label_path = f'{outdir}/{idx:04d}-{w1_name}+{w2_name}-mixed_label.npz'

                img = img_pil.save(f'{outdir}/{w1_name}+{w2_name}-mixed-image.png')                                         #   idx实则一直为0，因为ws中只有一个w，该函数是处理单张投影向量的
               
                label_path = f'{outdir}/{w1_name}+{w2_name}-mixed-label.npz'
                                
                np.savez(label_path, w = mixed_label.unsqueeze(0).cpu().numpy())                                                #   将latent code w村委outdir定义的输出路径下的projected_w.npz文件
              
                np.savez(f'{outdir}/{w1_name}+{w2_name}-mixed-image.npz', w = generated_x.cpu().numpy())                                                #   将latent code w村委outdir定义的输出路径下的projected_w.npz文件
                np.savez(f'{outdir}/{w1_name}+{w2_name}-mixed-label.npz', w = generated_y.cpu().numpy())                                                #   将latent code w村委outdir定义的输出路径下的projected_w.npz文件
                
                # print(f'{outdir}/{w1_name}+{w2_name}')
                #   result/interpolate/basemixup/uniformsampler/stylegan2ada-cifar10/20210709/00004/generate-cifar10-trainset/00000000-9-truck+6-frog 读tensor
                #   result/interpolate/basemixup/uniformsampler/stylegan2ada-cifar10/20210709/00003/generate-cifar10-trainset/00000000-6-frog+00000001-9-truck        读本地      
                # raise Exception("maggie stop")
              
              
                #------------------------------------------------------------------------
            # return
            #------------maggie add----------------
            return generated_x, generated_y
            #--------------------------------------

        #-----------20210903---------没有投影向量时才会往下进行--------------
        if seeds is None:
            ctx.fail('--seeds option is required when not using --projected_w')

        # Labels.
        label = torch.zeros([1, G.c_dim], device=device)
        if G.c_dim != 0:
            if class_idx is None:
                ctx.fail('Must specify class label with --class when using a conditional network')
            label[:, class_idx] = 1
        else:
            if class_idx is not None:
                print ('warn: --class=lbl ignored when running on an unconditional network')

        # Generate images.
        for seed_idx, seed in enumerate(seeds):
            print('Generating image for seed %d (%d/%d) ...' % (seed, seed_idx, len(seeds)))
            z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)
            img = G(z, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
            img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)

            #-----------maggie20210903-----------
            # print("img.shape:",img.shape)   #   img.shape: torch.Size([1, 32, 32, 1])
            _, _, _, channel_num = img.shape
            
            assert channel_num in [1, 3]
            if channel_num == 1:
                PIL.Image.fromarray(img[0][:, :, 0].cpu().numpy(), 'L').save(f'{outdir}/seed{seed:04d}.png') 
            if channel_num == 3:
                PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'{outdir}/seed{seed:04d}.png') 

            #-------------------------------------

            # PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'{outdir}/seed{seed:04d}.png') 

        #------------maggie add---------------
        generated_x = img[0]
        generated_y = label
        return generated_x, generated_y
        #------------------------------------

    def __num_range__(self, s: str) -> List[int]:
        '''Accept either a comma separated list of numbers 'a,b,c' or a range 'a-c' and return as a list of ints.'''
        range_re = re.compile(r'^(\d+)-(\d+)$')
        m = range_re.match(s)
        if m:
            return list(range(int(m.group(1)), int(m.group(2))+1))
        vals = s.split(',')
        return [int(x) for x in vals]
    
