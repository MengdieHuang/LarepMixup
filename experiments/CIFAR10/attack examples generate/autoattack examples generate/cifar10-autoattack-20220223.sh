source ~/.bashrc
source /home/xieyi/anaconda3/bin/activate mmat

# #----------------------alexnet---------------------------------
# #alexnet cifar autoattack 20220217 
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220217.py run --mode attack --attack_mode autoattack --whitebox --exp_name alexnet-cifar10 --cla_model alexnet --cla_network_pkl /home/maggie/mmat/result/train/cla-train/alexnet-cifar10/20210908/00015-testacc-0.8355/train-cifar10-dataset/standard-trained-classifier-alexnet-on-clean-cifar10-finished.pkl --dataset cifar10 --attack_eps 0.02 >> /home/maggie/mmat/log/cifar10-attack/cifar10-autoattack-20220217.log 2>&1

# # CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220217.py run --mode attack --attack_mode autoattack --whitebox --exp_name alexnet-cifar10 --cla_model alexnet --cla_network_pkl /home/maggie/mmat/result/train/cla-train/alexnet-cifar10/20210908/00015-testacc-0.8355/train-cifar10-dataset/standard-trained-classifier-alexnet-on-clean-cifar10-finished.pkl --dataset cifar10 --attack_eps 0.02 

# #alexnet cifar autoattack 20220217 
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220217.py run --mode attack --attack_mode autoattack --whitebox --exp_name alexnet-cifar10 --cla_model alexnet --cla_network_pkl /home/maggie/mmat/result/train/cla-train/alexnet-cifar10/20210908/00015-testacc-0.8355/train-cifar10-dataset/standard-trained-classifier-alexnet-on-clean-cifar10-finished.pkl --dataset cifar10 --attack_eps 0.05 >> /home/maggie/mmat/log/cifar10-attack/cifar10-autoattack-20220217.log 2>&1

# #alexnet cifar autoattack 20220217 
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220217.py run --mode attack --attack_mode autoattack --whitebox --exp_name alexnet-cifar10 --cla_model alexnet --cla_network_pkl /home/maggie/mmat/result/train/cla-train/alexnet-cifar10/20210908/00015-testacc-0.8355/train-cifar10-dataset/standard-trained-classifier-alexnet-on-clean-cifar10-finished.pkl --dataset cifar10 --attack_eps 0.1 >> /home/maggie/mmat/log/cifar10-attack/cifar10-autoattack-20220217.log 2>&1

# #alexnet cifar autoattack 20220217 
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220217.py run --mode attack --attack_mode autoattack --whitebox --exp_name alexnet-cifar10 --cla_model alexnet --cla_network_pkl /home/maggie/mmat/result/train/cla-train/alexnet-cifar10/20210908/00015-testacc-0.8355/train-cifar10-dataset/standard-trained-classifier-alexnet-on-clean-cifar10-finished.pkl --dataset cifar10 --attack_eps 0.2 >> /home/maggie/mmat/log/cifar10-attack/cifar10-autoattack-20220217.log 2>&1

# #alexnet cifar autoattack 20220217 
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220217.py run --mode attack --attack_mode autoattack --whitebox --exp_name alexnet-cifar10 --cla_model alexnet --cla_network_pkl /home/maggie/mmat/result/train/cla-train/alexnet-cifar10/20210908/00015-testacc-0.8355/train-cifar10-dataset/standard-trained-classifier-alexnet-on-clean-cifar10-finished.pkl --dataset cifar10 --attack_eps 0.3 >> /home/maggie/mmat/log/cifar10-attack/cifar10-autoattack-20220217.log 2>&1

# #----------------------resnet18---------------------------------
# #resnet18 cifar autoattack 20220218 
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220218.py run --mode attack --attack_mode autoattack --whitebox --exp_name resnet18-cifar10 --cla_model resnet18 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/resnet18-cifar10/20210908/00017-testacc-0.7779/train-cifar10-dataset/standard-trained-classifier-resnet18-on-clean-cifar10-finished.pkl --dataset cifar10 --attack_eps 0.02 >> /home/maggie/mmat/log/cifar10-attack/cifar10-resnet18-autoattack-20220218.log 2>&1

# #resnet18 cifar autoattack 20220218 
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220218.py run --mode attack --attack_mode autoattack --whitebox --exp_name resnet18-cifar10 --cla_model resnet18 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/resnet18-cifar10/20210908/00017-testacc-0.7779/train-cifar10-dataset/standard-trained-classifier-resnet18-on-clean-cifar10-finished.pkl --dataset cifar10 --attack_eps 0.05 >> /home/maggie/mmat/log/cifar10-attack/cifar10-resnet18-autoattack-20220218.log 2>&1

# #resnet18 cifar autoattack 20220218 
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220218.py run --mode attack --attack_mode autoattack --whitebox --exp_name resnet18-cifar10 --cla_model resnet18 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/resnet18-cifar10/20210908/00017-testacc-0.7779/train-cifar10-dataset/standard-trained-classifier-resnet18-on-clean-cifar10-finished.pkl --dataset cifar10 --attack_eps 0.1 >> /home/maggie/mmat/log/cifar10-attack/cifar10-resnet18-autoattack-20220218.log 2>&1

# #resnet18 cifar autoattack 20220218 
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220218.py run --mode attack --attack_mode autoattack --whitebox --exp_name resnet18-cifar10 --cla_model resnet18 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/resnet18-cifar10/20210908/00017-testacc-0.7779/train-cifar10-dataset/standard-trained-classifier-resnet18-on-clean-cifar10-finished.pkl --dataset cifar10 --attack_eps 0.2 >> /home/maggie/mmat/log/cifar10-attack/cifar10-resnet18-autoattack-20220218.log 2>&1

# #resnet18 cifar autoattack 20220218 
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220218.py run --mode attack --attack_mode autoattack --whitebox --exp_name resnet18-cifar10 --cla_model resnet18 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/resnet18-cifar10/20210908/00017-testacc-0.7779/train-cifar10-dataset/standard-trained-classifier-resnet18-on-clean-cifar10-finished.pkl --dataset cifar10 --attack_eps 0.3 >> /home/maggie/mmat/log/cifar10-attack/cifar10-resnet18-autoattack-20220218.log 2>&1

# #----------------------resnet34---------------------------------
# #resnet34 cifar autoattack 20220218 
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220218.py run --mode attack --attack_mode autoattack --whitebox --exp_name resnet34-cifar10 --cla_model resnet34 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/resnet34-cifar10/20210907/00030-testacc-0.7739/train-cifar10-dataset/standard-trained-classifier-resnet34-on-clean-cifar10-finished.pkl --dataset cifar10 --attack_eps 0.02 >> /home/maggie/mmat/log/cifar10-attack/cifar10-resnet34-autoattack-20220218.log 2>&1

# #resnet34 cifar autoattack 20220218 
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220218.py run --mode attack --attack_mode autoattack --whitebox --exp_name resnet34-cifar10 --cla_model resnet34 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/resnet34-cifar10/20210907/00030-testacc-0.7739/train-cifar10-dataset/standard-trained-classifier-resnet34-on-clean-cifar10-finished.pkl --dataset cifar10 --attack_eps 0.05 >> /home/maggie/mmat/log/cifar10-attack/cifar10-resnet34-autoattack-20220218.log 2>&1

# #resnet34 cifar autoattack 20220218 
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220218.py run --mode attack --attack_mode autoattack --whitebox --exp_name resnet34-cifar10 --cla_model resnet34 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/resnet34-cifar10/20210907/00030-testacc-0.7739/train-cifar10-dataset/standard-trained-classifier-resnet34-on-clean-cifar10-finished.pkl --dataset cifar10 --attack_eps 0.1 >> /home/maggie/mmat/log/cifar10-attack/cifar10-resnet34-autoattack-20220218.log 2>&1

# #resnet34 cifar autoattack 20220218 
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220218.py run --mode attack --attack_mode autoattack --whitebox --exp_name resnet34-cifar10 --cla_model resnet34 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/resnet34-cifar10/20210907/00030-testacc-0.7739/train-cifar10-dataset/standard-trained-classifier-resnet34-on-clean-cifar10-finished.pkl --dataset cifar10 --attack_eps 0.2 >> /home/maggie/mmat/log/cifar10-attack/cifar10-resnet34-autoattack-20220218.log 2>&1

# #resnet34 cifar autoattack 20220218 
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220218.py run --mode attack --attack_mode autoattack --whitebox --exp_name resnet34-cifar10 --cla_model resnet34 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/resnet34-cifar10/20210907/00030-testacc-0.7739/train-cifar10-dataset/standard-trained-classifier-resnet34-on-clean-cifar10-finished.pkl --dataset cifar10 --attack_eps 0.3 >> /home/maggie/mmat/log/cifar10-attack/cifar10-resnet34-autoattack-20220218.log 2>&1

#----------------------resnet50---------------------------------
# #resnet50 cifar autoattack 20220218 
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220218.py run --mode attack --attack_mode autoattack --whitebox --exp_name resnet50-cifar10 --cla_model resnet50 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/resnet50-cifar10/20210908/00002-attackacc-0.7624/train-cifar10-dataset/standard-trained-classifier-resnet50-on-clean-cifar10-finished.pkl --dataset cifar10 --attack_eps 0.02 >> /home/maggie/mmat/log/cifar10-attack/cifar10-resnet50-autoattack-20220218.log 2>&1

# #resnet50 cifar autoattack 20220218 
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220218.py run --mode attack --attack_mode autoattack --whitebox --exp_name resnet50-cifar10 --cla_model resnet50 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/resnet50-cifar10/20210908/00002-attackacc-0.7624/train-cifar10-dataset/standard-trained-classifier-resnet50-on-clean-cifar10-finished.pkl --dataset cifar10 --attack_eps 0.05 >> /home/maggie/mmat/log/cifar10-attack/cifar10-resnet50-autoattack-20220218.log 2>&1

# #resnet50 cifar autoattack 20220218 
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220218.py run --mode attack --attack_mode autoattack --whitebox --exp_name resnet50-cifar10 --cla_model resnet50 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/resnet50-cifar10/20210908/00002-attackacc-0.7624/train-cifar10-dataset/standard-trained-classifier-resnet50-on-clean-cifar10-finished.pkl --dataset cifar10 --attack_eps 0.1 >> /home/maggie/mmat/log/cifar10-attack/cifar10-resnet50-autoattack-20220218.log 2>&1

# #resnet50 cifar autoattack 20220218 
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220218.py run --mode attack --attack_mode autoattack --whitebox --exp_name resnet50-cifar10 --cla_model resnet50 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/resnet50-cifar10/20210908/00002-attackacc-0.7624/train-cifar10-dataset/standard-trained-classifier-resnet50-on-clean-cifar10-finished.pkl --dataset cifar10 --attack_eps 0.2 >> /home/maggie/mmat/log/cifar10-attack/cifar10-resnet50-autoattack-20220218.log 2>&1

# #resnet50 cifar autoattack 20220218 
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220218.py run --mode attack --attack_mode autoattack --whitebox --exp_name resnet50-cifar10 --cla_model resnet50 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/resnet50-cifar10/20210908/00002-attackacc-0.7624/train-cifar10-dataset/standard-trained-classifier-resnet50-on-clean-cifar10-finished.pkl --dataset cifar10 --attack_eps 0.3 >> /home/maggie/mmat/log/cifar10-attack/cifar10-resnet50-autoattack-20220218.log 2>&1

# #----------------------vgg19---------------------------------
# #vgg19 cifar autoattack 20220218 
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220218.py run --mode attack --attack_mode autoattack --whitebox --exp_name vgg19-cifar10 --cla_model vgg19 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/vgg19-cifar10/20210908/00002-testacc-0.8705/train-cifar10-dataset/standard-trained-classifier-vgg19-on-clean-cifar10-epoch-0013.pkl --dataset cifar10 --attack_eps 0.02 >> /home/maggie/mmat/log/cifar10-attack/cifar10-vgg19-autoattack-20220218.log 2>&1

# #vgg19 cifar autoattack 20220218 
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220218.py run --mode attack --attack_mode autoattack --whitebox --exp_name vgg19-cifar10 --cla_model vgg19 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/vgg19-cifar10/20210908/00002-testacc-0.8705/train-cifar10-dataset/standard-trained-classifier-vgg19-on-clean-cifar10-epoch-0013.pkl --dataset cifar10 --attack_eps 0.05 >> /home/maggie/mmat/log/cifar10-attack/cifar10-vgg19-autoattack-20220218.log 2>&1

# #vgg19 cifar autoattack 20220218 
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220218.py run --mode attack --attack_mode autoattack --whitebox --exp_name vgg19-cifar10 --cla_model vgg19 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/vgg19-cifar10/20210908/00002-testacc-0.8705/train-cifar10-dataset/standard-trained-classifier-vgg19-on-clean-cifar10-epoch-0013.pkl --dataset cifar10 --attack_eps 0.1 >> /home/maggie/mmat/log/cifar10-attack/cifar10-vgg19-autoattack-20220218.log 2>&1

# #vgg19 cifar autoattack 20220218 
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220218.py run --mode attack --attack_mode autoattack --whitebox --exp_name vgg19-cifar10 --cla_model vgg19 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/vgg19-cifar10/20210908/00002-testacc-0.8705/train-cifar10-dataset/standard-trained-classifier-vgg19-on-clean-cifar10-epoch-0013.pkl --dataset cifar10 --attack_eps 0.2 >> /home/maggie/mmat/log/cifar10-attack/cifar10-vgg19-autoattack-20220218.log 2>&1

# #vgg19 cifar autoattack 20220218 
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220218.py run --mode attack --attack_mode autoattack --whitebox --exp_name vgg19-cifar10 --cla_model vgg19 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/vgg19-cifar10/20210908/00002-testacc-0.8705/train-cifar10-dataset/standard-trained-classifier-vgg19-on-clean-cifar10-epoch-0013.pkl --dataset cifar10 --attack_eps 0.3 >> /home/maggie/mmat/log/cifar10-attack/cifar10-vgg19-autoattack-20220218.log 2>&1

# #----------------------densenet169---------------------------------
# #densenet169 cifar autoattack 20220218 
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220218.py run --mode attack --attack_mode autoattack --whitebox --exp_name densenet169-cifar10 --cla_model densenet169 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/densenet169-cifar10/20210909/00003-testacc-0.7929/train-cifar10-dataset/standard-trained-classifier-densenet169-on-clean-cifar10-epoch-0012.pkl --dataset cifar10 --attack_eps 0.02 >> /home/maggie/mmat/log/cifar10-attack/cifar10-densenet169-autoattack-20220218.log 2>&1

# #densenet169 cifar autoattack 20220218 
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220218.py run --mode attack --attack_mode autoattack --whitebox --exp_name densenet169-cifar10 --cla_model densenet169 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/densenet169-cifar10/20210909/00003-testacc-0.7929/train-cifar10-dataset/standard-trained-classifier-densenet169-on-clean-cifar10-epoch-0012.pkl --dataset cifar10 --attack_eps 0.05 >> /home/maggie/mmat/log/cifar10-attack/cifar10-densenet169-autoattack-20220218.log 2>&1

# #densenet169 cifar autoattack 20220218 
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220218.py run --mode attack --attack_mode autoattack --whitebox --exp_name densenet169-cifar10 --cla_model densenet169 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/densenet169-cifar10/20210909/00003-testacc-0.7929/train-cifar10-dataset/standard-trained-classifier-densenet169-on-clean-cifar10-epoch-0012.pkl --dataset cifar10 --attack_eps 0.1 >> /home/maggie/mmat/log/cifar10-attack/cifar10-densenet169-autoattack-20220218.log 2>&1

# #densenet169 cifar autoattack 20220218 
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220218.py run --mode attack --attack_mode autoattack --whitebox --exp_name densenet169-cifar10 --cla_model densenet169 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/densenet169-cifar10/20210909/00003-testacc-0.7929/train-cifar10-dataset/standard-trained-classifier-densenet169-on-clean-cifar10-epoch-0012.pkl --dataset cifar10 --attack_eps 0.2 >> /home/maggie/mmat/log/cifar10-attack/cifar10-densenet169-autoattack-20220218.log 2>&1

# #densenet169 cifar autoattack 20220218 
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220218.py run --mode attack --attack_mode autoattack --whitebox --exp_name densenet169-cifar10 --cla_model densenet169 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/densenet169-cifar10/20210909/00003-testacc-0.7929/train-cifar10-dataset/standard-trained-classifier-densenet169-on-clean-cifar10-epoch-0012.pkl --dataset cifar10 --attack_eps 0.3 >> /home/maggie/mmat/log/cifar10-attack/cifar10-densenet169-autoattack-20220218.log 2>&1

#----------------------googlenet---------------------------------
#googlenet cifar autoattack 20220218 
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220218.py run --mode attack --attack_mode autoattack --whitebox --exp_name googlenet-cifar10 --cla_model googlenet --cla_network_pkl /home/maggie/mmat/result/train/cla-train/googlenet-cifar10/20210909/00001-testacc-0.8016/train-cifar10-dataset/standard-trained-classifier-googlenet-on-clean-cifar10-epoch-0011.pkl --dataset cifar10 --attack_eps 0.02 >> /home/maggie/mmat/log/cifar10-attack/cifar10-googlenet-autoattack-20220218.log 2>&1

#googlenet cifar autoattack 20220218 
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220218.py run --mode attack --attack_mode autoattack --whitebox --exp_name googlenet-cifar10 --cla_model googlenet --cla_network_pkl /home/maggie/mmat/result/train/cla-train/googlenet-cifar10/20210909/00001-testacc-0.8016/train-cifar10-dataset/standard-trained-classifier-googlenet-on-clean-cifar10-epoch-0011.pkl --dataset cifar10 --attack_eps 0.05 >> /home/maggie/mmat/log/cifar10-attack/cifar10-googlenet-autoattack-20220218.log 2>&1

#googlenet cifar autoattack 20220218 
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220218.py run --mode attack --attack_mode autoattack --whitebox --exp_name googlenet-cifar10 --cla_model googlenet --cla_network_pkl /home/maggie/mmat/result/train/cla-train/googlenet-cifar10/20210909/00001-testacc-0.8016/train-cifar10-dataset/standard-trained-classifier-googlenet-on-clean-cifar10-epoch-0011.pkl --dataset cifar10 --attack_eps 0.1 >> /home/maggie/mmat/log/cifar10-attack/cifar10-googlenet-autoattack-20220218.log 2>&1

#googlenet cifar autoattack 20220218 
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220218.py run --mode attack --attack_mode autoattack --whitebox --exp_name googlenet-cifar10 --cla_model googlenet --cla_network_pkl /home/maggie/mmat/result/train/cla-train/googlenet-cifar10/20210909/00001-testacc-0.8016/train-cifar10-dataset/standard-trained-classifier-googlenet-on-clean-cifar10-epoch-0011.pkl --dataset cifar10 --attack_eps 0.2 >> /home/maggie/mmat/log/cifar10-attack/cifar10-googlenet-autoattack-20220218.log 2>&1

#googlenet cifar autoattack 20220218 
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220218.py run --mode attack --attack_mode autoattack --whitebox --exp_name googlenet-cifar10 --cla_model googlenet --cla_network_pkl /home/maggie/mmat/result/train/cla-train/googlenet-cifar10/20210909/00001-testacc-0.8016/train-cifar10-dataset/standard-trained-classifier-googlenet-on-clean-cifar10-epoch-0011.pkl --dataset cifar10 --attack_eps 0.3 >> /home/maggie/mmat/log/cifar10-attack/cifar10-googlenet-autoattack-20220218.log 2>&1

# #----------------------preactresnet18---------------------------------
# #preactresnet18 cifar autoattack 20220218 
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220218.py run --mode attack --attack_mode autoattack --whitebox --exp_name preactresnet18-cifar10 --cla_model preactresnet18 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/preactresnet18-cifar10/20210924/00001-teatacc-87.37/train-cifar10-dataset/standard-trained-classifier-preactresnet18-on-clean-cifar10-epoch-0011.pkl --dataset cifar10 --attack_eps 0.02 >> /home/maggie/mmat/log/cifar10-attack/cifar10-preactresnet18-autoattack-20220218.log 2>&1

# #preactresnet18 cifar autoattack 20220218 
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220218.py run --mode attack --attack_mode autoattack --whitebox --exp_name preactresnet18-cifar10 --cla_model preactresnet18 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/preactresnet18-cifar10/20210924/00001-teatacc-87.37/train-cifar10-dataset/standard-trained-classifier-preactresnet18-on-clean-cifar10-epoch-0011.pkl --dataset cifar10 --attack_eps 0.05 >> /home/maggie/mmat/log/cifar10-attack/cifar10-preactresnet18-autoattack-20220218.log 2>&1

# #preactresnet18 cifar autoattack 20220218 
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220218.py run --mode attack --attack_mode autoattack --whitebox --exp_name preactresnet18-cifar10 --cla_model preactresnet18 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/preactresnet18-cifar10/20210924/00001-teatacc-87.37/train-cifar10-dataset/standard-trained-classifier-preactresnet18-on-clean-cifar10-epoch-0011.pkl --dataset cifar10 --attack_eps 0.1 >> /home/maggie/mmat/log/cifar10-attack/cifar10-preactresnet18-autoattack-20220218.log 2>&1

# #preactresnet18 cifar autoattack 20220218 
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220218.py run --mode attack --attack_mode autoattack --whitebox --exp_name preactresnet18-cifar10 --cla_model preactresnet18 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/preactresnet18-cifar10/20210924/00001-teatacc-87.37/train-cifar10-dataset/standard-trained-classifier-preactresnet18-on-clean-cifar10-epoch-0011.pkl --dataset cifar10 --attack_eps 0.2 >> /home/maggie/mmat/log/cifar10-attack/cifar10-preactresnet18-autoattack-20220218.log 2>&1

# #preactresnet18 cifar autoattack 20220218 
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220218.py run --mode attack --attack_mode autoattack --whitebox --exp_name preactresnet18-cifar10 --cla_model preactresnet18 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/preactresnet18-cifar10/20210924/00001-teatacc-87.37/train-cifar10-dataset/standard-trained-classifier-preactresnet18-on-clean-cifar10-epoch-0011.pkl --dataset cifar10 --attack_eps 0.3 >> /home/maggie/mmat/log/cifar10-attack/cifar10-preactresnet18-autoattack-20220218.log 2>&1

# #----------------------preactresnet34---------------------------------
# #preactresnet34 cifar autoattack 20220218 
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220218.py run --mode attack --attack_mode autoattack --whitebox --exp_name preactresnet34-cifar10 --cla_model preactresnet34 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/preactresnet34-cifar10/20210927/00002-testacc-83.57/train-cifar10-dataset/standard-trained-classifier-preactresnet34-on-clean-cifar10-epoch-0021.pkl --dataset cifar10 --attack_eps 0.02 >> /home/maggie/mmat/log/cifar10-attack/cifar10-preactresnet34-autoattack-20220218.log 2>&1

# #preactresnet34 cifar autoattack 20220218 
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220218.py run --mode attack --attack_mode autoattack --whitebox --exp_name preactresnet34-cifar10 --cla_model preactresnet34 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/preactresnet34-cifar10/20210927/00002-testacc-83.57/train-cifar10-dataset/standard-trained-classifier-preactresnet34-on-clean-cifar10-epoch-0021.pkl --dataset cifar10 --attack_eps 0.05 >> /home/maggie/mmat/log/cifar10-attack/cifar10-preactresnet34-autoattack-20220218.log 2>&1

# #preactresnet34 cifar autoattack 20220218 
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220218.py run --mode attack --attack_mode autoattack --whitebox --exp_name preactresnet34-cifar10 --cla_model preactresnet34 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/preactresnet34-cifar10/20210927/00002-testacc-83.57/train-cifar10-dataset/standard-trained-classifier-preactresnet34-on-clean-cifar10-epoch-0021.pkl --dataset cifar10 --attack_eps 0.1 >> /home/maggie/mmat/log/cifar10-attack/cifar10-preactresnet34-autoattack-20220218.log 2>&1

# #preactresnet34 cifar autoattack 20220218 
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220218.py run --mode attack --attack_mode autoattack --whitebox --exp_name preactresnet34-cifar10 --cla_model preactresnet34 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/preactresnet34-cifar10/20210927/00002-testacc-83.57/train-cifar10-dataset/standard-trained-classifier-preactresnet34-on-clean-cifar10-epoch-0021.pkl --dataset cifar10 --attack_eps 0.2 >> /home/maggie/mmat/log/cifar10-attack/cifar10-preactresnet34-autoattack-20220218.log 2>&1

# #preactresnet34 cifar autoattack 20220218 
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220218.py run --mode attack --attack_mode autoattack --whitebox --exp_name preactresnet34-cifar10 --cla_model preactresnet34 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/preactresnet34-cifar10/20210927/00002-testacc-83.57/train-cifar10-dataset/standard-trained-classifier-preactresnet34-on-clean-cifar10-epoch-0021.pkl --dataset cifar10 --attack_eps 0.3 >> /home/maggie/mmat/log/cifar10-attack/cifar10-preactresnet34-autoattack-20220218.log 2>&1

# #----------------------preactresnet50---------------------------------
# #preactresnet50 cifar autoattack 20220218 
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220218.py run --mode attack --attack_mode autoattack --whitebox --exp_name preactresnet50-cifar10 --cla_model preactresnet50 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/preactresnet50-cifar10/20210925/00003-testacc-84.74/train-cifar10-dataset/standard-trained-classifier-preactresnet50-on-clean-cifar10-epoch-0021.pkl --dataset cifar10 --attack_eps 0.02 >> /home/maggie/mmat/log/cifar10-attack/cifar10-preactresnet50-autoattack-20220218.log 2>&1

# #preactresnet50 cifar autoattack 20220218 
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220218.py run --mode attack --attack_mode autoattack --whitebox --exp_name preactresnet50-cifar10 --cla_model preactresnet50 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/preactresnet50-cifar10/20210925/00003-testacc-84.74/train-cifar10-dataset/standard-trained-classifier-preactresnet50-on-clean-cifar10-epoch-0021.pkl --dataset cifar10 --attack_eps 0.05 >> /home/maggie/mmat/log/cifar10-attack/cifar10-preactresnet50-autoattack-20220218.log 2>&1

# #preactresnet50 cifar autoattack 20220218 
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220218.py run --mode attack --attack_mode autoattack --whitebox --exp_name preactresnet50-cifar10 --cla_model preactresnet50 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/preactresnet50-cifar10/20210925/00003-testacc-84.74/train-cifar10-dataset/standard-trained-classifier-preactresnet50-on-clean-cifar10-epoch-0021.pkl --dataset cifar10 --attack_eps 0.1 >> /home/maggie/mmat/log/cifar10-attack/cifar10-preactresnet50-autoattack-20220218.log 2>&1

# #preactresnet50 cifar autoattack 20220218 
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220218.py run --mode attack --attack_mode autoattack --whitebox --exp_name preactresnet50-cifar10 --cla_model preactresnet50 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/preactresnet50-cifar10/20210925/00003-testacc-84.74/train-cifar10-dataset/standard-trained-classifier-preactresnet50-on-clean-cifar10-epoch-0021.pkl --dataset cifar10 --attack_eps 0.2 >> /home/maggie/mmat/log/cifar10-attack/cifar10-preactresnet50-autoattack-20220218.log 2>&1

# #preactresnet50 cifar autoattack 20220218 
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220218.py run --mode attack --attack_mode autoattack --whitebox --exp_name preactresnet50-cifar10 --cla_model preactresnet50 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/preactresnet50-cifar10/20210925/00003-testacc-84.74/train-cifar10-dataset/standard-trained-classifier-preactresnet50-on-clean-cifar10-epoch-0021.pkl --dataset cifar10 --attack_eps 0.3 >> /home/maggie/mmat/log/cifar10-attack/cifar10-preactresnet50-autoattack-20220218.log 2>&1