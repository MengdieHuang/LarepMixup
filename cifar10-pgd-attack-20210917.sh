source ~/.bashrc
source /home/xieyi/anaconda3/bin/activate mmat

# # alexnet
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20210917.py run --mode attack --attack_mode pgd --exp_name alexnet-cifar10 --cla_model alexnet --cla_network_pkl /home/maggie/mmat/result/train/cla-train/alexnet-cifar10/20210908/00015-testacc-0.8355/train-cifar10-dataset/standard-trained-classifier-alexnet-on-clean-cifar10-finished.pkl --dataset cifar10 --attack_eps 0.3 >> /home/maggie/mmat/log/cifar10-attack/cifar10-pgd-attack-20210917.log 2>&1

# resnet-18
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20210917.py run --mode attack --attack_mode pgd --exp_name resnet18-cifar10 --cla_model resnet18 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/resnet18-cifar10/20210908/00017-testacc-0.7779/train-cifar10-dataset/standard-trained-classifier-resnet18-on-clean-cifar10-finished.pkl --dataset cifar10 --attack_eps 0.3 >> /home/maggie/mmat/log/cifar10-attack/cifar10-pgd-attack-20210917.log 2>&1

# resnet-34
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20210917.py run --mode attack --attack_mode pgd --exp_name resnet34-cifar10 --cla_model resnet34 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/resnet34-cifar10/20210907/00030-testscc-0.7739/train-cifar10-dataset/standard-trained-classifier-resnet34-on-clean-cifar10-finished.pkl --dataset cifar10 --attack_eps 0.3 >> /home/maggie/mmat/log/cifar10-attack/cifar10-pgd-attack-20210917.log 2>&1

# resnet-50
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20210917.py run --mode attack --attack_mode pgd --exp_name resnet50-cifar10 --cla_model resnet50 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/resnet50-cifar10/20210908/00001-testacc-0.7666/train-cifar10-dataset/standard-trained-classifier-resnet50-on-clean-cifar10-epoch-0011.pkl --dataset cifar10 --attack_eps 0.3 >> /home/maggie/mmat/log/cifar10-attack/cifar10-pgd-attack-20210917.log 2>&1

# vgg-19
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20210917.py run --mode attack --attack_mode pgd --exp_name vgg19-cifar10 --cla_model vgg19 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/vgg19-cifar10/20210908/00002-testacc-0.8705/train-cifar10-dataset/standard-trained-classifier-vgg19-on-clean-cifar10-epoch-0013.pkl --dataset cifar10 --attack_eps 0.3 >> /home/maggie/mmat/log/cifar10-attack/cifar10-pgd-attack-20210917.log 2>&1

# densenet-169
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20210917.py run --mode attack --attack_mode pgd --exp_name densenet169-cifar10 --cla_model densenet169 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/densenet169-cifar10/20210909/00003-testacc-0.7929/train-cifar10-dataset/standard-trained-classifier-densenet169-on-clean-cifar10-epoch-0012.pkl --dataset cifar10 --attack_eps 0.3 >> /home/maggie/mmat/log/cifar10-attack/cifar10-pgd-attack-20210917.log 2>&1

# googlenet
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20210917.py run --mode attack --attack_mode pgd --exp_name googlenet-cifar10 --cla_model googlenet --cla_network_pkl /home/maggie/mmat/result/train/cla-train/googlenet-cifar10/20210909/00001-testacc-0.8016/train-cifar10-dataset/standard-trained-classifier-googlenet-on-clean-cifar10-epoch-0011.pkl --dataset cifar10 --attack_eps 0.3 >> /home/maggie/mmat/log/cifar10-attack/cifar10-pgd-attack-20210917.log 2>&1