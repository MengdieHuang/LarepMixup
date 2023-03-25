source ~/.bashrc
source /home/xieyi/anaconda3/bin/activate mmat

# resnet18 cifar10 stdtrain  lr = 0.1 batch_size 256 max_epo 200
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20230325.py run --mode train --train_mode cla-train --exp_name resnet18-cifar10 --cla_model resnet18 --dataset cifar10 --batch_size 256 --epochs 200 --lr 0.1 --optimizer adam --lr_schedule CosineAnnealingLR >> /home/maggie/mmat/log/CIFAR10/stdtrain/resnet18-cifar10-stdtrain-adam-CosineAnnealingLR-20230325.log 2>&1

# resnet18 cifar10 stdtrain  lr = 0.01 batch_size 256 max_epo 200
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20230325.py run --mode train --train_mode cla-train --exp_name resnet18-cifar10 --cla_model resnet18 --dataset cifar10 --batch_size 256 --epochs 200 --lr 0.01 --optimizer adam --lr_schedule CosineAnnealingLR >> /home/maggie/mmat/log/CIFAR10/stdtrain/resnet18-cifar10-stdtrain-adam-CosineAnnealingLR-20230325.log 2>&1

# resnet18 cifar10 stdtrain  lr = 0.1 batch_size 256 max_epo 200
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20230325.py run --mode train --train_mode cla-train --exp_name resnet18-cifar10 --cla_model resnet18 --dataset cifar10 --batch_size 256 --epochs 200 --lr 0.1 --optimizer adam --lr_schedule StepLR >> /home/maggie/mmat/log/CIFAR10/stdtrain/resnet18-cifar10-stdtrain-adam-StepLR-20230325.log 2>&1

# resnet18 cifar10 stdtrain  lr = 0.01 batch_size 256 max_epo 200
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20230325.py run --mode train --train_mode cla-train --exp_name resnet18-cifar10 --cla_model resnet18 --dataset cifar10 --batch_size 256 --epochs 200 --lr 0.01 --optimizer adam --lr_schedule StepLR >> /home/maggie/mmat/log/CIFAR10/stdtrain/resnet18-cifar10-stdtrain-adam-StepLR-20230325.log 2>&1
