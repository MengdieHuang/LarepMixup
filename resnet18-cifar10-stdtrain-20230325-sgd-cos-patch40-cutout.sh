source ~/.bashrc
source /home/xieyi/anaconda3/bin/activate mmat



# # resnet18 cifar10 stdtrain  lr = 0.1 batch_size 128 max_epo 300 patience 40
# CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20230325.py run --mode train --train_mode cla-train --exp_name resnet18-cifar10 --cla_model resnet18 --dataset cifar10 --batch_size 128 --epochs 300 --lr 0.1 --optimizer sgd --lr_schedule CosineAnnealingLR --patience 40 >> /home/maggie/mmat/log/CIFAR10/stdtrain/resnet18-cifar10-stdtrain-sgd-CosineAnnealingLR-epo300-pat40-20230325.log 2>&1

# # resnet18 cifar10 stdtrain  lr = 0.1 batch_size 128 max_epo 250 patience 40
# CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20230325.py run --mode train --train_mode cla-train --exp_name resnet18-cifar10 --cla_model resnet18 --dataset cifar10 --batch_size 128 --epochs 250 --lr 0.1 --optimizer sgd --lr_schedule CosineAnnealingLR --patience 40 >> /home/maggie/mmat/log/CIFAR10/stdtrain/resnet18-cifar10-stdtrain-sgd-CosineAnnealingLR-epo250-pat40-20230325.log 2>&1

# # resnet18 cifar10 stdtrain  lr = 0.1 batch_size 128 max_epo 200 patience 40
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20230325.py run --mode train --train_mode cla-train --exp_name resnet18-cifar10 --cla_model resnet18 --dataset cifar10 --batch_size 128 --epochs 200 --lr 0.1 --optimizer sgd --lr_schedule CosineAnnealingLR --patience 40 >> /home/maggie/mmat/log/CIFAR10/stdtrain/resnet18-cifar10-stdtrain-sgd-CosineAnnealingLR-epo200-pat40-20230325.log 2>&1

# # resnet18 cifar10 stdtrain  lr = 0.1 batch_size 128 max_epo 200 patience 40
# CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20230325.py run --mode train --train_mode cla-train --exp_name resnet18-cifar10 --cla_model resnet18 --dataset cifar10 --batch_size 128 --epochs 200 --lr 0.1 --optimizer sgd --lr_schedule CosineAnnealingLR --patience 40 >> /home/maggie/mmat/log/CIFAR10/stdtrain/resnet18-cifar10-stdtrain-sgd-CosineAnnealingLR-epo200-pat40-20230325.log 2>&1

# # resnet18 cifar10 stdtrain  lr = 0.1 batch_size 256 max_epo 200 patience 40
# CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20230325.py run --mode train --train_mode cla-train --exp_name resnet18-cifar10 --cla_model resnet18 --dataset cifar10 --batch_size 256 --epochs 200 --lr 0.1 --optimizer sgd --lr_schedule CosineAnnealingLR --patience 40 >> /home/maggie/mmat/log/CIFAR10/stdtrain/resnet18-cifar10-stdtrain-sgd-CosineAnnealingLR-epo200-pat40-bat256-20230325.log 2>&1

# # resnet18 cifar10 stdtrain  lr = 0.1 batch_size 128 max_epo 200 patience 40 # VerticalFlip
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20230325.py run --mode train --train_mode cla-train --exp_name resnet18-cifar10 --cla_model resnet18 --dataset cifar10 --batch_size 128 --epochs 200 --lr 0.1 --optimizer sgd --lr_schedule CosineAnnealingLR --patience 40 >> /home/maggie/mmat/log/CIFAR10/stdtrain/resnet18-cifar10-stdtrain-sgd-CosineAnnealingLR-epo200-pat40-VerticalFlip-20230325.log 2>&1


# resnet18 cifar10 stdtrain  lr = 0.1 batch_size 128 max_epo 250 patience 40
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20230325.py run --mode train --train_mode cla-train --exp_name resnet18-cifar10 --cla_model resnet18 --dataset cifar10 --batch_size 128 --epochs 250 --lr 0.1 --optimizer sgd --lr_schedule CosineAnnealingLR --patience 40 >> /home/maggie/mmat/log/CIFAR10/stdtrain/resnet18-cifar10-stdtrain-sgd-CosineAnnealingLR-epo250-pat40-20230325.log 2>&1

# resnet18 cifar10 stdtrain  lr = 0.1 batch_size 128 max_epo 250 patience 40
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20230325.py run --mode train --train_mode cla-train --exp_name resnet18-cifar10 --cla_model resnet18 --dataset cifar10 --batch_size 128 --epochs 250 --lr 0.1 --optimizer sgd --lr_schedule CosineAnnealingLR --patience 40 >> /home/maggie/mmat/log/CIFAR10/stdtrain/resnet18-cifar10-stdtrain-sgd-CosineAnnealingLR-epo250-pat40-20230325.log 2>&1



