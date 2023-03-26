source ~/.bashrc
source /home/xieyi/anaconda3/bin/activate mmat

# preactresnet34 cifar10 stdtrain  lr = 0.1 batch_size 128 max_epo 250 patience 40
CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20230325.py run --mode train --train_mode cla-train --exp_name preactresnet34-cifar10 --cla_model preactresnet34 --dataset cifar10 --batch_size 128 --epochs 250 --lr 0.1 --optimizer sgd --lr_schedule CosineAnnealingLR --patience 40 >> /home/maggie/mmat/log/CIFAR10/stdtrain/preactresnet34-cifar10-stdtrain-sgd-CosineAnnealingLR-epo250-pat40-20230325.log 2>&1

# preactresnet50 cifar10 stdtrain  lr = 0.1 batch_size 128 max_epo 250 patience 40
CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20230325.py run --mode train --train_mode cla-train --exp_name preactresnet50-cifar10 --cla_model preactresnet50 --dataset cifar10 --batch_size 128 --epochs 250 --lr 0.1 --optimizer sgd --lr_schedule CosineAnnealingLR --patience 40 >> /home/maggie/mmat/log/CIFAR10/stdtrain/preactresnet50-cifar10-stdtrain-sgd-CosineAnnealingLR-epo250-pat40-20230325.log 2>&1

# cusdensenet169 cifar10 stdtrain  lr = 0.1 batch_size 64 max_epo 250 patience 40
CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20230325.py run --mode train --train_mode cla-train --exp_name cusdensenet169-cifar10 --cla_model cusdensenet169 --dataset cifar10 --batch_size 64 --epochs 250 --lr 0.1 --optimizer sgd --lr_schedule CosineAnnealingLR --patience 40 >> /home/maggie/mmat/log/CIFAR10/stdtrain/cusdensenet169-cifar10-stdtrain-sgd-CosineAnnealingLR-epo250-pat40-20230325.log 2>&1




