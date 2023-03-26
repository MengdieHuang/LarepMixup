source ~/.bashrc
source /home/xieyi/anaconda3/bin/activate mmat

# cusresnet34 cifar10 stdtrain  lr = 0.1 batch_size 128 max_epo 250 patience 40
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20230326.py run --mode train --train_mode cla-train --exp_name cusresnet34-cifar10 --cla_model cusresnet34 --dataset cifar10 --batch_size 128 --epochs 250 --lr 0.1 --optimizer sgd --lr_schedule CosineAnnealingLR --patience 40 >> /home/maggie/mmat/log/CIFAR10/stdtrain/cusresnet34-cifar10-stdtrain-sgd-CosineAnnealingLR-epo250-pat40-20230326.log 2>&1




