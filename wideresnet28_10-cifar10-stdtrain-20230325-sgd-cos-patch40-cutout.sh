source ~/.bashrc
source /home/xieyi/anaconda3/bin/activate mmat

# wideresnet28_10 cifar10 stdtrain  lr = 0.1 batch_size 128 max_epo 250 patience 40
CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20230325.py run --mode train --train_mode cla-train --exp_name wideresnet28_10-cifar10 --cla_model wideresnet28_10 --dataset cifar10 --batch_size 128 --epochs 250 --lr 0.1 --optimizer sgd --lr_schedule CosineAnnealingLR --patience 40 >> /home/maggie/mmat/log/CIFAR10/stdtrain/wideresnet28_10-cifar10-stdtrain-sgd-CosineAnnealingLR-epo250-pat40-20230325.log 2>&1



