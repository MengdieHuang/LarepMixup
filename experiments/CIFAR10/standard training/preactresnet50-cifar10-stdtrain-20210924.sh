source ~/.bashrc
source /home/xieyi/anaconda3/bin/activate mmat

# preactresnet50 cifar10 stdftrain 20210924000  lr = 0.01
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20210924.py run --mode train --train_mode cla-train --exp_name preactresnet50-cifar10 --cla_model preactresnet50 --dataset cifar10 --batch_size 256 --epochs 20 --lr 0.01 >> /home/maggie/mmat/log/preactresnet50-cifar10-stdtrain/preactresnet5--cifar10-stdtrain-20210924.log 2>&1

# preactresnet50 cifar10 stdftrain 20210924000  lr = 0.001
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20210924.py run --mode train --train_mode cla-train --exp_name preactresnet50-cifar10 --cla_model preactresnet50 --dataset cifar10 --batch_size 256 --epochs 20 --lr 0.001 >> /home/maggie/mmat/log/preactresnet50-cifar10-stdtrain/preactresnet5--cifar10-stdtrain-20210924.log 2>&1

# preactresnet50 cifar10 stdftrain 20210924000  lr = 0.01
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20210924.py run --mode train --train_mode cla-train --exp_name preactresnet50-cifar10 --cla_model preactresnet50 --dataset cifar10 --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/preactresnet50-cifar10-stdtrain/preactresnet5--cifar10-stdtrain-20210924.log 2>&1

# preactresnet50 cifar10 stdftrain 20210924000  lr = 0.001
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20210924.py run --mode train --train_mode cla-train --exp_name preactresnet50-cifar10 --cla_model preactresnet50 --dataset cifar10 --batch_size 256 --epochs 40 --lr 0.001 >> /home/maggie/mmat/log/preactresnet50-cifar10-stdtrain/preactresnet5--cifar10-stdtrain-20210924.log 2>&1