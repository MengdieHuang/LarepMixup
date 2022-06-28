source ~/.bashrc
source /home/xieyi/anaconda3/bin/activate mmat

#----------------preactresnet18----------------------
# preactresnet18 svhn stdftrain  lr = 0.01
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220627.py run --mode train --train_mode cla-train --exp_name preactresnet18-svhn --cla_model preactresnet18 --dataset svhn --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/SVHN/classsifier-standard-train/preactresnet18-svhn-stdtrain-20220627.log 2>&1

# preactresnet18 svhn stdftrain  lr = 0.001
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220627.py run --mode train --train_mode cla-train --exp_name preactresnet18-svhn --cla_model preactresnet18 --dataset svhn --batch_size 256 --epochs 40 --lr 0.001 >> /home/maggie/mmat/log/SVHN/classsifier-standard-train/preactresnet18-svhn-stdtrain-20220627.log 2>&1

# preactresnet18 svhn stdftrain  lr = 0.01
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220627.py run --mode train --train_mode cla-train --exp_name preactresnet18-svhn --cla_model preactresnet18 --dataset svhn --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/SVHN/classsifier-standard-train/preactresnet18-svhn-stdtrain-20220627.log 2>&1

# preactresnet18 svhn stdftrain  lr = 0.001
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220627.py run --mode train --train_mode cla-train --exp_name preactresnet18-svhn --cla_model preactresnet18 --dataset svhn --batch_size 256 --epochs 40 --lr 0.001 >> /home/maggie/mmat/log/SVHN/classsifier-standard-train/preactresnet18-svhn-stdtrain-20220627.log 2>&1

#----------------preactresnet34----------------------
# preactresnet34 svhn stdftrain  lr = 0.01
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220627.py run --mode train --train_mode cla-train --exp_name preactresnet34-svhn --cla_model preactresnet34 --dataset svhn --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/SVHN/classsifier-standard-train/preactresnet34-svhn-stdtrain-20220627.log 2>&1

# preactresnet34 svhn stdftrain  lr = 0.001
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220627.py run --mode train --train_mode cla-train --exp_name preactresnet34-svhn --cla_model preactresnet34 --dataset svhn --batch_size 256 --epochs 40 --lr 0.001 >> /home/maggie/mmat/log/SVHN/classsifier-standard-train/preactresnet34-svhn-stdtrain-20220627.log 2>&1

# preactresnet34 svhn stdftrain  lr = 0.01
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220627.py run --mode train --train_mode cla-train --exp_name preactresnet34-svhn --cla_model preactresnet34 --dataset svhn --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/SVHN/classsifier-standard-train/preactresnet34-svhn-stdtrain-20220627.log 2>&1

# preactresnet34 svhn stdftrain  lr = 0.001
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220627.py run --mode train --train_mode cla-train --exp_name preactresnet34-svhn --cla_model preactresnet34 --dataset svhn --batch_size 256 --epochs 40 --lr 0.001 >> /home/maggie/mmat/log/SVHN/classsifier-standard-train/preactresnet34-svhn-stdtrain-20220627.log 2>&1

#----------------preactresnet50----------------------
# preactresnet50 svhn stdftrain  lr = 0.01
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220627.py run --mode train --train_mode cla-train --exp_name preactresnet50-svhn --cla_model preactresnet50 --dataset svhn --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/SVHN/classsifier-standard-train/preactresnet50-svhn-stdtrain-20220627.log 2>&1

# preactresnet50 svhn stdftrain  lr = 0.001
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220627.py run --mode train --train_mode cla-train --exp_name preactresnet50-svhn --cla_model preactresnet50 --dataset svhn --batch_size 256 --epochs 40 --lr 0.001 >> /home/maggie/mmat/log/SVHN/classsifier-standard-train/preactresnet50-svhn-stdtrain-20220627.log 2>&1

# preactresnet50 svhn stdftrain  lr = 0.01
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220627.py run --mode train --train_mode cla-train --exp_name preactresnet50-svhn --cla_model preactresnet50 --dataset svhn --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/SVHN/classsifier-standard-train/preactresnet50-svhn-stdtrain-20220627.log 2>&1

# preactresnet50 svhn stdftrain  lr = 0.001
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220627.py run --mode train --train_mode cla-train --exp_name preactresnet50-svhn --cla_model preactresnet50 --dataset svhn --batch_size 256 --epochs 40 --lr 0.001 >> /home/maggie/mmat/log/SVHN/classsifier-standard-train/preactresnet50-svhn-stdtrain-20220627.log 2>&1