source ~/.bashrc
source /home/xieyi/anaconda3/bin/activate mmat

# # resnet18 cifar10 stdtrain  lr = 0.1 batch_size 256 max_epo 200
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20230325.py run --mode train --train_mode cla-train --exp_name resnet18-cifar10 --cla_model resnet18 --dataset cifar10 --batch_size 256 --epochs 200 --lr 0.1 --optimizer sgd --lr_schedule StepLR >> /home/maggie/mmat/log/CIFAR10/stdtrain/resnet18-cifar10-stdtrain-sgd-StepLR-20230325.log 2>&1

# # resnet18 cifar10 stdtrain  lr = 0.01 batch_size 256 max_epo 200
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20230325.py run --mode train --train_mode cla-train --exp_name resnet18-cifar10 --cla_model resnet18 --dataset cifar10 --batch_size 256 --epochs 200 --lr 0.01 --optimizer sgd --lr_schedule StepLR >> /home/maggie/mmat/log/CIFAR10/stdtrain/resnet18-cifar10-stdtrain-sgd-StepLR-20230325.log 2>&1

# # resnet18 cifar10 stdtrain  lr = 0.1 batch_size 256 max_epo 200
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20230325.py run --mode train --train_mode cla-train --exp_name resnet18-cifar10 --cla_model resnet18 --dataset cifar10 --batch_size 256 --epochs 200 --lr 0.1 --optimizer sgd --lr_schedule CosineAnnealingLR >> /home/maggie/mmat/log/CIFAR10/stdtrain/resnet18-cifar10-stdtrain-sgd-CosineAnnealingLR-20230325.log 2>&1

# # resnet18 cifar10 stdtrain  lr = 0.1 batch_size 128 max_epo 200 patience 60
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20230325.py run --mode train --train_mode cla-train --exp_name resnet18-cifar10 --cla_model resnet18 --dataset cifar10 --batch_size 128 --epochs 200 --lr 0.1 --optimizer sgd --lr_schedule CosineAnnealingLR --patience 60 >> /home/maggie/mmat/log/CIFAR10/stdtrain/resnet18-cifar10-stdtrain-sgd-CosineAnnealingLR-20230325.log 2>&1

# resnet18 cifar10 stdtrain  lr = 0.1 batch_size 128 max_epo 400 patience 60
CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20230325.py run --mode train --train_mode cla-train --exp_name resnet18-cifar10 --cla_model resnet18 --dataset cifar10 --batch_size 128 --epochs 400 --lr 0.1 --optimizer sgd --lr_schedule CosineAnnealingLR --patience 60 >> /home/maggie/mmat/log/CIFAR10/stdtrain/resnet18-cifar10-stdtrain-sgd-CosineAnnealingLR-20230325.log 2>&1

# resnet18 cifar10 stdtrain  lr = 0.1 batch_size 128 max_epo 400 patience 60
CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20230325.py run --mode train --train_mode cla-train --exp_name resnet18-cifar10 --cla_model resnet18 --dataset cifar10 --batch_size 128 --epochs 400 --lr 0.1 --optimizer sgd --lr_schedule CosineAnnealingLR --patience 60 >> /home/maggie/mmat/log/CIFAR10/stdtrain/resnet18-cifar10-stdtrain-sgd-CosineAnnealingLR-epo40-20230325.log 2>&1

# # resnet18 cifar10 stdtrain  lr = 0.01 batch_size 256 max_epo 200
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20230325.py run --mode train --train_mode cla-train --exp_name resnet18-cifar10 --cla_model resnet18 --dataset cifar10 --batch_size 256 --epochs 200 --lr 0.01 --optimizer sgd --lr_schedule CosineAnnealingLR >> /home/maggie/mmat/log/CIFAR10/stdtrain/resnet18-cifar10-stdtrain-sgd-CosineAnnealingLR-20230325.log 2>&1



# # resnet18 cifar10 stdtrain  lr = 0.1 batch_size 512 max_epo 200
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20230325.py run --mode train --train_mode cla-train --exp_name resnet18-cifar10 --cla_model resnet18 --dataset cifar10 --batch_size 512 --epochs 200 --lr 0.1 --optimizer sgd --lr_schedule CosineAnnealingLR >> /home/maggie/mmat/log/CIFAR10/stdtrain/resnet18-cifar10-stdtrain-sgd-CosineAnnealingLR-20230325.log 2>&1

# # resnet18 cifar10 stdtrain  lr = 0.01 batch_size 512 max_epo 200
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20230325.py run --mode train --train_mode cla-train --exp_name resnet18-cifar10 --cla_model resnet18 --dataset cifar10 --batch_size 512 --epochs 200 --lr 0.01 --optimizer sgd --lr_schedule CosineAnnealingLR >> /home/maggie/mmat/log/CIFAR10/stdtrain/resnet18-cifar10-stdtrain-sgd-CosineAnnealingLR-20230325.log 2>&1


