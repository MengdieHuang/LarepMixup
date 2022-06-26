source ~/.bashrc
source /home/xieyi/anaconda3/bin/activate mmat

# cifar pgd generate 20220624
# #----------------------alexnet---------------------------------
# CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220624.py run --mode attack --attack_mode pgd --whitebox --exp_name alexnet-cifar10 --cla_model alexnet --cla_network_pkl /home/maggie/mmat/result/train/cla-train/alexnet-cifar10/20210908/00015-testacc-0.8355/train-cifar10-dataset/standard-trained-classifier-alexnet-on-clean-cifar10-finished.pkl --dataset cifar10 --attack_eps 0.02 >> /home/maggie/mmat/log/CIFAR10/attack-example-generate/pgd/alexnet-pgd-generate-20220624.log 2>&1

# CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220624.py run --mode attack --attack_mode pgd --whitebox --exp_name alexnet-cifar10 --cla_model alexnet --cla_network_pkl /home/maggie/mmat/result/train/cla-train/alexnet-cifar10/20210908/00015-testacc-0.8355/train-cifar10-dataset/standard-trained-classifier-alexnet-on-clean-cifar10-finished.pkl --dataset cifar10 --attack_eps 0.05 >> /home/maggie/mmat/log/CIFAR10/attack-example-generate/pgd/alexnet-pgd-generate-20220624.log 2>&1

# CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220624.py run --mode attack --attack_mode pgd --whitebox --exp_name alexnet-cifar10 --cla_model alexnet --cla_network_pkl /home/maggie/mmat/result/train/cla-train/alexnet-cifar10/20210908/00015-testacc-0.8355/train-cifar10-dataset/standard-trained-classifier-alexnet-on-clean-cifar10-finished.pkl --dataset cifar10 --attack_eps 0.1 >> /home/maggie/mmat/log/CIFAR10/attack-example-generate/pgd/alexnet-pgd-generate-20220624.log 2>&1

# CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220624.py run --mode attack --attack_mode pgd --whitebox --exp_name alexnet-cifar10 --cla_model alexnet --cla_network_pkl /home/maggie/mmat/result/train/cla-train/alexnet-cifar10/20210908/00015-testacc-0.8355/train-cifar10-dataset/standard-trained-classifier-alexnet-on-clean-cifar10-finished.pkl --dataset cifar10 --attack_eps 0.2 >> /home/maggie/mmat/log/CIFAR10/attack-example-generate/pgd/alexnet-pgd-generate-20220624.log 2>&1

# CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220624.py run --mode attack --attack_mode pgd --whitebox --exp_name alexnet-cifar10 --cla_model alexnet --cla_network_pkl /home/maggie/mmat/result/train/cla-train/alexnet-cifar10/20210908/00015-testacc-0.8355/train-cifar10-dataset/standard-trained-classifier-alexnet-on-clean-cifar10-finished.pkl --dataset cifar10 --attack_eps 0.3 >> /home/maggie/mmat/log/CIFAR10/attack-example-generate/pgd/alexnet-pgd-generate-20220624.log 2>&1


# #----------------------resnet18---------------------------------
# CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220624.py run --mode attack --attack_mode pgd --whitebox --exp_name resnet18-cifar10 --cla_model resnet18 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/resnet18-cifar10/20210908/00017-testacc-0.7779/train-cifar10-dataset/standard-trained-classifier-resnet18-on-clean-cifar10-finished.pkl --dataset cifar10 --attack_eps 0.02 >> /home/maggie/mmat/log/CIFAR10/attack-example-generate/pgd/resnet18-pgd-generate-20220624.log 2>&1

# CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220624.py run --mode attack --attack_mode pgd --whitebox --exp_name resnet18-cifar10 --cla_model resnet18 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/resnet18-cifar10/20210908/00017-testacc-0.7779/train-cifar10-dataset/standard-trained-classifier-resnet18-on-clean-cifar10-finished.pkl --dataset cifar10 --attack_eps 0.05 >> /home/maggie/mmat/log/CIFAR10/attack-example-generate/pgd/resnet18-pgd-generate-20220624.log 2>&1

# CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220624.py run --mode attack --attack_mode pgd --whitebox --exp_name resnet18-cifar10 --cla_model resnet18 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/resnet18-cifar10/20210908/00017-testacc-0.7779/train-cifar10-dataset/standard-trained-classifier-resnet18-on-clean-cifar10-finished.pkl --dataset cifar10 --attack_eps 0.1 >> /home/maggie/mmat/log/CIFAR10/attack-example-generate/pgd/resnet18-pgd-generate-20220624.log 2>&1

# CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220624.py run --mode attack --attack_mode pgd --whitebox --exp_name resnet18-cifar10 --cla_model resnet18 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/resnet18-cifar10/20210908/00017-testacc-0.7779/train-cifar10-dataset/standard-trained-classifier-resnet18-on-clean-cifar10-finished.pkl --dataset cifar10 --attack_eps 0.2 >> /home/maggie/mmat/log/CIFAR10/attack-example-generate/pgd/resnet18-pgd-generate-20220624.log 2>&1

# CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220624.py run --mode attack --attack_mode pgd --whitebox --exp_name resnet18-cifar10 --cla_model resnet18 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/resnet18-cifar10/20210908/00017-testacc-0.7779/train-cifar10-dataset/standard-trained-classifier-resnet18-on-clean-cifar10-finished.pkl --dataset cifar10 --attack_eps 0.3 >> /home/maggie/mmat/log/CIFAR10/attack-example-generate/pgd/resnet18-pgd-generate-20220624.log 2>&1


# #----------------------resnet34---------------------------------
# CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220624.py run --mode attack --attack_mode pgd --whitebox --exp_name resnet34-cifar10 --cla_model resnet34 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/resnet34-cifar10/20210907/00030-testacc-0.7739/train-cifar10-dataset/standard-trained-classifier-resnet34-on-clean-cifar10-finished.pkl --dataset cifar10 --attack_eps 0.02 >> /home/maggie/mmat/log/CIFAR10/attack-example-generate/pgd/resnet34-pgd-generate-20220624.log 2>&1

# CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220624.py run --mode attack --attack_mode pgd --whitebox --exp_name resnet34-cifar10 --cla_model resnet34 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/resnet34-cifar10/20210907/00030-testacc-0.7739/train-cifar10-dataset/standard-trained-classifier-resnet34-on-clean-cifar10-finished.pkl --dataset cifar10 --attack_eps 0.05 >> /home/maggie/mmat/log/CIFAR10/attack-example-generate/pgd/resnet34-pgd-generate-20220624.log 2>&1

# CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220624.py run --mode attack --attack_mode pgd --whitebox --exp_name resnet34-cifar10 --cla_model resnet34 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/resnet34-cifar10/20210907/00030-testacc-0.7739/train-cifar10-dataset/standard-trained-classifier-resnet34-on-clean-cifar10-finished.pkl --dataset cifar10 --attack_eps 0.1 >> /home/maggie/mmat/log/CIFAR10/attack-example-generate/pgd/resnet34-pgd-generate-20220624.log 2>&1

# CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220624.py run --mode attack --attack_mode pgd --whitebox --exp_name resnet34-cifar10 --cla_model resnet34 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/resnet34-cifar10/20210907/00030-testacc-0.7739/train-cifar10-dataset/standard-trained-classifier-resnet34-on-clean-cifar10-finished.pkl --dataset cifar10 --attack_eps 0.2 >> /home/maggie/mmat/log/CIFAR10/attack-example-generate/pgd/resnet34-pgd-generate-20220624.log 2>&1

# CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220624.py run --mode attack --attack_mode pgd --whitebox --exp_name resnet34-cifar10 --cla_model resnet34 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/resnet34-cifar10/20210907/00030-testacc-0.7739/train-cifar10-dataset/standard-trained-classifier-resnet34-on-clean-cifar10-finished.pkl --dataset cifar10 --attack_eps 0.3 >> /home/maggie/mmat/log/CIFAR10/attack-example-generate/pgd/resnet34-pgd-generate-20220624.log 2>&1


# #----------------------resnet50---------------------------------
# CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220624.py run --mode attack --attack_mode pgd --whitebox --exp_name resnet50-cifar10 --cla_model resnet50 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/resnet50-cifar10/20210908/00002-attackacc-0.7624/train-cifar10-dataset/standard-trained-classifier-resnet50-on-clean-cifar10-finished.pkl --dataset cifar10 --attack_eps 0.02 >> /home/maggie/mmat/log/CIFAR10/attack-example-generate/pgd/resnet50-pgd-generate-20220624.log 2>&1

# CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220624.py run --mode attack --attack_mode pgd --whitebox --exp_name resnet50-cifar10 --cla_model resnet50 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/resnet50-cifar10/20210908/00002-attackacc-0.7624/train-cifar10-dataset/standard-trained-classifier-resnet50-on-clean-cifar10-finished.pkl --dataset cifar10 --attack_eps 0.05 >> /home/maggie/mmat/log/CIFAR10/attack-example-generate/pgd/resnet50-pgd-generate-20220624.log 2>&1

# CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220624.py run --mode attack --attack_mode pgd --whitebox --exp_name resnet50-cifar10 --cla_model resnet50 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/resnet50-cifar10/20210908/00002-attackacc-0.7624/train-cifar10-dataset/standard-trained-classifier-resnet50-on-clean-cifar10-finished.pkl --dataset cifar10 --attack_eps 0.1 >> /home/maggie/mmat/log/CIFAR10/attack-example-generate/pgd/resnet50-pgd-generate-20220624.log 2>&1

# CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220624.py run --mode attack --attack_mode pgd --whitebox --exp_name resnet50-cifar10 --cla_model resnet50 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/resnet50-cifar10/20210908/00002-attackacc-0.7624/train-cifar10-dataset/standard-trained-classifier-resnet50-on-clean-cifar10-finished.pkl --dataset cifar10 --attack_eps 0.2 >> /home/maggie/mmat/log/CIFAR10/attack-example-generate/pgd/resnet50-pgd-generate-20220624.log 2>&1

CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220624.py run --mode attack --attack_mode pgd --whitebox --exp_name resnet50-cifar10 --cla_model resnet50 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/resnet50-cifar10/20210908/00002-attackacc-0.7624/train-cifar10-dataset/standard-trained-classifier-resnet50-on-clean-cifar10-finished.pkl --dataset cifar10 --attack_eps 0.3 >> /home/maggie/mmat/log/CIFAR10/attack-example-generate/pgd/resnet50-pgd-generate-20220624.log 2>&1
