source ~/.bashrc
source /home/xieyi/anaconda3/bin/activate mmat

# cifar pgd generate 20220624

# #----------------------googlenet---------------------------------
# CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220624.py run --mode attack --attack_mode pgd --whitebox --exp_name googlenet-cifar10 --cla_model googlenet --cla_network_pkl /home/maggie/mmat/result/train/cla-train/googlenet-cifar10/20210909/00001-testacc-0.8016/train-cifar10-dataset/standard-trained-classifier-googlenet-on-clean-cifar10-epoch-0011.pkl --dataset cifar10 --attack_eps 0.02 >> /home/maggie/mmat/log/CIFAR10/attack-example-generate/pgd/googlenet-pgd-generate-20220624.log 2>&1

# CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220624.py run --mode attack --attack_mode pgd --whitebox --exp_name googlenet-cifar10 --cla_model googlenet --cla_network_pkl /home/maggie/mmat/result/train/cla-train/googlenet-cifar10/20210909/00001-testacc-0.8016/train-cifar10-dataset/standard-trained-classifier-googlenet-on-clean-cifar10-epoch-0011.pkl --dataset cifar10 --attack_eps 0.05 >> /home/maggie/mmat/log/CIFAR10/attack-example-generate/pgd/googlenet-pgd-generate-20220624.log 2>&1

# CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220624.py run --mode attack --attack_mode pgd --whitebox --exp_name googlenet-cifar10 --cla_model googlenet --cla_network_pkl /home/maggie/mmat/result/train/cla-train/googlenet-cifar10/20210909/00001-testacc-0.8016/train-cifar10-dataset/standard-trained-classifier-googlenet-on-clean-cifar10-epoch-0011.pkl --dataset cifar10 --attack_eps 0.1 >> /home/maggie/mmat/log/CIFAR10/attack-example-generate/pgd/googlenet-pgd-generate-20220624.log 2>&1

# CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220624.py run --mode attack --attack_mode pgd --whitebox --exp_name googlenet-cifar10 --cla_model googlenet --cla_network_pkl /home/maggie/mmat/result/train/cla-train/googlenet-cifar10/20210909/00001-testacc-0.8016/train-cifar10-dataset/standard-trained-classifier-googlenet-on-clean-cifar10-epoch-0011.pkl --dataset cifar10 --attack_eps 0.2 >> /home/maggie/mmat/log/CIFAR10/attack-example-generate/pgd/googlenet-pgd-generate-20220624.log 2>&1

# CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220624.py run --mode attack --attack_mode pgd --whitebox --exp_name googlenet-cifar10 --cla_model googlenet --cla_network_pkl /home/maggie/mmat/result/train/cla-train/googlenet-cifar10/20210909/00001-testacc-0.8016/train-cifar10-dataset/standard-trained-classifier-googlenet-on-clean-cifar10-epoch-0011.pkl --dataset cifar10 --attack_eps 0.3 >> /home/maggie/mmat/log/CIFAR10/attack-example-generate/pgd/googlenet-pgd-generate-20220624.log 2>&1

# #----------------------densenet169---------------------------------
# CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220624.py run --mode attack --attack_mode pgd --whitebox --exp_name densenet169-cifar10 --cla_model densenet169 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/densenet169-cifar10/20210909/00003-testacc-0.7929/train-cifar10-dataset/standard-trained-classifier-densenet169-on-clean-cifar10-epoch-0012.pkl --dataset cifar10 --attack_eps 0.02 >> /home/maggie/mmat/log/CIFAR10/attack-example-generate/pgd/densenet169-pgd-generate-20220624.log 2>&1

# CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220624.py run --mode attack --attack_mode pgd --whitebox --exp_name densenet169-cifar10 --cla_model densenet169 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/densenet169-cifar10/20210909/00003-testacc-0.7929/train-cifar10-dataset/standard-trained-classifier-densenet169-on-clean-cifar10-epoch-0012.pkl --dataset cifar10 --attack_eps 0.05 >> /home/maggie/mmat/log/CIFAR10/attack-example-generate/pgd/densenet169-pgd-generate-20220624.log 2>&1

# CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220624.py run --mode attack --attack_mode pgd --whitebox --exp_name densenet169-cifar10 --cla_model densenet169 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/densenet169-cifar10/20210909/00003-testacc-0.7929/train-cifar10-dataset/standard-trained-classifier-densenet169-on-clean-cifar10-epoch-0012.pkl --dataset cifar10 --attack_eps 0.1 >> /home/maggie/mmat/log/CIFAR10/attack-example-generate/pgd/densenet169-pgd-generate-20220624.log 2>&1

# CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220624.py run --mode attack --attack_mode pgd --whitebox --exp_name densenet169-cifar10 --cla_model densenet169 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/densenet169-cifar10/20210909/00003-testacc-0.7929/train-cifar10-dataset/standard-trained-classifier-densenet169-on-clean-cifar10-epoch-0012.pkl --dataset cifar10 --attack_eps 0.2 >> /home/maggie/mmat/log/CIFAR10/attack-example-generate/pgd/densenet169-pgd-generate-20220624.log 2>&1

CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220624.py run --mode attack --attack_mode pgd --whitebox --exp_name densenet169-cifar10 --cla_model densenet169 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/densenet169-cifar10/20210909/00003-testacc-0.7929/train-cifar10-dataset/standard-trained-classifier-densenet169-on-clean-cifar10-epoch-0012.pkl --dataset cifar10 --attack_eps 0.3 >> /home/maggie/mmat/log/CIFAR10/attack-example-generate/pgd/densenet169-pgd-generate-20220624.log 2>&1

