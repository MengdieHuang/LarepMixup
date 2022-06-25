source ~/.bashrc
source /home/xieyi/anaconda3/bin/activate mmat

# cifar pgd generate 20220624
#----------------------vgg19---------------------------------
CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220624.py run --mode attack --attack_mode pgd --whitebox --exp_name vgg19-cifar10 --cla_model vgg19 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/vgg19-cifar10/20210908/00002-testacc-0.8705/train-cifar10-dataset/standard-trained-classifier-vgg19-on-clean-cifar10-epoch-0013.pkl --dataset cifar10 --attack_eps 0.02 >> /home/maggie/mmat/log/CIFAR10/attack-example-generate/pgd/vgg19-pgd-generate-20220624.log 2>&1

CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220624.py run --mode attack --attack_mode pgd --whitebox --exp_name vgg19-cifar10 --cla_model vgg19 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/vgg19-cifar10/20210908/00002-testacc-0.8705/train-cifar10-dataset/standard-trained-classifier-vgg19-on-clean-cifar10-epoch-0013.pkl --dataset cifar10 --attack_eps 0.05 >> /home/maggie/mmat/log/CIFAR10/attack-example-generate/pgd/vgg19-pgd-generate-20220624.log 2>&1

CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220624.py run --mode attack --attack_mode pgd --whitebox --exp_name vgg19-cifar10 --cla_model vgg19 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/vgg19-cifar10/20210908/00002-testacc-0.8705/train-cifar10-dataset/standard-trained-classifier-vgg19-on-clean-cifar10-epoch-0013.pkl --dataset cifar10 --attack_eps 0.1 >> /home/maggie/mmat/log/CIFAR10/attack-example-generate/pgd/vgg19-pgd-generate-20220624.log 2>&1

CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220624.py run --mode attack --attack_mode pgd --whitebox --exp_name vgg19-cifar10 --cla_model vgg19 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/vgg19-cifar10/20210908/00002-testacc-0.8705/train-cifar10-dataset/standard-trained-classifier-vgg19-on-clean-cifar10-epoch-0013.pkl --dataset cifar10 --attack_eps 0.2 >> /home/maggie/mmat/log/CIFAR10/attack-example-generate/pgd/vgg19-pgd-generate-20220624.log 2>&1

CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220624.py run --mode attack --attack_mode pgd --whitebox --exp_name vgg19-cifar10 --cla_model vgg19 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/vgg19-cifar10/20210908/00002-testacc-0.8705/train-cifar10-dataset/standard-trained-classifier-vgg19-on-clean-cifar10-epoch-0013.pkl --dataset cifar10 --attack_eps 0.3 >> /home/maggie/mmat/log/CIFAR10/attack-example-generate/pgd/vgg19-pgd-generate-20220624.log 2>&1