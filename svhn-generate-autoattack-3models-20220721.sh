source ~/.bashrc
source /home/xieyi/anaconda3/bin/activate mmat

# # generate svhn autoattack adversarial examples
# #----------------------preactresnet18---------------------------------
# CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220720.py run --mode attack --attack_mode autoattack  --whitebox --exp_name preactresnet18-svhn --cla_model preactresnet18 --cla_network_pkl /home/data/maggie/result-newhome/train/cla-train/preactresnet18-svhn/20220627/00003-testacc-95.97/train-svhn-dataset/standard-trained-classifier-preactresnet18-on-clean-svhn-epoch-0011.pkl --dataset svhn --attack_eps 0.1 >> /home/maggie/mmat/log/SVHN/attack-example-generate/autoattack/svhn-preactresnet18-autoattack-generate-20220721.log 2>&1

# #----------------------preactresnet34---------------------------------
# CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220720.py run --mode attack --attack_mode autoattack  --whitebox --exp_name preactresnet34-svhn --cla_model preactresnet34 --cla_network_pkl /home/data/maggie/result-newhome/train/cla-train/preactresnet34-svhn/20220627/00003-testacc-95.75/train-svhn-dataset/standard-trained-classifier-preactresnet34-on-clean-svhn-epoch-0013.pkl --dataset svhn --attack_eps 0.1 >> /home/maggie/mmat/log/SVHN/attack-example-generate/autoattack/svhn-preactresnet34-autoattack-generate-20220721.log 2>&1

# #----------------------preactresnet50---------------------------------
# CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220720.py run --mode attack --attack_mode autoattack  --whitebox --exp_name preactresnet50-svhn --cla_model preactresnet50 --cla_network_pkl /home/data/maggie/result-newhome/train/cla-train/preactresnet50-svhn/20220627/00001-testacc-95.76/train-svhn-dataset/standard-trained-classifier-preactresnet50-on-clean-svhn-epoch-0013.pkl --dataset svhn --attack_eps 0.1 >> /home/maggie/mmat/log/SVHN/attack-example-generate/autoattack/svhn-preactresnet50-autoattack-generate-20220721.log 2>&1

# # generate cifar10 preactresnet50 cw adversarial examples
# #----------------------preactresnet50---------------------------------
# CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220720.py run --save_path /home/data/maggie/result-newhome --mode attack --attack_mode cw --whitebox --exp_name preactresnet50-cifar10 --cla_model preactresnet50 --cla_network_pkl /home/data/maggie/result-newhome/train/cla-train/preactresnet50-cifar10/20210925/00003-testacc-84.74/train-cifar10-dataset/standard-trained-classifier-preactresnet50-on-clean-cifar10-epoch-0021.pkl --dataset cifar10 >> /home/maggie/mmat/log/CIFAR10/attack-example-generate/cw/cifar10-preactresnet50-cw-generate-20220721.log 2>&1

# generate svhn preactresnet50 cw adversarial examples
#----------------------preactresnet50---------------------------------
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220720.py run --save_path /home/data/maggie/result-newhome --mode attack --attack_mode cw --whitebox --exp_name preactresnet50-svhn --cla_model preactresnet50 --cla_network_pkl /home/data/maggie/result-newhome/train/cla-train/preactresnet50-svhn/20220627/00001-testacc-95.76/train-svhn-dataset/standard-trained-classifier-preactresnet50-on-clean-svhn-epoch-0013.pkl --dataset svhn >> /home/maggie/mmat/log/SVHN/attack-example-generate/cw/svhn-preactresnet50-cw-generate-20220721.log 2>&1
