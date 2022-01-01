source ~/.bashrc
source /home/xieyi/anaconda3/bin/activate mmat

# adversarial training for FGSM
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220101.py run --mode defense --defense_mode at --attack_mode fgsm --attack_eps 0.3 --whitebox --exp_name preactresnet18-cifar10 --cla_model preactresnet18 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/preactresnet18-cifar10/20210924/00001-teatacc-87.37/train-cifar10-dataset/standard-trained-classifier-preactresnet18-on-clean-cifar10-epoch-0011.pkl --dataset cifar10 --adv_dataset /home/maggie/mmat/result/attack/fgsm/preactresnet18-cifar10/20210927/00000-attackacc-14.71/attack-cifar10-dataset/samples --batch_size 256 --epochs 40 --lr 0.001 >> /home/maggie/mmat/log/preactresnet18-cifar10-at/preactresnet18-cifar10-at-20220101.log 2>&1













# # preactresnet18 defense fgsm at beta(0.5,0.5)
# CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20211110.py run --mode defense --defense_mode at --attack_mode fgsm --attack_eps 0.3 --whitebox --exp_name preactresnet18-cifar10 --cla_model preactresnet18 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/preactresnet18-cifar10/20210924/00001-teatacc-87.37/train-cifar10-dataset/standard-trained-classifier-preactresnet18-on-clean-cifar10-epoch-0011.pkl --dataset cifar10 --adv_dataset /home/maggie/mmat/result/attack/fgsm/preactresnet18-cifar10/20210927/00000-attackacc-14.71/attack-cifar10-dataset/samples --batch_size 256 --epochs 40 --lr 0.001 >> /home/maggie/mmat/log/preactresnet18-cifar10-at/preactresnet18-cifar10-at-20211110.log 2>&1

# # preactresnet34 defense fgsm at beta(0.5,0.5)
# CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20211110.py run --mode defense --defense_mode at --attack_mode fgsm --attack_eps 0.3 --whitebox --exp_name preactresnet34-cifar10 --cla_model preactresnet34 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/preactresnet34-cifar10/20210927/00002-testacc-83.57/train-cifar10-dataset/standard-trained-classifier-preactresnet34-on-clean-cifar10-epoch-0021.pkl --dataset cifar10 --adv_dataset /home/maggie/mmat/result/attack/fgsm/preactresnet34-cifar10/20210927/00000-attackacc-18.04/attack-cifar10-dataset/samples --batch_size 256 --epochs 40 --lr 0.001 >> /home/maggie/mmat/log/preactresnet34-cifar10-at/preactresnet34-cifar10-at-20211110.log 2>&1

# # preactresnet50 defense fgsm at beta(0.5,0.5)
# CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20211110.py run --mode defense --defense_mode at --attack_mode fgsm --attack_eps 0.3 --whitebox --exp_name preactresnet50-cifar10 --cla_model preactresnet50 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/preactresnet50-cifar10/20210925/00003-testacc-84.74/train-cifar10-dataset/standard-trained-classifier-preactresnet50-on-clean-cifar10-epoch-0021.pkl --dataset cifar10 --adv_dataset /home/maggie/mmat/result/attack/fgsm/preactresnet50-cifar10/20210927/00000-attackacc-22.55/attack-cifar10-dataset/samples --batch_size 256 --epochs 40 --lr 0.001 >> /home/maggie/mmat/log/preactresnet50-cifar10-at/preactresnet50-cifar10-at-20211110.log 2>&1

