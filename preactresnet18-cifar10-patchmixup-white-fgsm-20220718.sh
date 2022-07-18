source ~/.bashrc
source /home/xieyi/anaconda3/bin/activate mmat

#   preactresnet18 manifold mixup fgsm
#------------------FGSM------------------------------------
# preactresnet18 defense fgsm patchmixup dual mask bernoulli
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220718.py run --exp_name preactresnet18-cifar10 --mode defense --defense_mode patchmixup --mix_mode maskmixup --sample_mode bernoullisampler --attack_mode fgsm --attack_eps 0.02 --blackbox --cla_model preactresnet18 --dataset cifar10 --cla_network_pkl /home/data/maggie/result-newhome/train/cla-train/preactresnet18-cifar10/20210924/00001-teatacc-87.37/train-cifar10-dataset/standard-trained-classifier-preactresnet18-on-clean-cifar10-epoch-0011.pkl --adv_dataset /home/data/maggie/result-newhome/attack/fgsm/preactresnet18-cifar10/20220627/00000-fgsm-eps-0.02-acc-53.98/attack-cifar10-dataset/samples/test --batch_size 32 --epochs 2 --lr 0.001 >> /home/maggie/mmat/log/CIFAR10/defense/patch-mixup-training/cifar10-preactresnet18-patchmixup-20220718.log 2>&1

# preactresnet18 defense fgsm patchmixup dual mask bernoulli

# preactresnet18 defense fgsm patchmixup dual mask bernoulli
