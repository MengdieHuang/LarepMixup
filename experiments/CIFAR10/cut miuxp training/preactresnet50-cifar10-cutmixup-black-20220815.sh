source ~/.bashrc
source /home/xieyi/anaconda3/bin/activate mmat

# # adv50000 cat cle50000
# # PreActResNet50 cutmixup beta(1,1) against FGSM (eps=0.05)
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220815.py run --mode defense --defense_mode cutmixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode fgsm --attack_eps 0.05 --blackbox --exp_name preactresnet50-cifar10 --cla_model preactresnet50 --cla_network_pkl /home/data/maggie/result-newhome/train/cla-train/preactresnet50-cifar10/20210925/00003-testacc-84.74/train-cifar10-dataset/standard-trained-classifier-preactresnet50-on-clean-cifar10-epoch-0021.pkl --dataset cifar10 --test_adv_dataset /home/data/maggie/result-newhome/attack/fgsm/preactresnet50-cifar10/20220627/00001-fgsm-eps-0.05-acc-35.27/attack-cifar10-dataset/samples/test --batch_size 256 --epochs 30 --lr 0.01 >> /home/maggie/mmat/log/CIFAR10/defense/cut-mixup-training/preactresnet50-cifar10-cutmixup-fgsm-20220815.log 2>&1

# # PreActResNet50 cutmixup beta(1,1) against FGSM (eps=0.05)
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220815.py run --mode defense --defense_mode cutmixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode fgsm --attack_eps 0.05 --blackbox --exp_name preactresnet50-cifar10 --cla_model preactresnet50 --cla_network_pkl /home/data/maggie/result-newhome/train/cla-train/preactresnet50-cifar10/20210925/00003-testacc-84.74/train-cifar10-dataset/standard-trained-classifier-preactresnet50-on-clean-cifar10-epoch-0021.pkl --dataset cifar10 --test_adv_dataset /home/data/maggie/result-newhome/attack/fgsm/preactresnet50-cifar10/20220627/00001-fgsm-eps-0.05-acc-35.27/attack-cifar10-dataset/samples/test --batch_size 256 --epochs 30 --lr 0.01 >> /home/maggie/mmat/log/CIFAR10/defense/cut-mixup-training/preactresnet50-cifar10-cutmixup-fgsm-20220815.log 2>&1

# # PreActResNet50 cutmixup beta(1,1) against FGSM (eps=0.05)
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220815.py run --mode defense --defense_mode cutmixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode fgsm --attack_eps 0.05 --blackbox --exp_name preactresnet50-cifar10 --cla_model preactresnet50 --cla_network_pkl /home/data/maggie/result-newhome/train/cla-train/preactresnet50-cifar10/20210925/00003-testacc-84.74/train-cifar10-dataset/standard-trained-classifier-preactresnet50-on-clean-cifar10-epoch-0021.pkl --dataset cifar10 --test_adv_dataset /home/data/maggie/result-newhome/attack/fgsm/preactresnet50-cifar10/20220627/00001-fgsm-eps-0.05-acc-35.27/attack-cifar10-dataset/samples/test --batch_size 256 --epochs 30 --lr 0.01 >> /home/maggie/mmat/log/CIFAR10/defense/cut-mixup-training/preactresnet50-cifar10-cutmixup-fgsm-20220815.log 2>&1

# # PreActResNet50 cutmixup beta(1,1) against FGSM (eps=0.05)
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220815.py run --mode defense --defense_mode cutmixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode fgsm --attack_eps 0.05 --blackbox --exp_name preactresnet50-cifar10 --cla_model preactresnet50 --cla_network_pkl /home/data/maggie/result-newhome/train/cla-train/preactresnet50-cifar10/20210925/00003-testacc-84.74/train-cifar10-dataset/standard-trained-classifier-preactresnet50-on-clean-cifar10-epoch-0021.pkl --dataset cifar10 --test_adv_dataset /home/data/maggie/result-newhome/attack/fgsm/preactresnet50-cifar10/20220627/00001-fgsm-eps-0.05-acc-35.27/attack-cifar10-dataset/samples/test --batch_size 256 --epochs 30 --lr 0.01 >> /home/maggie/mmat/log/CIFAR10/defense/cut-mixup-training/preactresnet50-cifar10-cutmixup-fgsm-20220815.log 2>&1

# # PreActResNet50 cutmixup beta(1,1) against FGSM (eps=0.05)
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220815.py run --mode defense --defense_mode cutmixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode fgsm --attack_eps 0.05 --blackbox --exp_name preactresnet50-cifar10 --cla_model preactresnet50 --cla_network_pkl /home/data/maggie/result-newhome/train/cla-train/preactresnet50-cifar10/20210925/00003-testacc-84.74/train-cifar10-dataset/standard-trained-classifier-preactresnet50-on-clean-cifar10-epoch-0021.pkl --dataset cifar10 --test_adv_dataset /home/data/maggie/result-newhome/attack/fgsm/preactresnet50-cifar10/20220627/00001-fgsm-eps-0.05-acc-35.27/attack-cifar10-dataset/samples/test --batch_size 256 --epochs 30 --lr 0.01 >> /home/maggie/mmat/log/CIFAR10/defense/cut-mixup-training/preactresnet50-cifar10-cutmixup-fgsm-20220815.log 2>&1

# # PreActResNet50 cutmixup beta(1,1) against FGSM (eps=0.05)
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220815.py run --mode defense --defense_mode cutmixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode fgsm --attack_eps 0.05 --blackbox --exp_name preactresnet50-cifar10 --cla_model preactresnet50 --cla_network_pkl /home/data/maggie/result-newhome/train/cla-train/preactresnet50-cifar10/20210925/00003-testacc-84.74/train-cifar10-dataset/standard-trained-classifier-preactresnet50-on-clean-cifar10-epoch-0021.pkl --dataset cifar10 --test_adv_dataset /home/data/maggie/result-newhome/attack/fgsm/preactresnet50-cifar10/20220627/00001-fgsm-eps-0.05-acc-35.27/attack-cifar10-dataset/samples/test --batch_size 256 --epochs 30 --lr 0.01 >> /home/maggie/mmat/log/CIFAR10/defense/cut-mixup-training/preactresnet50-cifar10-cutmixup-fgsm-20220815.log 2>&1

#-------------以上要存储at训练好的模型-----------
# #   使用PGD对抗训练后的 cla1
# PreActResNet50 cutmixup beta(1,1) against pgd (eps=0.05)
CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220815.py run --mode defense --defense_mode cutmixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode pgd --attack_eps 0.05 --blackbox --exp_name preactresnet50-cifar10 --cla_model preactresnet50 --cla_network_pkl /home/data/maggie/result-newhome/defense/cutmixup/fgsm/basemixup-betasampler/preactresnet50-cifar10/blackbox/20220822/00000/cutmixup-cifar10-dataset/cutmixup-trained-classifier-preactresnet50-on-cifar10-epoch-0040-cutmix-cla1.pkl --dataset cifar10 --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/preactresnet50-cifar10/20220625/00001-pgd-eps-0.05-acc-26.89/attack-cifar10-dataset/samples/test --batch_size 256 --epochs 30 --lr 0.01 >> /home/maggie/mmat/log/CIFAR10/defense/cut-mixup-training/preactresnet50-cifar10-cutmixup-pgd-20220815.log 2>&1

# PreActResNet50 cutmixup beta(1,1) against autoattack (eps=0.05)
CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220815.py run --mode defense --defense_mode cutmixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode autoattack --attack_eps 0.05 --blackbox --exp_name preactresnet50-cifar10 --cla_model preactresnet50 --cla_network_pkl /home/data/maggie/result-newhome/defense/cutmixup/fgsm/basemixup-betasampler/preactresnet50-cifar10/blackbox/20220822/00000/cutmixup-cifar10-dataset/cutmixup-trained-classifier-preactresnet50-on-cifar10-epoch-0040-cutmix-cla1.pkl --dataset cifar10 --test_adv_dataset /home/data/maggie/result-newhome/attack/autoattack/preactresnet50-cifar10/20220716/00000-autoattack-eps-0.05-acc-5.43/attack-cifar10-dataset/samples/test --batch_size 256 --epochs 30 --lr 0.01 >> /home/maggie/mmat/log/CIFAR10/defense/cut-mixup-training/preactresnet50-cifar10-cutmixup-autoattack-20220815.log 2>&1

# PreActResNet50 cutmixup beta(1,1) against deepfool (eps=0.02)
CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220815.py run --mode defense --defense_mode cutmixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode deepfool --attack_eps 0.02 --blackbox --exp_name preactresnet50-cifar10 --cla_model preactresnet50 --cla_network_pkl /home/data/maggie/result-newhome/defense/cutmixup/fgsm/basemixup-betasampler/preactresnet50-cifar10/blackbox/20220822/00000/cutmixup-cifar10-dataset/cutmixup-trained-classifier-preactresnet50-on-cifar10-epoch-0040-cutmix-cla1.pkl --dataset cifar10 --test_adv_dataset /home/data/maggie/result-newhome/attack/deepfool/preactresnet50-cifar10/20220708/00000-deepfool-eps-0.02-acc-12.03/attack-cifar10-dataset/samples/test --batch_size 256 --epochs 30 --lr 0.01 >> /home/maggie/mmat/log/CIFAR10/defense/cut-mixup-training/preactresnet50-cifar10-cutmixup-deepfool-20220815.log 2>&1

# PreActResNet50 cutmixup beta(1,1) against cw (eps=0.02)
CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220815.py run --mode defense --defense_mode cutmixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode cw --attack_eps 0.05 --blackbox --exp_name preactresnet50-cifar10 --cla_model preactresnet50 --cla_network_pkl /home/data/maggie/result-newhome/defense/cutmixup/fgsm/basemixup-betasampler/preactresnet50-cifar10/blackbox/20220822/00000/cutmixup-cifar10-dataset/cutmixup-trained-classifier-preactresnet50-on-cifar10-epoch-0040-cutmix-cla1.pkl --dataset cifar10 --test_adv_dataset /home/data/maggie/result-newhome/attack/cw/preactresnet50-cifar10/20220802/00000-cw-confidence-0.0-acc-1.13/attack-cifar10-dataset/samples/test --batch_size 256 --epochs 30 --lr 0.01 >> /home/maggie/mmat/log/CIFAR10/defense/cut-mixup-training/preactresnet50-cifar10-cutmixup-cw-20220815.log 2>&1

# PreActResNet50 cutmixup beta(1,1) against om-fgsm (eps=0.05)
CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220815.py run --mode defense --defense_mode cutmixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode om-fgsm --attack_eps 0.05 --blackbox --exp_name preactresnet50-cifar10 --cla_model preactresnet50 --cla_network_pkl /home/data/maggie/result-newhome/defense/cutmixup/fgsm/basemixup-betasampler/preactresnet50-cifar10/blackbox/20220822/00000/cutmixup-cifar10-dataset/cutmixup-trained-classifier-preactresnet50-on-cifar10-epoch-0040-cutmix-cla1.pkl --dataset cifar10 --test_adv_dataset /home/data/maggie/result-newhome/attack/fgsm/preactresnet50-cifar10/20220702/00001-omfgsm-eps-0.05-acc-50.26/attack-cifar10-dataset/latent-attack-samples/test --batch_size 256 --epochs 30 --lr 0.01 >> /home/maggie/mmat/log/CIFAR10/defense/cut-mixup-training/preactresnet50-cifar10-cutmixup-om-fgsm-20220815.log 2>&1

# PreActResNet50 cutmixup beta(1,1) against om-pgd (eps=0.05)
CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220815.py run --mode defense --defense_mode cutmixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode om-pgd --attack_eps 0.05 --blackbox --exp_name preactresnet50-cifar10 --cla_model preactresnet50 --cla_network_pkl /home/data/maggie/result-newhome/defense/cutmixup/fgsm/basemixup-betasampler/preactresnet50-cifar10/blackbox/20220822/00000/cutmixup-cifar10-dataset/cutmixup-trained-classifier-preactresnet50-on-cifar10-epoch-0040-cutmix-cla1.pkl --dataset cifar10 --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/preactresnet50-cifar10/20220704/00001-ompgd-eps-0.05-acc-19.13/attack-cifar10-dataset/latent-attack-samples/test --batch_size 256 --epochs 30 --lr 0.01 >> /home/maggie/mmat/log/CIFAR10/defense/cut-mixup-training/preactresnet50-cifar10-cutmixup-om-pgd-20220815.log 2>&1

# #   使用PGD对抗训练后的 cla2
# PreActResNet50 cutmixup beta(1,1) against pgd (eps=0.05)
CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220815.py run --mode defense --defense_mode cutmixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode pgd --attack_eps 0.05 --blackbox --exp_name preactresnet50-cifar10 --cla_model preactresnet50 --cla_network_pkl /home/data/maggie/result-newhome/defense/cutmixup/fgsm/basemixup-betasampler/preactresnet50-cifar10/blackbox/20220823/00001/cutmixup-cifar10-dataset/cutmixup-trained-classifier-preactresnet50-on-cifar10-epoch-0040-cutmix-cla2.pkl --dataset cifar10 --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/preactresnet50-cifar10/20220625/00001-pgd-eps-0.05-acc-26.89/attack-cifar10-dataset/samples/test --batch_size 256 --epochs 30 --lr 0.01 >> /home/maggie/mmat/log/CIFAR10/defense/cut-mixup-training/preactresnet50-cifar10-cutmixup-pgd-20220815.log 2>&1

# PreActResNet50 cutmixup beta(1,1) against autoattack (eps=0.05)
CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220815.py run --mode defense --defense_mode cutmixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode autoattack --attack_eps 0.05 --blackbox --exp_name preactresnet50-cifar10 --cla_model preactresnet50 --cla_network_pkl /home/data/maggie/result-newhome/defense/cutmixup/fgsm/basemixup-betasampler/preactresnet50-cifar10/blackbox/20220823/00001/cutmixup-cifar10-dataset/cutmixup-trained-classifier-preactresnet50-on-cifar10-epoch-0040-cutmix-cla2.pkl --dataset cifar10 --test_adv_dataset /home/data/maggie/result-newhome/attack/autoattack/preactresnet50-cifar10/20220716/00000-autoattack-eps-0.05-acc-5.43/attack-cifar10-dataset/samples/test --batch_size 256 --epochs 30 --lr 0.01 >> /home/maggie/mmat/log/CIFAR10/defense/cut-mixup-training/preactresnet50-cifar10-cutmixup-autoattack-20220815.log 2>&1

# PreActResNet50 cutmixup beta(1,1) against deepfool (eps=0.02)
CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220815.py run --mode defense --defense_mode cutmixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode deepfool --attack_eps 0.02 --blackbox --exp_name preactresnet50-cifar10 --cla_model preactresnet50 --cla_network_pkl /home/data/maggie/result-newhome/defense/cutmixup/fgsm/basemixup-betasampler/preactresnet50-cifar10/blackbox/20220823/00001/cutmixup-cifar10-dataset/cutmixup-trained-classifier-preactresnet50-on-cifar10-epoch-0040-cutmix-cla2.pkl --dataset cifar10 --test_adv_dataset /home/data/maggie/result-newhome/attack/deepfool/preactresnet50-cifar10/20220708/00000-deepfool-eps-0.02-acc-12.03/attack-cifar10-dataset/samples/test --batch_size 256 --epochs 30 --lr 0.01 >> /home/maggie/mmat/log/CIFAR10/defense/cut-mixup-training/preactresnet50-cifar10-cutmixup-deepfool-20220815.log 2>&1

# PreActResNet50 cutmixup beta(1,1) against cw (eps=0.02)
CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220815.py run --mode defense --defense_mode cutmixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode cw --attack_eps 0.05 --blackbox --exp_name preactresnet50-cifar10 --cla_model preactresnet50 --cla_network_pkl /home/data/maggie/result-newhome/defense/cutmixup/fgsm/basemixup-betasampler/preactresnet50-cifar10/blackbox/20220823/00001/cutmixup-cifar10-dataset/cutmixup-trained-classifier-preactresnet50-on-cifar10-epoch-0040-cutmix-cla2.pkl --dataset cifar10 --test_adv_dataset /home/data/maggie/result-newhome/attack/cw/preactresnet50-cifar10/20220802/00000-cw-confidence-0.0-acc-1.13/attack-cifar10-dataset/samples/test --batch_size 256 --epochs 30 --lr 0.01 >> /home/maggie/mmat/log/CIFAR10/defense/cut-mixup-training/preactresnet50-cifar10-cutmixup-cw-20220815.log 2>&1

# PreActResNet50 cutmixup beta(1,1) against om-fgsm (eps=0.05)
CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220815.py run --mode defense --defense_mode cutmixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode om-fgsm --attack_eps 0.05 --blackbox --exp_name preactresnet50-cifar10 --cla_model preactresnet50 --cla_network_pkl /home/data/maggie/result-newhome/defense/cutmixup/fgsm/basemixup-betasampler/preactresnet50-cifar10/blackbox/20220823/00001/cutmixup-cifar10-dataset/cutmixup-trained-classifier-preactresnet50-on-cifar10-epoch-0040-cutmix-cla2.pkl --dataset cifar10 --test_adv_dataset /home/data/maggie/result-newhome/attack/fgsm/preactresnet50-cifar10/20220702/00001-omfgsm-eps-0.05-acc-50.26/attack-cifar10-dataset/latent-attack-samples/test --batch_size 256 --epochs 30 --lr 0.01 >> /home/maggie/mmat/log/CIFAR10/defense/cut-mixup-training/preactresnet50-cifar10-cutmixup-om-fgsm-20220815.log 2>&1

# PreActResNet50 cutmixup beta(1,1) against om-pgd (eps=0.05)
CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220815.py run --mode defense --defense_mode cutmixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode om-pgd --attack_eps 0.05 --blackbox --exp_name preactresnet50-cifar10 --cla_model preactresnet50 --cla_network_pkl /home/data/maggie/result-newhome/defense/cutmixup/fgsm/basemixup-betasampler/preactresnet50-cifar10/blackbox/20220823/00001/cutmixup-cifar10-dataset/cutmixup-trained-classifier-preactresnet50-on-cifar10-epoch-0040-cutmix-cla2.pkl --dataset cifar10 --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/preactresnet50-cifar10/20220704/00001-ompgd-eps-0.05-acc-19.13/attack-cifar10-dataset/latent-attack-samples/test --batch_size 256 --epochs 30 --lr 0.01 >> /home/maggie/mmat/log/CIFAR10/defense/cut-mixup-training/preactresnet50-cifar10-cutmixup-om-pgd-20220815.log 2>&1

# #   使用PGD对抗训练后的 cla3
# PreActResNet50 cutmixup beta(1,1) against pgd (eps=0.05)
CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220815.py run --mode defense --defense_mode cutmixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode pgd --attack_eps 0.05 --blackbox --exp_name preactresnet50-cifar10 --cla_model preactresnet50 --cla_network_pkl /home/data/maggie/result-newhome/defense/cutmixup/fgsm/basemixup-betasampler/preactresnet50-cifar10/blackbox/20220823/00002/cutmixup-cifar10-dataset/cutmixup-trained-classifier-preactresnet50-on-cifar10-epoch-0040-cutmix-cla3.pkl --dataset cifar10 --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/preactresnet50-cifar10/20220625/00001-pgd-eps-0.05-acc-26.89/attack-cifar10-dataset/samples/test --batch_size 256 --epochs 30 --lr 0.01 >> /home/maggie/mmat/log/CIFAR10/defense/cut-mixup-training/preactresnet50-cifar10-cutmixup-pgd-20220815.log 2>&1

# PreActResNet50 cutmixup beta(1,1) against autoattack (eps=0.05)
CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220815.py run --mode defense --defense_mode cutmixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode autoattack --attack_eps 0.05 --blackbox --exp_name preactresnet50-cifar10 --cla_model preactresnet50 --cla_network_pkl /home/data/maggie/result-newhome/defense/cutmixup/fgsm/basemixup-betasampler/preactresnet50-cifar10/blackbox/20220823/00002/cutmixup-cifar10-dataset/cutmixup-trained-classifier-preactresnet50-on-cifar10-epoch-0040-cutmix-cla3.pkl --dataset cifar10 --test_adv_dataset /home/data/maggie/result-newhome/attack/autoattack/preactresnet50-cifar10/20220716/00000-autoattack-eps-0.05-acc-5.43/attack-cifar10-dataset/samples/test --batch_size 256 --epochs 30 --lr 0.01 >> /home/maggie/mmat/log/CIFAR10/defense/cut-mixup-training/preactresnet50-cifar10-cutmixup-autoattack-20220815.log 2>&1

# PreActResNet50 cutmixup beta(1,1) against deepfool (eps=0.02)
CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220815.py run --mode defense --defense_mode cutmixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode deepfool --attack_eps 0.02 --blackbox --exp_name preactresnet50-cifar10 --cla_model preactresnet50 --cla_network_pkl /home/data/maggie/result-newhome/defense/cutmixup/fgsm/basemixup-betasampler/preactresnet50-cifar10/blackbox/20220823/00002/cutmixup-cifar10-dataset/cutmixup-trained-classifier-preactresnet50-on-cifar10-epoch-0040-cutmix-cla3.pkl --dataset cifar10 --test_adv_dataset /home/data/maggie/result-newhome/attack/deepfool/preactresnet50-cifar10/20220708/00000-deepfool-eps-0.02-acc-12.03/attack-cifar10-dataset/samples/test --batch_size 256 --epochs 30 --lr 0.01 >> /home/maggie/mmat/log/CIFAR10/defense/cut-mixup-training/preactresnet50-cifar10-cutmixup-deepfool-20220815.log 2>&1

# PreActResNet50 cutmixup beta(1,1) against cw (eps=0.02)
CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220815.py run --mode defense --defense_mode cutmixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode cw --attack_eps 0.05 --blackbox --exp_name preactresnet50-cifar10 --cla_model preactresnet50 --cla_network_pkl /home/data/maggie/result-newhome/defense/cutmixup/fgsm/basemixup-betasampler/preactresnet50-cifar10/blackbox/20220823/00002/cutmixup-cifar10-dataset/cutmixup-trained-classifier-preactresnet50-on-cifar10-epoch-0040-cutmix-cla3.pkl --dataset cifar10 --test_adv_dataset /home/data/maggie/result-newhome/attack/cw/preactresnet50-cifar10/20220802/00000-cw-confidence-0.0-acc-1.13/attack-cifar10-dataset/samples/test --batch_size 256 --epochs 30 --lr 0.01 >> /home/maggie/mmat/log/CIFAR10/defense/cut-mixup-training/preactresnet50-cifar10-cutmixup-cw-20220815.log 2>&1

# PreActResNet50 cutmixup beta(1,1) against om-fgsm (eps=0.05)
CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220815.py run --mode defense --defense_mode cutmixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode om-fgsm --attack_eps 0.05 --blackbox --exp_name preactresnet50-cifar10 --cla_model preactresnet50 --cla_network_pkl /home/data/maggie/result-newhome/defense/cutmixup/fgsm/basemixup-betasampler/preactresnet50-cifar10/blackbox/20220823/00002/cutmixup-cifar10-dataset/cutmixup-trained-classifier-preactresnet50-on-cifar10-epoch-0040-cutmix-cla3.pkl --dataset cifar10 --test_adv_dataset /home/data/maggie/result-newhome/attack/fgsm/preactresnet50-cifar10/20220702/00001-omfgsm-eps-0.05-acc-50.26/attack-cifar10-dataset/latent-attack-samples/test --batch_size 256 --epochs 30 --lr 0.01 >> /home/maggie/mmat/log/CIFAR10/defense/cut-mixup-training/preactresnet50-cifar10-cutmixup-om-fgsm-20220815.log 2>&1

# PreActResNet50 cutmixup beta(1,1) against om-pgd (eps=0.05)
CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220815.py run --mode defense --defense_mode cutmixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode om-pgd --attack_eps 0.05 --blackbox --exp_name preactresnet50-cifar10 --cla_model preactresnet50 --cla_network_pkl /home/data/maggie/result-newhome/defense/cutmixup/fgsm/basemixup-betasampler/preactresnet50-cifar10/blackbox/20220823/00002/cutmixup-cifar10-dataset/cutmixup-trained-classifier-preactresnet50-on-cifar10-epoch-0040-cutmix-cla3.pkl --dataset cifar10 --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/preactresnet50-cifar10/20220704/00001-ompgd-eps-0.05-acc-19.13/attack-cifar10-dataset/latent-attack-samples/test --batch_size 256 --epochs 30 --lr 0.01 >> /home/maggie/mmat/log/CIFAR10/defense/cut-mixup-training/preactresnet50-cifar10-cutmixup-om-pgd-20220815.log 2>&1

# #   使用PGD对抗训练后的 cla4
# PreActResNet50 cutmixup beta(1,1) against pgd (eps=0.05)
CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220815.py run --mode defense --defense_mode cutmixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode pgd --attack_eps 0.05 --blackbox --exp_name preactresnet50-cifar10 --cla_model preactresnet50 --cla_network_pkl /home/data/maggie/result-newhome/defense/cutmixup/fgsm/basemixup-betasampler/preactresnet50-cifar10/blackbox/20220823/00003/cutmixup-cifar10-dataset/cutmixup-trained-classifier-preactresnet50-on-cifar10-epoch-0040-cutmix-cla4.pkl --dataset cifar10 --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/preactresnet50-cifar10/20220625/00001-pgd-eps-0.05-acc-26.89/attack-cifar10-dataset/samples/test --batch_size 256 --epochs 30 --lr 0.01 >> /home/maggie/mmat/log/CIFAR10/defense/cut-mixup-training/preactresnet50-cifar10-cutmixup-pgd-20220815.log 2>&1

# PreActResNet50 cutmixup beta(1,1) against autoattack (eps=0.05)
CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220815.py run --mode defense --defense_mode cutmixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode autoattack --attack_eps 0.05 --blackbox --exp_name preactresnet50-cifar10 --cla_model preactresnet50 --cla_network_pkl /home/data/maggie/result-newhome/defense/cutmixup/fgsm/basemixup-betasampler/preactresnet50-cifar10/blackbox/20220823/00003/cutmixup-cifar10-dataset/cutmixup-trained-classifier-preactresnet50-on-cifar10-epoch-0040-cutmix-cla4.pkl --dataset cifar10 --test_adv_dataset /home/data/maggie/result-newhome/attack/autoattack/preactresnet50-cifar10/20220716/00000-autoattack-eps-0.05-acc-5.43/attack-cifar10-dataset/samples/test --batch_size 256 --epochs 30 --lr 0.01 >> /home/maggie/mmat/log/CIFAR10/defense/cut-mixup-training/preactresnet50-cifar10-cutmixup-autoattack-20220815.log 2>&1

# PreActResNet50 cutmixup beta(1,1) against deepfool (eps=0.02)
CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220815.py run --mode defense --defense_mode cutmixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode deepfool --attack_eps 0.02 --blackbox --exp_name preactresnet50-cifar10 --cla_model preactresnet50 --cla_network_pkl /home/data/maggie/result-newhome/defense/cutmixup/fgsm/basemixup-betasampler/preactresnet50-cifar10/blackbox/20220823/00003/cutmixup-cifar10-dataset/cutmixup-trained-classifier-preactresnet50-on-cifar10-epoch-0040-cutmix-cla4.pkl --dataset cifar10 --test_adv_dataset /home/data/maggie/result-newhome/attack/deepfool/preactresnet50-cifar10/20220708/00000-deepfool-eps-0.02-acc-12.03/attack-cifar10-dataset/samples/test --batch_size 256 --epochs 30 --lr 0.01 >> /home/maggie/mmat/log/CIFAR10/defense/cut-mixup-training/preactresnet50-cifar10-cutmixup-deepfool-20220815.log 2>&1

# PreActResNet50 cutmixup beta(1,1) against cw (eps=0.02)
CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220815.py run --mode defense --defense_mode cutmixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode cw --attack_eps 0.05 --blackbox --exp_name preactresnet50-cifar10 --cla_model preactresnet50 --cla_network_pkl /home/data/maggie/result-newhome/defense/cutmixup/fgsm/basemixup-betasampler/preactresnet50-cifar10/blackbox/20220823/00003/cutmixup-cifar10-dataset/cutmixup-trained-classifier-preactresnet50-on-cifar10-epoch-0040-cutmix-cla4.pkl --dataset cifar10 --test_adv_dataset /home/data/maggie/result-newhome/attack/cw/preactresnet50-cifar10/20220802/00000-cw-confidence-0.0-acc-1.13/attack-cifar10-dataset/samples/test --batch_size 256 --epochs 30 --lr 0.01 >> /home/maggie/mmat/log/CIFAR10/defense/cut-mixup-training/preactresnet50-cifar10-cutmixup-cw-20220815.log 2>&1

# PreActResNet50 cutmixup beta(1,1) against om-fgsm (eps=0.05)
CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220815.py run --mode defense --defense_mode cutmixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode om-fgsm --attack_eps 0.05 --blackbox --exp_name preactresnet50-cifar10 --cla_model preactresnet50 --cla_network_pkl /home/data/maggie/result-newhome/defense/cutmixup/fgsm/basemixup-betasampler/preactresnet50-cifar10/blackbox/20220823/00003/cutmixup-cifar10-dataset/cutmixup-trained-classifier-preactresnet50-on-cifar10-epoch-0040-cutmix-cla4.pkl --dataset cifar10 --test_adv_dataset /home/data/maggie/result-newhome/attack/fgsm/preactresnet50-cifar10/20220702/00001-omfgsm-eps-0.05-acc-50.26/attack-cifar10-dataset/latent-attack-samples/test --batch_size 256 --epochs 30 --lr 0.01 >> /home/maggie/mmat/log/CIFAR10/defense/cut-mixup-training/preactresnet50-cifar10-cutmixup-om-fgsm-20220815.log 2>&1

# PreActResNet50 cutmixup beta(1,1) against om-pgd (eps=0.05)
CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220815.py run --mode defense --defense_mode cutmixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode om-pgd --attack_eps 0.05 --blackbox --exp_name preactresnet50-cifar10 --cla_model preactresnet50 --cla_network_pkl /home/data/maggie/result-newhome/defense/cutmixup/fgsm/basemixup-betasampler/preactresnet50-cifar10/blackbox/20220823/00003/cutmixup-cifar10-dataset/cutmixup-trained-classifier-preactresnet50-on-cifar10-epoch-0040-cutmix-cla4.pkl --dataset cifar10 --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/preactresnet50-cifar10/20220704/00001-ompgd-eps-0.05-acc-19.13/attack-cifar10-dataset/latent-attack-samples/test --batch_size 256 --epochs 30 --lr 0.01 >> /home/maggie/mmat/log/CIFAR10/defense/cut-mixup-training/preactresnet50-cifar10-cutmixup-om-pgd-20220815.log 2>&1

# #   使用PGD对抗训练后的 cla5
# PreActResNet50 cutmixup beta(1,1) against pgd (eps=0.05)
CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220815.py run --mode defense --defense_mode cutmixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode pgd --attack_eps 0.05 --blackbox --exp_name preactresnet50-cifar10 --cla_model preactresnet50 --cla_network_pkl /home/data/maggie/result-newhome/defense/cutmixup/fgsm/basemixup-betasampler/preactresnet50-cifar10/blackbox/20220823/00004/cutmixup-cifar10-dataset/cutmixup-trained-classifier-preactresnet50-on-cifar10-epoch-0040-cutmix-cla5.pkl --dataset cifar10 --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/preactresnet50-cifar10/20220625/00001-pgd-eps-0.05-acc-26.89/attack-cifar10-dataset/samples/test --batch_size 256 --epochs 30 --lr 0.01 >> /home/maggie/mmat/log/CIFAR10/defense/cut-mixup-training/preactresnet50-cifar10-cutmixup-pgd-20220815.log 2>&1

# PreActResNet50 cutmixup beta(1,1) against autoattack (eps=0.05)
CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220815.py run --mode defense --defense_mode cutmixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode autoattack --attack_eps 0.05 --blackbox --exp_name preactresnet50-cifar10 --cla_model preactresnet50 --cla_network_pkl /home/data/maggie/result-newhome/defense/cutmixup/fgsm/basemixup-betasampler/preactresnet50-cifar10/blackbox/20220823/00004/cutmixup-cifar10-dataset/cutmixup-trained-classifier-preactresnet50-on-cifar10-epoch-0040-cutmix-cla5.pkl --dataset cifar10 --test_adv_dataset /home/data/maggie/result-newhome/attack/autoattack/preactresnet50-cifar10/20220716/00000-autoattack-eps-0.05-acc-5.43/attack-cifar10-dataset/samples/test --batch_size 256 --epochs 30 --lr 0.01 >> /home/maggie/mmat/log/CIFAR10/defense/cut-mixup-training/preactresnet50-cifar10-cutmixup-autoattack-20220815.log 2>&1

# PreActResNet50 cutmixup beta(1,1) against deepfool (eps=0.02)
CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220815.py run --mode defense --defense_mode cutmixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode deepfool --attack_eps 0.02 --blackbox --exp_name preactresnet50-cifar10 --cla_model preactresnet50 --cla_network_pkl /home/data/maggie/result-newhome/defense/cutmixup/fgsm/basemixup-betasampler/preactresnet50-cifar10/blackbox/20220823/00004/cutmixup-cifar10-dataset/cutmixup-trained-classifier-preactresnet50-on-cifar10-epoch-0040-cutmix-cla5.pkl --dataset cifar10 --test_adv_dataset /home/data/maggie/result-newhome/attack/deepfool/preactresnet50-cifar10/20220708/00000-deepfool-eps-0.02-acc-12.03/attack-cifar10-dataset/samples/test --batch_size 256 --epochs 30 --lr 0.01 >> /home/maggie/mmat/log/CIFAR10/defense/cut-mixup-training/preactresnet50-cifar10-cutmixup-deepfool-20220815.log 2>&1

# PreActResNet50 cutmixup beta(1,1) against cw (eps=0.02)
CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220815.py run --mode defense --defense_mode cutmixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode cw --attack_eps 0.05 --blackbox --exp_name preactresnet50-cifar10 --cla_model preactresnet50 --cla_network_pkl /home/data/maggie/result-newhome/defense/cutmixup/fgsm/basemixup-betasampler/preactresnet50-cifar10/blackbox/20220823/00004/cutmixup-cifar10-dataset/cutmixup-trained-classifier-preactresnet50-on-cifar10-epoch-0040-cutmix-cla5.pkl --dataset cifar10 --test_adv_dataset /home/data/maggie/result-newhome/attack/cw/preactresnet50-cifar10/20220802/00000-cw-confidence-0.0-acc-1.13/attack-cifar10-dataset/samples/test --batch_size 256 --epochs 30 --lr 0.01 >> /home/maggie/mmat/log/CIFAR10/defense/cut-mixup-training/preactresnet50-cifar10-cutmixup-cw-20220815.log 2>&1

# PreActResNet50 cutmixup beta(1,1) against om-fgsm (eps=0.05)
CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220815.py run --mode defense --defense_mode cutmixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode om-fgsm --attack_eps 0.05 --blackbox --exp_name preactresnet50-cifar10 --cla_model preactresnet50 --cla_network_pkl /home/data/maggie/result-newhome/defense/cutmixup/fgsm/basemixup-betasampler/preactresnet50-cifar10/blackbox/20220823/00004/cutmixup-cifar10-dataset/cutmixup-trained-classifier-preactresnet50-on-cifar10-epoch-0040-cutmix-cla5.pkl --dataset cifar10 --test_adv_dataset /home/data/maggie/result-newhome/attack/fgsm/preactresnet50-cifar10/20220702/00001-omfgsm-eps-0.05-acc-50.26/attack-cifar10-dataset/latent-attack-samples/test --batch_size 256 --epochs 30 --lr 0.01 >> /home/maggie/mmat/log/CIFAR10/defense/cut-mixup-training/preactresnet50-cifar10-cutmixup-om-fgsm-20220815.log 2>&1

# PreActResNet50 cutmixup beta(1,1) against om-pgd (eps=0.05)
CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220815.py run --mode defense --defense_mode cutmixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode om-pgd --attack_eps 0.05 --blackbox --exp_name preactresnet50-cifar10 --cla_model preactresnet50 --cla_network_pkl /home/data/maggie/result-newhome/defense/cutmixup/fgsm/basemixup-betasampler/preactresnet50-cifar10/blackbox/20220823/00004/cutmixup-cifar10-dataset/cutmixup-trained-classifier-preactresnet50-on-cifar10-epoch-0040-cutmix-cla5.pkl --dataset cifar10 --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/preactresnet50-cifar10/20220704/00001-ompgd-eps-0.05-acc-19.13/attack-cifar10-dataset/latent-attack-samples/test --batch_size 256 --epochs 30 --lr 0.01 >> /home/maggie/mmat/log/CIFAR10/defense/cut-mixup-training/preactresnet50-cifar10-cutmixup-om-pgd-20220815.log 2>&1

# #   使用PGD对抗训练后的 cla6
# PreActResNet50 cutmixup beta(1,1) against pgd (eps=0.05)
CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220815.py run --mode defense --defense_mode cutmixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode pgd --attack_eps 0.05 --blackbox --exp_name preactresnet50-cifar10 --cla_model preactresnet50 --cla_network_pkl /home/data/maggie/result-newhome/defense/cutmixup/fgsm/basemixup-betasampler/preactresnet50-cifar10/blackbox/20220823/00005/cutmixup-cifar10-dataset/cutmixup-trained-classifier-preactresnet50-on-cifar10-epoch-0040-cutmix-cla6.pkl --dataset cifar10 --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/preactresnet50-cifar10/20220625/00001-pgd-eps-0.05-acc-26.89/attack-cifar10-dataset/samples/test --batch_size 256 --epochs 30 --lr 0.01 >> /home/maggie/mmat/log/CIFAR10/defense/cut-mixup-training/preactresnet50-cifar10-cutmixup-pgd-20220815.log 2>&1

# PreActResNet50 cutmixup beta(1,1) against autoattack (eps=0.05)
CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220815.py run --mode defense --defense_mode cutmixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode autoattack --attack_eps 0.05 --blackbox --exp_name preactresnet50-cifar10 --cla_model preactresnet50 --cla_network_pkl /home/data/maggie/result-newhome/defense/cutmixup/fgsm/basemixup-betasampler/preactresnet50-cifar10/blackbox/20220823/00005/cutmixup-cifar10-dataset/cutmixup-trained-classifier-preactresnet50-on-cifar10-epoch-0040-cutmix-cla6.pkl --dataset cifar10 --test_adv_dataset /home/data/maggie/result-newhome/attack/autoattack/preactresnet50-cifar10/20220716/00000-autoattack-eps-0.05-acc-5.43/attack-cifar10-dataset/samples/test --batch_size 256 --epochs 30 --lr 0.01 >> /home/maggie/mmat/log/CIFAR10/defense/cut-mixup-training/preactresnet50-cifar10-cutmixup-autoattack-20220815.log 2>&1

# PreActResNet50 cutmixup beta(1,1) against deepfool (eps=0.02)
CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220815.py run --mode defense --defense_mode cutmixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode deepfool --attack_eps 0.02 --blackbox --exp_name preactresnet50-cifar10 --cla_model preactresnet50 --cla_network_pkl /home/data/maggie/result-newhome/defense/cutmixup/fgsm/basemixup-betasampler/preactresnet50-cifar10/blackbox/20220823/00005/cutmixup-cifar10-dataset/cutmixup-trained-classifier-preactresnet50-on-cifar10-epoch-0040-cutmix-cla6.pkl --dataset cifar10 --test_adv_dataset /home/data/maggie/result-newhome/attack/deepfool/preactresnet50-cifar10/20220708/00000-deepfool-eps-0.02-acc-12.03/attack-cifar10-dataset/samples/test --batch_size 256 --epochs 30 --lr 0.01 >> /home/maggie/mmat/log/CIFAR10/defense/cut-mixup-training/preactresnet50-cifar10-cutmixup-deepfool-20220815.log 2>&1

# PreActResNet50 cutmixup beta(1,1) against cw (eps=0.02)
CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220815.py run --mode defense --defense_mode cutmixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode cw --attack_eps 0.05 --blackbox --exp_name preactresnet50-cifar10 --cla_model preactresnet50 --cla_network_pkl /home/data/maggie/result-newhome/defense/cutmixup/fgsm/basemixup-betasampler/preactresnet50-cifar10/blackbox/20220823/00005/cutmixup-cifar10-dataset/cutmixup-trained-classifier-preactresnet50-on-cifar10-epoch-0040-cutmix-cla6.pkl --dataset cifar10 --test_adv_dataset /home/data/maggie/result-newhome/attack/cw/preactresnet50-cifar10/20220802/00000-cw-confidence-0.0-acc-1.13/attack-cifar10-dataset/samples/test --batch_size 256 --epochs 30 --lr 0.01 >> /home/maggie/mmat/log/CIFAR10/defense/cut-mixup-training/preactresnet50-cifar10-cutmixup-cw-20220815.log 2>&1

# PreActResNet50 cutmixup beta(1,1) against om-fgsm (eps=0.05)
CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220815.py run --mode defense --defense_mode cutmixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode om-fgsm --attack_eps 0.05 --blackbox --exp_name preactresnet50-cifar10 --cla_model preactresnet50 --cla_network_pkl /home/data/maggie/result-newhome/defense/cutmixup/fgsm/basemixup-betasampler/preactresnet50-cifar10/blackbox/20220823/00005/cutmixup-cifar10-dataset/cutmixup-trained-classifier-preactresnet50-on-cifar10-epoch-0040-cutmix-cla6.pkl --dataset cifar10 --test_adv_dataset /home/data/maggie/result-newhome/attack/fgsm/preactresnet50-cifar10/20220702/00001-omfgsm-eps-0.05-acc-50.26/attack-cifar10-dataset/latent-attack-samples/test --batch_size 256 --epochs 30 --lr 0.01 >> /home/maggie/mmat/log/CIFAR10/defense/cut-mixup-training/preactresnet50-cifar10-cutmixup-om-fgsm-20220815.log 2>&1

# PreActResNet50 cutmixup beta(1,1) against om-pgd (eps=0.05)
CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220815.py run --mode defense --defense_mode cutmixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode om-pgd --attack_eps 0.05 --blackbox --exp_name preactresnet50-cifar10 --cla_model preactresnet50 --cla_network_pkl /home/data/maggie/result-newhome/defense/cutmixup/fgsm/basemixup-betasampler/preactresnet50-cifar10/blackbox/20220823/00005/cutmixup-cifar10-dataset/cutmixup-trained-classifier-preactresnet50-on-cifar10-epoch-0040-cutmix-cla6.pkl --dataset cifar10 --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/preactresnet50-cifar10/20220704/00001-ompgd-eps-0.05-acc-19.13/attack-cifar10-dataset/latent-attack-samples/test --batch_size 256 --epochs 30 --lr 0.01 >> /home/maggie/mmat/log/CIFAR10/defense/cut-mixup-training/preactresnet50-cifar10-cutmixup-om-pgd-20220815.log 2>&1