source ~/.bashrc
source /home/xieyi/anaconda3/bin/activate mmat

# # adv50000 cat cle50000
# # PreActResNet18 puzzlemixup beta(1,1) against FGSM (eps=0.1)
# CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220826.py run --mode defense --defense_mode puzzlemixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode fgsm --attack_eps 0.1 --blackbox --exp_name preactresnet18-svhn --cla_model preactresnet18 --cla_network_pkl /home/data/maggie/result-newhome/train/cla-train/preactresnet18-svhn/20220627/00003-testacc-95.97/train-svhn-dataset/standard-trained-classifier-preactresnet18-on-clean-svhn-epoch-0011.pkl --dataset svhn --test_adv_dataset /home/data/maggie/result-newhome/attack/fgsm/preactresnet18-svhn/20220628/00002-fgsm-eps-0.1-acc-57.29/attack-svhn-dataset/samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/SVHN/defense/puzzle-mixup-training/preactresnet18-svhn-puzzlemixup-fgsm-20220826.log 2>&1

# # PreActResNet18 puzzlemixup beta(1,1) against FGSM (eps=0.1)
# CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220826.py run --mode defense --defense_mode puzzlemixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode fgsm --attack_eps 0.1 --blackbox --exp_name preactresnet18-svhn --cla_model preactresnet18 --cla_network_pkl /home/data/maggie/result-newhome/train/cla-train/preactresnet18-svhn/20220627/00003-testacc-95.97/train-svhn-dataset/standard-trained-classifier-preactresnet18-on-clean-svhn-epoch-0011.pkl --dataset svhn --test_adv_dataset /home/data/maggie/result-newhome/attack/fgsm/preactresnet18-svhn/20220628/00002-fgsm-eps-0.1-acc-57.29/attack-svhn-dataset/samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/SVHN/defense/puzzle-mixup-training/preactresnet18-svhn-puzzlemixup-fgsm-20220826.log 2>&1

# # PreActResNet18 puzzlemixup beta(1,1) against FGSM (eps=0.1)
# CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220826.py run --mode defense --defense_mode puzzlemixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode fgsm --attack_eps 0.1 --blackbox --exp_name preactresnet18-svhn --cla_model preactresnet18 --cla_network_pkl /home/data/maggie/result-newhome/train/cla-train/preactresnet18-svhn/20220627/00003-testacc-95.97/train-svhn-dataset/standard-trained-classifier-preactresnet18-on-clean-svhn-epoch-0011.pkl --dataset svhn --test_adv_dataset /home/data/maggie/result-newhome/attack/fgsm/preactresnet18-svhn/20220628/00002-fgsm-eps-0.1-acc-57.29/attack-svhn-dataset/samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/SVHN/defense/puzzle-mixup-training/preactresnet18-svhn-puzzlemixup-fgsm-20220826.log 2>&1

# # PreActResNet18 puzzlemixup beta(1,1) against FGSM (eps=0.1)
# CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220826.py run --mode defense --defense_mode puzzlemixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode fgsm --attack_eps 0.1 --blackbox --exp_name preactresnet18-svhn --cla_model preactresnet18 --cla_network_pkl /home/data/maggie/result-newhome/train/cla-train/preactresnet18-svhn/20220627/00003-testacc-95.97/train-svhn-dataset/standard-trained-classifier-preactresnet18-on-clean-svhn-epoch-0011.pkl --dataset svhn --test_adv_dataset /home/data/maggie/result-newhome/attack/fgsm/preactresnet18-svhn/20220628/00002-fgsm-eps-0.1-acc-57.29/attack-svhn-dataset/samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/SVHN/defense/puzzle-mixup-training/preactresnet18-svhn-puzzlemixup-fgsm-20220826.log 2>&1

# # PreActResNet18 puzzlemixup beta(1,1) against FGSM (eps=0.1)
# CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220826.py run --mode defense --defense_mode puzzlemixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode fgsm --attack_eps 0.1 --blackbox --exp_name preactresnet18-svhn --cla_model preactresnet18 --cla_network_pkl /home/data/maggie/result-newhome/train/cla-train/preactresnet18-svhn/20220627/00003-testacc-95.97/train-svhn-dataset/standard-trained-classifier-preactresnet18-on-clean-svhn-epoch-0011.pkl --dataset svhn --test_adv_dataset /home/data/maggie/result-newhome/attack/fgsm/preactresnet18-svhn/20220628/00002-fgsm-eps-0.1-acc-57.29/attack-svhn-dataset/samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/SVHN/defense/puzzle-mixup-training/preactresnet18-svhn-puzzlemixup-fgsm-20220826.log 2>&1

# # PreActResNet18 puzzlemixup beta(1,1) against FGSM (eps=0.1)
# CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220826.py run --mode defense --defense_mode puzzlemixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode fgsm --attack_eps 0.1 --blackbox --exp_name preactresnet18-svhn --cla_model preactresnet18 --cla_network_pkl /home/data/maggie/result-newhome/train/cla-train/preactresnet18-svhn/20220627/00003-testacc-95.97/train-svhn-dataset/standard-trained-classifier-preactresnet18-on-clean-svhn-epoch-0011.pkl --dataset svhn --test_adv_dataset /home/data/maggie/result-newhome/attack/fgsm/preactresnet18-svhn/20220628/00002-fgsm-eps-0.1-acc-57.29/attack-svhn-dataset/samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/SVHN/defense/puzzle-mixup-training/preactresnet18-svhn-puzzlemixup-fgsm-20220826.log 2>&1

#-------------以上要存储at训练好的模型-----------
# #   使用PGD对抗训练后的 cla1
# PreActResNet18 puzzlemixup beta(1,1) against pgd (eps=0.1)
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220826.py run --mode defense --defense_mode puzzlemixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode pgd --attack_eps 0.1 --blackbox --exp_name preactresnet18-svhn --cla_model preactresnet18 --cla_network_pkl /home/data/maggie/result-newhome/defense/puzzlemixup/fgsm/basemixup-betasampler/preactresnet18-svhn/blackbox/20220829/00000/puzzlemixup-svhn-dataset/puzzlemixup-trained-classifier-preactresnet18-on-svhn-epoch-0040-puzzle-cla1.pkl --dataset svhn --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/preactresnet18-svhn/20220628/00002-pgd-eps-0.1-acc-34.57/attack-svhn-dataset/samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/SVHN/defense/puzzle-mixup-training/preactresnet18-svhn-puzzlemixup-pgd-20220826.log 2>&1

# PreActResNet18 puzzlemixup beta(1,1) against autoattack (eps=0.1)
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220826.py run --mode defense --defense_mode puzzlemixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode autoattack --attack_eps 0.1 --blackbox --exp_name preactresnet18-svhn --cla_model preactresnet18 --cla_network_pkl /home/data/maggie/result-newhome/defense/puzzlemixup/fgsm/basemixup-betasampler/preactresnet18-svhn/blackbox/20220829/00000/puzzlemixup-svhn-dataset/puzzlemixup-trained-classifier-preactresnet18-on-svhn-epoch-0040-puzzle-cla1.pkl --dataset svhn --test_adv_dataset /home/data/maggie/result-newhome/attack/autoattack/preactresnet18-svhn/20220721/00000-autoattack-eps-0.1-acc-29.21/attack-svhn-dataset/samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/SVHN/defense/puzzle-mixup-training/preactresnet18-svhn-puzzlemixup-autoattack-20220826.log 2>&1

# PreActResNet18 puzzlemixup beta(1,1) against deepfool (eps=0.1)
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220826.py run --mode defense --defense_mode puzzlemixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode deepfool --attack_eps 0.1 --blackbox --exp_name preactresnet18-svhn --cla_model preactresnet18 --cla_network_pkl /home/data/maggie/result-newhome/defense/puzzlemixup/fgsm/basemixup-betasampler/preactresnet18-svhn/blackbox/20220829/00000/puzzlemixup-svhn-dataset/puzzlemixup-trained-classifier-preactresnet18-on-svhn-epoch-0040-puzzle-cla1.pkl --dataset svhn --test_adv_dataset /home/data/maggie/result-newhome/attack/deepfool/preactresnet18-svhn/20220718/00000-deepfool-eps-0.1-acc-22.51/attack-svhn-dataset/samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/SVHN/defense/puzzle-mixup-training/preactresnet18-svhn-puzzlemixup-deepfool-20220826.log 2>&1

# PreActResNet18 puzzlemixup beta(1,1) against cw (eps=0.1)
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220826.py run --mode defense --defense_mode puzzlemixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode cw --attack_eps 0.1 --blackbox --exp_name preactresnet18-svhn --cla_model preactresnet18 --cla_network_pkl /home/data/maggie/result-newhome/defense/puzzlemixup/fgsm/basemixup-betasampler/preactresnet18-svhn/blackbox/20220829/00000/puzzlemixup-svhn-dataset/puzzlemixup-trained-classifier-preactresnet18-on-svhn-epoch-0040-puzzle-cla1.pkl --dataset svhn --test_adv_dataset /home/data/maggie/result-newhome/attack/cw/preactresnet18-svhn/20220630/00000-cw-confidence-0.0-acc-21.54/attack-svhn-dataset/samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/SVHN/defense/puzzle-mixup-training/preactresnet18-svhn-puzzlemixup-cw-20220826.log 2>&1

# PreActResNet18 puzzlemixup beta(1,1) against om-fgsm (eps=0.1)
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220826.py run --mode defense --defense_mode puzzlemixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode om-fgsm --attack_eps 0.1 --blackbox --exp_name preactresnet18-svhn --cla_model preactresnet18 --cla_network_pkl /home/data/maggie/result-newhome/defense/puzzlemixup/fgsm/basemixup-betasampler/preactresnet18-svhn/blackbox/20220829/00000/puzzlemixup-svhn-dataset/puzzlemixup-trained-classifier-preactresnet18-on-svhn-epoch-0040-puzzle-cla1.pkl --dataset svhn --test_adv_dataset /home/data/maggie/result-newhome/attack/fgsm/preactresnet18-svhn/20220710/00002-omfgsm-eps-0.1-acc-41.04/attack-svhn-dataset/latent-attack-samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/SVHN/defense/puzzle-mixup-training/preactresnet18-svhn-puzzlemixup-om-fgsm-20220826.log 2>&1

# PreActResNet18 puzzlemixup beta(1,1) against om-pgd (eps=0.1)
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220826.py run --mode defense --defense_mode puzzlemixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode om-pgd --attack_eps 0.1 --blackbox --exp_name preactresnet18-svhn --cla_model preactresnet18 --cla_network_pkl /home/data/maggie/result-newhome/defense/puzzlemixup/fgsm/basemixup-betasampler/preactresnet18-svhn/blackbox/20220829/00000/puzzlemixup-svhn-dataset/puzzlemixup-trained-classifier-preactresnet18-on-svhn-epoch-0040-puzzle-cla1.pkl --dataset svhn --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/preactresnet18-svhn/20220711/00002-ompgd-eps-0.1-acc-6.78/attack-svhn-dataset/latent-attack-samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/SVHN/defense/puzzle-mixup-training/preactresnet18-svhn-puzzlemixup-om-pgd-20220826.log 2>&1

# #   使用PGD对抗训练后的 cla2
# PreActResNet18 puzzlemixup beta(1,1) against pgd (eps=0.1)
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220826.py run --mode defense --defense_mode puzzlemixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode pgd --attack_eps 0.1 --blackbox --exp_name preactresnet18-svhn --cla_model preactresnet18 --cla_network_pkl /home/data/maggie/result-newhome/defense/puzzlemixup/fgsm/basemixup-betasampler/preactresnet18-svhn/blackbox/20220829/00001/puzzlemixup-svhn-dataset/puzzlemixup-trained-classifier-preactresnet18-on-svhn-epoch-0040-puzzle-cla2.pkl --dataset svhn --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/preactresnet18-svhn/20220628/00002-pgd-eps-0.1-acc-34.57/attack-svhn-dataset/samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/SVHN/defense/puzzle-mixup-training/preactresnet18-svhn-puzzlemixup-pgd-20220826.log 2>&1

# PreActResNet18 puzzlemixup beta(1,1) against autoattack (eps=0.1)
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220826.py run --mode defense --defense_mode puzzlemixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode autoattack --attack_eps 0.1 --blackbox --exp_name preactresnet18-svhn --cla_model preactresnet18 --cla_network_pkl /home/data/maggie/result-newhome/defense/puzzlemixup/fgsm/basemixup-betasampler/preactresnet18-svhn/blackbox/20220829/00001/puzzlemixup-svhn-dataset/puzzlemixup-trained-classifier-preactresnet18-on-svhn-epoch-0040-puzzle-cla2.pkl --dataset svhn --test_adv_dataset /home/data/maggie/result-newhome/attack/autoattack/preactresnet18-svhn/20220721/00000-autoattack-eps-0.1-acc-29.21/attack-svhn-dataset/samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/SVHN/defense/puzzle-mixup-training/preactresnet18-svhn-puzzlemixup-autoattack-20220826.log 2>&1

# PreActResNet18 puzzlemixup beta(1,1) against deepfool (eps=0.1)
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220826.py run --mode defense --defense_mode puzzlemixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode deepfool --attack_eps 0.1 --blackbox --exp_name preactresnet18-svhn --cla_model preactresnet18 --cla_network_pkl /home/data/maggie/result-newhome/defense/puzzlemixup/fgsm/basemixup-betasampler/preactresnet18-svhn/blackbox/20220829/00001/puzzlemixup-svhn-dataset/puzzlemixup-trained-classifier-preactresnet18-on-svhn-epoch-0040-puzzle-cla2.pkl --dataset svhn --test_adv_dataset /home/data/maggie/result-newhome/attack/deepfool/preactresnet18-svhn/20220718/00000-deepfool-eps-0.1-acc-22.51/attack-svhn-dataset/samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/SVHN/defense/puzzle-mixup-training/preactresnet18-svhn-puzzlemixup-deepfool-20220826.log 2>&1

# PreActResNet18 puzzlemixup beta(1,1) against cw (eps=0.1)
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220826.py run --mode defense --defense_mode puzzlemixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode cw --attack_eps 0.1 --blackbox --exp_name preactresnet18-svhn --cla_model preactresnet18 --cla_network_pkl /home/data/maggie/result-newhome/defense/puzzlemixup/fgsm/basemixup-betasampler/preactresnet18-svhn/blackbox/20220829/00001/puzzlemixup-svhn-dataset/puzzlemixup-trained-classifier-preactresnet18-on-svhn-epoch-0040-puzzle-cla2.pkl --dataset svhn --test_adv_dataset /home/data/maggie/result-newhome/attack/cw/preactresnet18-svhn/20220630/00000-cw-confidence-0.0-acc-21.54/attack-svhn-dataset/samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/SVHN/defense/puzzle-mixup-training/preactresnet18-svhn-puzzlemixup-cw-20220826.log 2>&1

# PreActResNet18 puzzlemixup beta(1,1) against om-fgsm (eps=0.1)
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220826.py run --mode defense --defense_mode puzzlemixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode om-fgsm --attack_eps 0.1 --blackbox --exp_name preactresnet18-svhn --cla_model preactresnet18 --cla_network_pkl /home/data/maggie/result-newhome/defense/puzzlemixup/fgsm/basemixup-betasampler/preactresnet18-svhn/blackbox/20220829/00001/puzzlemixup-svhn-dataset/puzzlemixup-trained-classifier-preactresnet18-on-svhn-epoch-0040-puzzle-cla2.pkl --dataset svhn --test_adv_dataset /home/data/maggie/result-newhome/attack/fgsm/preactresnet18-svhn/20220710/00002-omfgsm-eps-0.1-acc-41.04/attack-svhn-dataset/latent-attack-samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/SVHN/defense/puzzle-mixup-training/preactresnet18-svhn-puzzlemixup-om-fgsm-20220826.log 2>&1

# PreActResNet18 puzzlemixup beta(1,1) against om-pgd (eps=0.1)
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220826.py run --mode defense --defense_mode puzzlemixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode om-pgd --attack_eps 0.1 --blackbox --exp_name preactresnet18-svhn --cla_model preactresnet18 --cla_network_pkl /home/data/maggie/result-newhome/defense/puzzlemixup/fgsm/basemixup-betasampler/preactresnet18-svhn/blackbox/20220829/00001/puzzlemixup-svhn-dataset/puzzlemixup-trained-classifier-preactresnet18-on-svhn-epoch-0040-puzzle-cla2.pkl --dataset svhn --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/preactresnet18-svhn/20220711/00002-ompgd-eps-0.1-acc-6.78/attack-svhn-dataset/latent-attack-samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/SVHN/defense/puzzle-mixup-training/preactresnet18-svhn-puzzlemixup-om-pgd-20220826.log 2>&1

# #   使用PGD对抗训练后的 cla3
# PreActResNet18 puzzlemixup beta(1,1) against pgd (eps=0.1)
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220826.py run --mode defense --defense_mode puzzlemixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode pgd --attack_eps 0.1 --blackbox --exp_name preactresnet18-svhn --cla_model preactresnet18 --cla_network_pkl /home/data/maggie/result-newhome/defense/puzzlemixup/fgsm/basemixup-betasampler/preactresnet18-svhn/blackbox/20220829/00002/puzzlemixup-svhn-dataset/puzzlemixup-trained-classifier-preactresnet18-on-svhn-epoch-0040-puzzle-cla3.pkl --dataset svhn --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/preactresnet18-svhn/20220628/00002-pgd-eps-0.1-acc-34.57/attack-svhn-dataset/samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/SVHN/defense/puzzle-mixup-training/preactresnet18-svhn-puzzlemixup-pgd-20220826.log 2>&1

# PreActResNet18 puzzlemixup beta(1,1) against autoattack (eps=0.1)
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220826.py run --mode defense --defense_mode puzzlemixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode autoattack --attack_eps 0.1 --blackbox --exp_name preactresnet18-svhn --cla_model preactresnet18 --cla_network_pkl /home/data/maggie/result-newhome/defense/puzzlemixup/fgsm/basemixup-betasampler/preactresnet18-svhn/blackbox/20220829/00002/puzzlemixup-svhn-dataset/puzzlemixup-trained-classifier-preactresnet18-on-svhn-epoch-0040-puzzle-cla3.pkl --dataset svhn --test_adv_dataset /home/data/maggie/result-newhome/attack/autoattack/preactresnet18-svhn/20220721/00000-autoattack-eps-0.1-acc-29.21/attack-svhn-dataset/samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/SVHN/defense/puzzle-mixup-training/preactresnet18-svhn-puzzlemixup-autoattack-20220826.log 2>&1

# PreActResNet18 puzzlemixup beta(1,1) against deepfool (eps=0.1)
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220826.py run --mode defense --defense_mode puzzlemixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode deepfool --attack_eps 0.1 --blackbox --exp_name preactresnet18-svhn --cla_model preactresnet18 --cla_network_pkl /home/data/maggie/result-newhome/defense/puzzlemixup/fgsm/basemixup-betasampler/preactresnet18-svhn/blackbox/20220829/00002/puzzlemixup-svhn-dataset/puzzlemixup-trained-classifier-preactresnet18-on-svhn-epoch-0040-puzzle-cla3.pkl --dataset svhn --test_adv_dataset /home/data/maggie/result-newhome/attack/deepfool/preactresnet18-svhn/20220718/00000-deepfool-eps-0.1-acc-22.51/attack-svhn-dataset/samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/SVHN/defense/puzzle-mixup-training/preactresnet18-svhn-puzzlemixup-deepfool-20220826.log 2>&1

# PreActResNet18 puzzlemixup beta(1,1) against cw (eps=0.1)
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220826.py run --mode defense --defense_mode puzzlemixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode cw --attack_eps 0.1 --blackbox --exp_name preactresnet18-svhn --cla_model preactresnet18 --cla_network_pkl /home/data/maggie/result-newhome/defense/puzzlemixup/fgsm/basemixup-betasampler/preactresnet18-svhn/blackbox/20220829/00002/puzzlemixup-svhn-dataset/puzzlemixup-trained-classifier-preactresnet18-on-svhn-epoch-0040-puzzle-cla3.pkl --dataset svhn --test_adv_dataset /home/data/maggie/result-newhome/attack/cw/preactresnet18-svhn/20220630/00000-cw-confidence-0.0-acc-21.54/attack-svhn-dataset/samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/SVHN/defense/puzzle-mixup-training/preactresnet18-svhn-puzzlemixup-cw-20220826.log 2>&1

# PreActResNet18 puzzlemixup beta(1,1) against om-fgsm (eps=0.1)
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220826.py run --mode defense --defense_mode puzzlemixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode om-fgsm --attack_eps 0.1 --blackbox --exp_name preactresnet18-svhn --cla_model preactresnet18 --cla_network_pkl /home/data/maggie/result-newhome/defense/puzzlemixup/fgsm/basemixup-betasampler/preactresnet18-svhn/blackbox/20220829/00002/puzzlemixup-svhn-dataset/puzzlemixup-trained-classifier-preactresnet18-on-svhn-epoch-0040-puzzle-cla3.pkl --dataset svhn --test_adv_dataset /home/data/maggie/result-newhome/attack/fgsm/preactresnet18-svhn/20220710/00002-omfgsm-eps-0.1-acc-41.04/attack-svhn-dataset/latent-attack-samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/SVHN/defense/puzzle-mixup-training/preactresnet18-svhn-puzzlemixup-om-fgsm-20220826.log 2>&1

# PreActResNet18 puzzlemixup beta(1,1) against om-pgd (eps=0.1)
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220826.py run --mode defense --defense_mode puzzlemixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode om-pgd --attack_eps 0.1 --blackbox --exp_name preactresnet18-svhn --cla_model preactresnet18 --cla_network_pkl /home/data/maggie/result-newhome/defense/puzzlemixup/fgsm/basemixup-betasampler/preactresnet18-svhn/blackbox/20220829/00002/puzzlemixup-svhn-dataset/puzzlemixup-trained-classifier-preactresnet18-on-svhn-epoch-0040-puzzle-cla3.pkl --dataset svhn --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/preactresnet18-svhn/20220711/00002-ompgd-eps-0.1-acc-6.78/attack-svhn-dataset/latent-attack-samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/SVHN/defense/puzzle-mixup-training/preactresnet18-svhn-puzzlemixup-om-pgd-20220826.log 2>&1

# #   使用PGD对抗训练后的 cla4
# PreActResNet18 puzzlemixup beta(1,1) against pgd (eps=0.1)
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220826.py run --mode defense --defense_mode puzzlemixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode pgd --attack_eps 0.1 --blackbox --exp_name preactresnet18-svhn --cla_model preactresnet18 --cla_network_pkl /home/data/maggie/result-newhome/defense/puzzlemixup/fgsm/basemixup-betasampler/preactresnet18-svhn/blackbox/20220829/00003/puzzlemixup-svhn-dataset/puzzlemixup-trained-classifier-preactresnet18-on-svhn-epoch-0040-puzzle-cla4.pkl --dataset svhn --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/preactresnet18-svhn/20220628/00002-pgd-eps-0.1-acc-34.57/attack-svhn-dataset/samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/SVHN/defense/puzzle-mixup-training/preactresnet18-svhn-puzzlemixup-pgd-20220826.log 2>&1

# PreActResNet18 puzzlemixup beta(1,1) against autoattack (eps=0.1)
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220826.py run --mode defense --defense_mode puzzlemixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode autoattack --attack_eps 0.1 --blackbox --exp_name preactresnet18-svhn --cla_model preactresnet18 --cla_network_pkl /home/data/maggie/result-newhome/defense/puzzlemixup/fgsm/basemixup-betasampler/preactresnet18-svhn/blackbox/20220829/00003/puzzlemixup-svhn-dataset/puzzlemixup-trained-classifier-preactresnet18-on-svhn-epoch-0040-puzzle-cla4.pkl --dataset svhn --test_adv_dataset /home/data/maggie/result-newhome/attack/autoattack/preactresnet18-svhn/20220721/00000-autoattack-eps-0.1-acc-29.21/attack-svhn-dataset/samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/SVHN/defense/puzzle-mixup-training/preactresnet18-svhn-puzzlemixup-autoattack-20220826.log 2>&1

# PreActResNet18 puzzlemixup beta(1,1) against deepfool (eps=0.1)
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220826.py run --mode defense --defense_mode puzzlemixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode deepfool --attack_eps 0.1 --blackbox --exp_name preactresnet18-svhn --cla_model preactresnet18 --cla_network_pkl /home/data/maggie/result-newhome/defense/puzzlemixup/fgsm/basemixup-betasampler/preactresnet18-svhn/blackbox/20220829/00003/puzzlemixup-svhn-dataset/puzzlemixup-trained-classifier-preactresnet18-on-svhn-epoch-0040-puzzle-cla4.pkl --dataset svhn --test_adv_dataset /home/data/maggie/result-newhome/attack/deepfool/preactresnet18-svhn/20220718/00000-deepfool-eps-0.1-acc-22.51/attack-svhn-dataset/samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/SVHN/defense/puzzle-mixup-training/preactresnet18-svhn-puzzlemixup-deepfool-20220826.log 2>&1

# PreActResNet18 puzzlemixup beta(1,1) against cw (eps=0.1)
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220826.py run --mode defense --defense_mode puzzlemixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode cw --attack_eps 0.1 --blackbox --exp_name preactresnet18-svhn --cla_model preactresnet18 --cla_network_pkl /home/data/maggie/result-newhome/defense/puzzlemixup/fgsm/basemixup-betasampler/preactresnet18-svhn/blackbox/20220829/00003/puzzlemixup-svhn-dataset/puzzlemixup-trained-classifier-preactresnet18-on-svhn-epoch-0040-puzzle-cla4.pkl --dataset svhn --test_adv_dataset /home/data/maggie/result-newhome/attack/cw/preactresnet18-svhn/20220630/00000-cw-confidence-0.0-acc-21.54/attack-svhn-dataset/samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/SVHN/defense/puzzle-mixup-training/preactresnet18-svhn-puzzlemixup-cw-20220826.log 2>&1

# PreActResNet18 puzzlemixup beta(1,1) against om-fgsm (eps=0.1)
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220826.py run --mode defense --defense_mode puzzlemixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode om-fgsm --attack_eps 0.1 --blackbox --exp_name preactresnet18-svhn --cla_model preactresnet18 --cla_network_pkl /home/data/maggie/result-newhome/defense/puzzlemixup/fgsm/basemixup-betasampler/preactresnet18-svhn/blackbox/20220829/00003/puzzlemixup-svhn-dataset/puzzlemixup-trained-classifier-preactresnet18-on-svhn-epoch-0040-puzzle-cla4.pkl --dataset svhn --test_adv_dataset /home/data/maggie/result-newhome/attack/fgsm/preactresnet18-svhn/20220710/00002-omfgsm-eps-0.1-acc-41.04/attack-svhn-dataset/latent-attack-samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/SVHN/defense/puzzle-mixup-training/preactresnet18-svhn-puzzlemixup-om-fgsm-20220826.log 2>&1

# PreActResNet18 puzzlemixup beta(1,1) against om-pgd (eps=0.1)
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220826.py run --mode defense --defense_mode puzzlemixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode om-pgd --attack_eps 0.1 --blackbox --exp_name preactresnet18-svhn --cla_model preactresnet18 --cla_network_pkl /home/data/maggie/result-newhome/defense/puzzlemixup/fgsm/basemixup-betasampler/preactresnet18-svhn/blackbox/20220829/00003/puzzlemixup-svhn-dataset/puzzlemixup-trained-classifier-preactresnet18-on-svhn-epoch-0040-puzzle-cla4.pkl --dataset svhn --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/preactresnet18-svhn/20220711/00002-ompgd-eps-0.1-acc-6.78/attack-svhn-dataset/latent-attack-samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/SVHN/defense/puzzle-mixup-training/preactresnet18-svhn-puzzlemixup-om-pgd-20220826.log 2>&1

# #   使用PGD对抗训练后的 cla5
# PreActResNet18 puzzlemixup beta(1,1) against pgd (eps=0.1)
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220826.py run --mode defense --defense_mode puzzlemixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode pgd --attack_eps 0.1 --blackbox --exp_name preactresnet18-svhn --cla_model preactresnet18 --cla_network_pkl /home/data/maggie/result-newhome/defense/puzzlemixup/fgsm/basemixup-betasampler/preactresnet18-svhn/blackbox/20220829/00005/puzzlemixup-svhn-dataset/puzzlemixup-trained-classifier-preactresnet18-on-svhn-epoch-0040-puzzle-cla5.pkl --dataset svhn --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/preactresnet18-svhn/20220628/00002-pgd-eps-0.1-acc-34.57/attack-svhn-dataset/samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/SVHN/defense/puzzle-mixup-training/preactresnet18-svhn-puzzlemixup-pgd-20220826.log 2>&1

# PreActResNet18 puzzlemixup beta(1,1) against autoattack (eps=0.1)
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220826.py run --mode defense --defense_mode puzzlemixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode autoattack --attack_eps 0.1 --blackbox --exp_name preactresnet18-svhn --cla_model preactresnet18 --cla_network_pkl /home/data/maggie/result-newhome/defense/puzzlemixup/fgsm/basemixup-betasampler/preactresnet18-svhn/blackbox/20220829/00005/puzzlemixup-svhn-dataset/puzzlemixup-trained-classifier-preactresnet18-on-svhn-epoch-0040-puzzle-cla5.pkl --dataset svhn --test_adv_dataset /home/data/maggie/result-newhome/attack/autoattack/preactresnet18-svhn/20220721/00000-autoattack-eps-0.1-acc-29.21/attack-svhn-dataset/samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/SVHN/defense/puzzle-mixup-training/preactresnet18-svhn-puzzlemixup-autoattack-20220826.log 2>&1

# PreActResNet18 puzzlemixup beta(1,1) against deepfool (eps=0.1)
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220826.py run --mode defense --defense_mode puzzlemixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode deepfool --attack_eps 0.1 --blackbox --exp_name preactresnet18-svhn --cla_model preactresnet18 --cla_network_pkl /home/data/maggie/result-newhome/defense/puzzlemixup/fgsm/basemixup-betasampler/preactresnet18-svhn/blackbox/20220829/00005/puzzlemixup-svhn-dataset/puzzlemixup-trained-classifier-preactresnet18-on-svhn-epoch-0040-puzzle-cla5.pkl --dataset svhn --test_adv_dataset /home/data/maggie/result-newhome/attack/deepfool/preactresnet18-svhn/20220718/00000-deepfool-eps-0.1-acc-22.51/attack-svhn-dataset/samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/SVHN/defense/puzzle-mixup-training/preactresnet18-svhn-puzzlemixup-deepfool-20220826.log 2>&1

# PreActResNet18 puzzlemixup beta(1,1) against cw (eps=0.1)
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220826.py run --mode defense --defense_mode puzzlemixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode cw --attack_eps 0.1 --blackbox --exp_name preactresnet18-svhn --cla_model preactresnet18 --cla_network_pkl /home/data/maggie/result-newhome/defense/puzzlemixup/fgsm/basemixup-betasampler/preactresnet18-svhn/blackbox/20220829/00005/puzzlemixup-svhn-dataset/puzzlemixup-trained-classifier-preactresnet18-on-svhn-epoch-0040-puzzle-cla5.pkl --dataset svhn --test_adv_dataset /home/data/maggie/result-newhome/attack/cw/preactresnet18-svhn/20220630/00000-cw-confidence-0.0-acc-21.54/attack-svhn-dataset/samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/SVHN/defense/puzzle-mixup-training/preactresnet18-svhn-puzzlemixup-cw-20220826.log 2>&1

# PreActResNet18 puzzlemixup beta(1,1) against om-fgsm (eps=0.1)
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220826.py run --mode defense --defense_mode puzzlemixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode om-fgsm --attack_eps 0.1 --blackbox --exp_name preactresnet18-svhn --cla_model preactresnet18 --cla_network_pkl /home/data/maggie/result-newhome/defense/puzzlemixup/fgsm/basemixup-betasampler/preactresnet18-svhn/blackbox/20220829/00005/puzzlemixup-svhn-dataset/puzzlemixup-trained-classifier-preactresnet18-on-svhn-epoch-0040-puzzle-cla5.pkl --dataset svhn --test_adv_dataset /home/data/maggie/result-newhome/attack/fgsm/preactresnet18-svhn/20220710/00002-omfgsm-eps-0.1-acc-41.04/attack-svhn-dataset/latent-attack-samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/SVHN/defense/puzzle-mixup-training/preactresnet18-svhn-puzzlemixup-om-fgsm-20220826.log 2>&1

# PreActResNet18 puzzlemixup beta(1,1) against om-pgd (eps=0.1)
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220826.py run --mode defense --defense_mode puzzlemixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode om-pgd --attack_eps 0.1 --blackbox --exp_name preactresnet18-svhn --cla_model preactresnet18 --cla_network_pkl /home/data/maggie/result-newhome/defense/puzzlemixup/fgsm/basemixup-betasampler/preactresnet18-svhn/blackbox/20220829/00005/puzzlemixup-svhn-dataset/puzzlemixup-trained-classifier-preactresnet18-on-svhn-epoch-0040-puzzle-cla5.pkl --dataset svhn --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/preactresnet18-svhn/20220711/00002-ompgd-eps-0.1-acc-6.78/attack-svhn-dataset/latent-attack-samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/SVHN/defense/puzzle-mixup-training/preactresnet18-svhn-puzzlemixup-om-pgd-20220826.log 2>&1

# #   使用PGD对抗训练后的 cla6
# PreActResNet18 puzzlemixup beta(1,1) against pgd (eps=0.1)
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220826.py run --mode defense --defense_mode puzzlemixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode pgd --attack_eps 0.1 --blackbox --exp_name preactresnet18-svhn --cla_model preactresnet18 --cla_network_pkl /home/data/maggie/result-newhome/defense/puzzlemixup/fgsm/basemixup-betasampler/preactresnet18-svhn/blackbox/20220829/00006/puzzlemixup-svhn-dataset/puzzlemixup-trained-classifier-preactresnet18-on-svhn-epoch-0040-puzzle-cla6.pkl --dataset svhn --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/preactresnet18-svhn/20220628/00002-pgd-eps-0.1-acc-34.57/attack-svhn-dataset/samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/SVHN/defense/puzzle-mixup-training/preactresnet18-svhn-puzzlemixup-pgd-20220826.log 2>&1

# PreActResNet18 puzzlemixup beta(1,1) against autoattack (eps=0.1)
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220826.py run --mode defense --defense_mode puzzlemixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode autoattack --attack_eps 0.1 --blackbox --exp_name preactresnet18-svhn --cla_model preactresnet18 --cla_network_pkl /home/data/maggie/result-newhome/defense/puzzlemixup/fgsm/basemixup-betasampler/preactresnet18-svhn/blackbox/20220829/00006/puzzlemixup-svhn-dataset/puzzlemixup-trained-classifier-preactresnet18-on-svhn-epoch-0040-puzzle-cla6.pkl --dataset svhn --test_adv_dataset /home/data/maggie/result-newhome/attack/autoattack/preactresnet18-svhn/20220721/00000-autoattack-eps-0.1-acc-29.21/attack-svhn-dataset/samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/SVHN/defense/puzzle-mixup-training/preactresnet18-svhn-puzzlemixup-autoattack-20220826.log 2>&1

# PreActResNet18 puzzlemixup beta(1,1) against deepfool (eps=0.1)
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220826.py run --mode defense --defense_mode puzzlemixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode deepfool --attack_eps 0.1 --blackbox --exp_name preactresnet18-svhn --cla_model preactresnet18 --cla_network_pkl /home/data/maggie/result-newhome/defense/puzzlemixup/fgsm/basemixup-betasampler/preactresnet18-svhn/blackbox/20220829/00006/puzzlemixup-svhn-dataset/puzzlemixup-trained-classifier-preactresnet18-on-svhn-epoch-0040-puzzle-cla6.pkl --dataset svhn --test_adv_dataset /home/data/maggie/result-newhome/attack/deepfool/preactresnet18-svhn/20220718/00000-deepfool-eps-0.1-acc-22.51/attack-svhn-dataset/samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/SVHN/defense/puzzle-mixup-training/preactresnet18-svhn-puzzlemixup-deepfool-20220826.log 2>&1

# PreActResNet18 puzzlemixup beta(1,1) against cw (eps=0.1)
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220826.py run --mode defense --defense_mode puzzlemixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode cw --attack_eps 0.1 --blackbox --exp_name preactresnet18-svhn --cla_model preactresnet18 --cla_network_pkl /home/data/maggie/result-newhome/defense/puzzlemixup/fgsm/basemixup-betasampler/preactresnet18-svhn/blackbox/20220829/00006/puzzlemixup-svhn-dataset/puzzlemixup-trained-classifier-preactresnet18-on-svhn-epoch-0040-puzzle-cla6.pkl --dataset svhn --test_adv_dataset /home/data/maggie/result-newhome/attack/cw/preactresnet18-svhn/20220630/00000-cw-confidence-0.0-acc-21.54/attack-svhn-dataset/samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/SVHN/defense/puzzle-mixup-training/preactresnet18-svhn-puzzlemixup-cw-20220826.log 2>&1

# PreActResNet18 puzzlemixup beta(1,1) against om-fgsm (eps=0.1)
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220826.py run --mode defense --defense_mode puzzlemixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode om-fgsm --attack_eps 0.1 --blackbox --exp_name preactresnet18-svhn --cla_model preactresnet18 --cla_network_pkl /home/data/maggie/result-newhome/defense/puzzlemixup/fgsm/basemixup-betasampler/preactresnet18-svhn/blackbox/20220829/00006/puzzlemixup-svhn-dataset/puzzlemixup-trained-classifier-preactresnet18-on-svhn-epoch-0040-puzzle-cla6.pkl --dataset svhn --test_adv_dataset /home/data/maggie/result-newhome/attack/fgsm/preactresnet18-svhn/20220710/00002-omfgsm-eps-0.1-acc-41.04/attack-svhn-dataset/latent-attack-samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/SVHN/defense/puzzle-mixup-training/preactresnet18-svhn-puzzlemixup-om-fgsm-20220826.log 2>&1

# PreActResNet18 puzzlemixup beta(1,1) against om-pgd (eps=0.1)
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220826.py run --mode defense --defense_mode puzzlemixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode om-pgd --attack_eps 0.1 --blackbox --exp_name preactresnet18-svhn --cla_model preactresnet18 --cla_network_pkl /home/data/maggie/result-newhome/defense/puzzlemixup/fgsm/basemixup-betasampler/preactresnet18-svhn/blackbox/20220829/00006/puzzlemixup-svhn-dataset/puzzlemixup-trained-classifier-preactresnet18-on-svhn-epoch-0040-puzzle-cla6.pkl --dataset svhn --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/preactresnet18-svhn/20220711/00002-ompgd-eps-0.1-acc-6.78/attack-svhn-dataset/latent-attack-samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/SVHN/defense/puzzle-mixup-training/preactresnet18-svhn-puzzlemixup-om-pgd-20220826.log 2>&1