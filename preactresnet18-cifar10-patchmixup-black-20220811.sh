source ~/.bashrc
source /home/xieyi/anaconda3/bin/activate mmat

# preactresnet18 patchmixup beta(1,1) defense against fgsm(eps=0.05)
CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220812.py run --mode defense --defense_mode patchmixup --mix_mode maskmixup --sample_mode bernoullisampler --attack_mode fgsm --attack_eps 0.05 --blackbox --exp_name preactresnet18-cifar10 --cla_model preactresnet18 --cla_network_pkl /home/data/maggie/result-newhome/train/cla-train/preactresnet18-cifar10/20210924/00001-teatacc-87.37/train-cifar10-dataset/standard-trained-classifier-preactresnet18-on-clean-cifar10-epoch-0011.pkl --dataset cifar10 --test_adv_dataset /home/data/maggie/result-newhome/attack/fgsm/preactresnet18-cifar10/20220627/00001-fgsm-eps-0.05-acc-32.07/attack-cifar10-dataset/samples/test --batch_size 256 --epochs 40 --lr 0.001 >> /home/maggie/mmat/log/CIFAR10/defense/patch-mixup-training/preactresnet18-cifar10-patchmixup-fgsm-20220812.log 2>&1

CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220812.py run --mode defense --defense_mode patchmixup --mix_mode maskmixup --sample_mode bernoullisampler --attack_mode fgsm --attack_eps 0.05 --blackbox --exp_name preactresnet18-cifar10 --cla_model preactresnet18 --cla_network_pkl /home/data/maggie/result-newhome/train/cla-train/preactresnet18-cifar10/20210924/00001-teatacc-87.37/train-cifar10-dataset/standard-trained-classifier-preactresnet18-on-clean-cifar10-epoch-0011.pkl --dataset cifar10 --test_adv_dataset /home/data/maggie/result-newhome/attack/fgsm/preactresnet18-cifar10/20220627/00001-fgsm-eps-0.05-acc-32.07/attack-cifar10-dataset/samples/test --batch_size 256 --epochs 40 --lr 0.001 >> /home/maggie/mmat/log/CIFAR10/defense/patch-mixup-training/preactresnet18-cifar10-patchmixup-fgsm-20220812.log 2>&1

CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220812.py run --mode defense --defense_mode patchmixup --mix_mode maskmixup --sample_mode bernoullisampler --attack_mode fgsm --attack_eps 0.05 --blackbox --exp_name preactresnet18-cifar10 --cla_model preactresnet18 --cla_network_pkl /home/data/maggie/result-newhome/train/cla-train/preactresnet18-cifar10/20210924/00001-teatacc-87.37/train-cifar10-dataset/standard-trained-classifier-preactresnet18-on-clean-cifar10-epoch-0011.pkl --dataset cifar10 --test_adv_dataset /home/data/maggie/result-newhome/attack/fgsm/preactresnet18-cifar10/20220627/00001-fgsm-eps-0.05-acc-32.07/attack-cifar10-dataset/samples/test --batch_size 256 --epochs 40 --lr 0.001 >> /home/maggie/mmat/log/CIFAR10/defense/patch-mixup-training/preactresnet18-cifar10-patchmixup-fgsm-20220812.log 2>&1

CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220812.py run --mode defense --defense_mode patchmixup --mix_mode maskmixup --sample_mode bernoullisampler --attack_mode fgsm --attack_eps 0.05 --blackbox --exp_name preactresnet18-cifar10 --cla_model preactresnet18 --cla_network_pkl /home/data/maggie/result-newhome/train/cla-train/preactresnet18-cifar10/20210924/00001-teatacc-87.37/train-cifar10-dataset/standard-trained-classifier-preactresnet18-on-clean-cifar10-epoch-0011.pkl --dataset cifar10 --test_adv_dataset /home/data/maggie/result-newhome/attack/fgsm/preactresnet18-cifar10/20220627/00001-fgsm-eps-0.05-acc-32.07/attack-cifar10-dataset/samples/test --batch_size 256 --epochs 40 --lr 0.001 >> /home/maggie/mmat/log/CIFAR10/defense/patch-mixup-training/preactresnet18-cifar10-patchmixup-fgsm-20220812.log 2>&1

CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220812.py run --mode defense --defense_mode patchmixup --mix_mode maskmixup --sample_mode bernoullisampler --attack_mode fgsm --attack_eps 0.05 --blackbox --exp_name preactresnet18-cifar10 --cla_model preactresnet18 --cla_network_pkl /home/data/maggie/result-newhome/train/cla-train/preactresnet18-cifar10/20210924/00001-teatacc-87.37/train-cifar10-dataset/standard-trained-classifier-preactresnet18-on-clean-cifar10-epoch-0011.pkl --dataset cifar10 --test_adv_dataset /home/data/maggie/result-newhome/attack/fgsm/preactresnet18-cifar10/20220627/00001-fgsm-eps-0.05-acc-32.07/attack-cifar10-dataset/samples/test --batch_size 256 --epochs 40 --lr 0.001 >> /home/maggie/mmat/log/CIFAR10/defense/patch-mixup-training/preactresnet18-cifar10-patchmixup-fgsm-20220812.log 2>&1

CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220812.py run --mode defense --defense_mode patchmixup --mix_mode maskmixup --sample_mode bernoullisampler --attack_mode fgsm --attack_eps 0.05 --blackbox --exp_name preactresnet18-cifar10 --cla_model preactresnet18 --cla_network_pkl /home/data/maggie/result-newhome/train/cla-train/preactresnet18-cifar10/20210924/00001-teatacc-87.37/train-cifar10-dataset/standard-trained-classifier-preactresnet18-on-clean-cifar10-epoch-0011.pkl --dataset cifar10 --test_adv_dataset /home/data/maggie/result-newhome/attack/fgsm/preactresnet18-cifar10/20220627/00001-fgsm-eps-0.05-acc-32.07/attack-cifar10-dataset/samples/test --batch_size 256 --epochs 40 --lr 0.001 >> /home/maggie/mmat/log/CIFAR10/defense/patch-mixup-training/preactresnet18-cifar10-patchmixup-fgsm-20220812.log 2>&1

# #-------------以上要存储at训练好的模型-----------
# #--------------------cla-1---------------------
# # preactresnet18 patchmixup beta(1,1) defense for pgd(eps=0.05)
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220812.py run --mode defense --defense_mode patchmixup --mix_mode maskmixup --sample_mode bernoullisampler --attack_mode pgd --attack_eps 0.05 --blackbox --exp_name preactresnet18-cifar10 --cla_model preactresnet18 --cla_network_pkl XXX-cla1-XXX --dataset cifar10 --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/preactresnet18-cifar10/20220624/00001-pgd-eps-0.05-acc-28.93/attack-cifar10-dataset/samples/test --batch_size 256 --epochs 40 --lr 0.001 >> /home/maggie/mmat/log/CIFAR10/defense/patch-mixup-training/preactresnet18-cifar10-patchmixup-pgd-20220812.log 2>&1

# # # preactresnet18 patchmixup beta(1,1) defense for autoattack (eps=0.05)
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220812.py run --mode defense --defense_mode patchmixup --mix_mode maskmixup --sample_mode bernoullisampler --attack_mode autoattack --attack_eps 0.05 --blackbox --exp_name preactresnet18-cifar10 --cla_model preactresnet18 --cla_network_pkl XXX-cla1-XXX --dataset cifar10 --test_adv_dataset /home/data/maggie/result-newhome/attack/autoattack/preactresnet18-cifar10/20220708/00000-autoattack-eps-0.05-acc-7.59/attack-cifar10-dataset/samples/test --batch_size 256 --epochs 40 --lr 0.001 >> /home/maggie/mmat/log/CIFAR10/defense/patch-mixup-training/preactresnet18-cifar10-patchmixup-autoattack-20220812.log 2>&1

# # # preactresnet18 patchmixup beta(1,1) defense for deepfool (eps=0.02)
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220812.py run --mode defense --defense_mode patchmixup --mix_mode maskmixup --sample_mode bernoullisampler --attack_mode deepfool --attack_eps 0.02 --blackbox --exp_name preactresnet18-cifar10 --cla_model preactresnet18 --cla_network_pkl XXX-cla1-XXX --dataset cifar10 --test_adv_dataset /home/data/maggie/result-newhome/attack/deepfool/preactresnet18-cifar10/20220707/00000-deepfool-eps-0.02-acc-10.36/attack-cifar10-dataset/samples/test --batch_size 256 --epochs 40 --lr 0.001 >> /home/maggie/mmat/log/CIFAR10/defense/patch-mixup-training/preactresnet18-cifar10-patchmixup-deepfool-20220812.log 2>&1

# # # preactresnet18 patchmixup beta(1,1) defense for cw (conf=0)
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220812.py run --mode defense --defense_mode patchmixup --mix_mode maskmixup --sample_mode bernoullisampler --attack_mode cw --attack_eps 0.05 --blackbox --exp_name preactresnet18-cifar10 --cla_model preactresnet18 --cla_network_pkl XXX-cla1-XXX --dataset cifar10 --test_adv_dataset /home/data/maggie/result-newhome/attack/cw/preactresnet18-cifar10/20220629/00000-cw-confidence-0.0-acc-2.6/attack-cifar10-dataset/samples/test --batch_size 256 --epochs 40 --lr 0.001 >> /home/maggie/mmat/log/CIFAR10/defense/patch-mixup-training/preactresnet18-cifar10-patchmixup-cw-20220812.log 2>&1

# # # preactresnet18 patchmixup beta(1,1) defense for om-fgsm (eps=0.3)
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220812.py run --mode defense --defense_mode patchmixup --mix_mode maskmixup --sample_mode bernoullisampler --attack_mode om-fgsm --attack_eps 0.3 --blackbox --exp_name preactresnet18-cifar10 --cla_model preactresnet18 --cla_network_pkl XXX-cla1-XXX --dataset cifar10 --test_adv_dataset /home/data/maggie/result-newhome/attack/fgsm/preactresnet18-cifar10/20220702/00004-omfgsm-eps-0.3-acc-27.96/attack-cifar10-dataset/latent-attack-samples/test --batch_size 256 --epochs 40 --lr 0.001 >> /home/maggie/mmat/log/CIFAR10/defense/patch-mixup-training/preactresnet18-cifar10-patchmixup-om-fgsm-20220812.log 2>&1

# # # preactresnet18 patchmixup beta(1,1) defense for om-pgd (eps=0.05)
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220812.py run --mode defense --defense_mode patchmixup --mix_mode maskmixup --sample_mode bernoullisampler --attack_mode om-pgd --attack_eps 0.05 --blackbox --exp_name preactresnet18-cifar10 --cla_model preactresnet18 --cla_network_pkl XXX-cla1-XXX --dataset cifar10 --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/preactresnet18-cifar10/20220703/00001-ompgd-eps-0.05-acc-21.68/attack-cifar10-dataset/latent-attack-samples/test --batch_size 256 --epochs 40 --lr 0.001 >> /home/maggie/mmat/log/CIFAR10/defense/patch-mixup-training/preactresnet18-cifar10-patchmixup-om-pgd-20220812.log 2>&1


# # #--------------------cla-2---------------------
# # preactresnet18 patchmixup beta(1,1) defense for pgd(eps=0.05)
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220812.py run --mode defense --defense_mode patchmixup --mix_mode maskmixup --sample_mode bernoullisampler --attack_mode pgd --attack_eps 0.05 --blackbox --exp_name preactresnet18-cifar10 --cla_model preactresnet18 --cla_network_pkl XXX-cla2-XXX --dataset cifar10 --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/preactresnet18-cifar10/20220624/00001-pgd-eps-0.05-acc-28.93/attack-cifar10-dataset/samples/test --batch_size 256 --epochs 40 --lr 0.001 >> /home/maggie/mmat/log/CIFAR10/defense/patch-mixup-training/preactresnet18-cifar10-patchmixup-pgd-20220812.log 2>&1

# # # preactresnet18 patchmixup beta(1,1) defense for autoattack (eps=0.05)
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220812.py run --mode defense --defense_mode patchmixup --mix_mode maskmixup --sample_mode bernoullisampler --attack_mode autoattack --attack_eps 0.05 --blackbox --exp_name preactresnet18-cifar10 --cla_model preactresnet18 --cla_network_pkl XXX-cla2-XXX --dataset cifar10 --test_adv_dataset /home/data/maggie/result-newhome/attack/autoattack/preactresnet18-cifar10/20220708/00000-autoattack-eps-0.05-acc-7.59/attack-cifar10-dataset/samples/test --batch_size 256 --epochs 40 --lr 0.001 >> /home/maggie/mmat/log/CIFAR10/defense/patch-mixup-training/preactresnet18-cifar10-patchmixup-autoattack-20220812.log 2>&1

# # # preactresnet18 patchmixup beta(1,1) defense for deepfool (eps=0.02)
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220812.py run --mode defense --defense_mode patchmixup --mix_mode maskmixup --sample_mode bernoullisampler --attack_mode deepfool --attack_eps 0.02 --blackbox --exp_name preactresnet18-cifar10 --cla_model preactresnet18 --cla_network_pkl XXX-cla2-XXX --dataset cifar10 --test_adv_dataset /home/data/maggie/result-newhome/attack/deepfool/preactresnet18-cifar10/20220707/00000-deepfool-eps-0.02-acc-10.36/attack-cifar10-dataset/samples/test --batch_size 256 --epochs 40 --lr 0.001 >> /home/maggie/mmat/log/CIFAR10/defense/patch-mixup-training/preactresnet18-cifar10-patchmixup-deepfool-20220812.log 2>&1

# # # preactresnet18 patchmixup beta(1,1) defense for cw (conf=0)
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220812.py run --mode defense --defense_mode patchmixup --mix_mode maskmixup --sample_mode bernoullisampler --attack_mode cw --attack_eps 0.05 --blackbox --exp_name preactresnet18-cifar10 --cla_model preactresnet18 --cla_network_pkl XXX-cla2-XXX --dataset cifar10 --test_adv_dataset /home/data/maggie/result-newhome/attack/cw/preactresnet18-cifar10/20220629/00000-cw-confidence-0.0-acc-2.6/attack-cifar10-dataset/samples/test --batch_size 256 --epochs 40 --lr 0.001 >> /home/maggie/mmat/log/CIFAR10/defense/patch-mixup-training/preactresnet18-cifar10-patchmixup-cw-20220812.log 2>&1

# # # preactresnet18 patchmixup beta(1,1) defense for om-fgsm (eps=0.3)
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220812.py run --mode defense --defense_mode patchmixup --mix_mode maskmixup --sample_mode bernoullisampler --attack_mode om-fgsm --attack_eps 0.3 --blackbox --exp_name preactresnet18-cifar10 --cla_model preactresnet18 --cla_network_pkl XXX-cla2-XXX --dataset cifar10 --test_adv_dataset /home/data/maggie/result-newhome/attack/fgsm/preactresnet18-cifar10/20220702/00004-omfgsm-eps-0.3-acc-27.96/attack-cifar10-dataset/latent-attack-samples/test --batch_size 256 --epochs 40 --lr 0.001 >> /home/maggie/mmat/log/CIFAR10/defense/patch-mixup-training/preactresnet18-cifar10-patchmixup-om-fgsm-20220812.log 2>&1

# # # preactresnet18 patchmixup beta(1,1) defense for om-pgd (eps=0.05)
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220812.py run --mode defense --defense_mode patchmixup --mix_mode maskmixup --sample_mode bernoullisampler --attack_mode om-pgd --attack_eps 0.05 --blackbox --exp_name preactresnet18-cifar10 --cla_model preactresnet18 --cla_network_pkl XXX-cla2-XXX --dataset cifar10 --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/preactresnet18-cifar10/20220703/00001-ompgd-eps-0.05-acc-21.68/attack-cifar10-dataset/latent-attack-samples/test --batch_size 256 --epochs 40 --lr 0.001 >> /home/maggie/mmat/log/CIFAR10/defense/patch-mixup-training/preactresnet18-cifar10-patchmixup-om-pgd-20220812.log 2>&1


# # #--------------------cla-3---------------------
# # preactresnet18 patchmixup beta(1,1) defense for pgd(eps=0.05)
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220812.py run --mode defense --defense_mode patchmixup --mix_mode maskmixup --sample_mode bernoullisampler --attack_mode pgd --attack_eps 0.05 --blackbox --exp_name preactresnet18-cifar10 --cla_model preactresnet18 --cla_network_pkl XXX-cla3-XXX --dataset cifar10 --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/preactresnet18-cifar10/20220624/00001-pgd-eps-0.05-acc-28.93/attack-cifar10-dataset/samples/test --batch_size 256 --epochs 40 --lr 0.001 >> /home/maggie/mmat/log/CIFAR10/defense/patch-mixup-training/preactresnet18-cifar10-patchmixup-pgd-20220812.log 2>&1

# # # preactresnet18 patchmixup beta(1,1) defense for autoattack (eps=0.05)
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220812.py run --mode defense --defense_mode patchmixup --mix_mode maskmixup --sample_mode bernoullisampler --attack_mode autoattack --attack_eps 0.05 --blackbox --exp_name preactresnet18-cifar10 --cla_model preactresnet18 --cla_network_pkl XXX-cla3-XXX --dataset cifar10 --test_adv_dataset /home/data/maggie/result-newhome/attack/autoattack/preactresnet18-cifar10/20220708/00000-autoattack-eps-0.05-acc-7.59/attack-cifar10-dataset/samples/test --batch_size 256 --epochs 40 --lr 0.001 >> /home/maggie/mmat/log/CIFAR10/defense/patch-mixup-training/preactresnet18-cifar10-patchmixup-autoattack-20220812.log 2>&1

# # # preactresnet18 patchmixup beta(1,1) defense for deepfool (eps=0.02)
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220812.py run --mode defense --defense_mode patchmixup --mix_mode maskmixup --sample_mode bernoullisampler --attack_mode deepfool --attack_eps 0.02 --blackbox --exp_name preactresnet18-cifar10 --cla_model preactresnet18 --cla_network_pkl XXX-cla3-XXX --dataset cifar10 --test_adv_dataset /home/data/maggie/result-newhome/attack/deepfool/preactresnet18-cifar10/20220707/00000-deepfool-eps-0.02-acc-10.36/attack-cifar10-dataset/samples/test --batch_size 256 --epochs 40 --lr 0.001 >> /home/maggie/mmat/log/CIFAR10/defense/patch-mixup-training/preactresnet18-cifar10-patchmixup-deepfool-20220812.log 2>&1

# # # preactresnet18 patchmixup beta(1,1) defense for cw (conf=0)
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220812.py run --mode defense --defense_mode patchmixup --mix_mode maskmixup --sample_mode bernoullisampler --attack_mode cw --attack_eps 0.05 --blackbox --exp_name preactresnet18-cifar10 --cla_model preactresnet18 --cla_network_pkl XXX-cla3-XXX --dataset cifar10 --test_adv_dataset /home/data/maggie/result-newhome/attack/cw/preactresnet18-cifar10/20220629/00000-cw-confidence-0.0-acc-2.6/attack-cifar10-dataset/samples/test --batch_size 256 --epochs 40 --lr 0.001 >> /home/maggie/mmat/log/CIFAR10/defense/patch-mixup-training/preactresnet18-cifar10-patchmixup-cw-20220812.log 2>&1

# # # preactresnet18 patchmixup beta(1,1) defense for om-fgsm (eps=0.3)
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220812.py run --mode defense --defense_mode patchmixup --mix_mode maskmixup --sample_mode bernoullisampler --attack_mode om-fgsm --attack_eps 0.3 --blackbox --exp_name preactresnet18-cifar10 --cla_model preactresnet18 --cla_network_pkl XXX-cla3-XXX --dataset cifar10 --test_adv_dataset /home/data/maggie/result-newhome/attack/fgsm/preactresnet18-cifar10/20220702/00004-omfgsm-eps-0.3-acc-27.96/attack-cifar10-dataset/latent-attack-samples/test --batch_size 256 --epochs 40 --lr 0.001 >> /home/maggie/mmat/log/CIFAR10/defense/patch-mixup-training/preactresnet18-cifar10-patchmixup-om-fgsm-20220812.log 2>&1

# # # preactresnet18 patchmixup beta(1,1) defense for om-pgd (eps=0.05)
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220812.py run --mode defense --defense_mode patchmixup --mix_mode maskmixup --sample_mode bernoullisampler --attack_mode om-pgd --attack_eps 0.05 --blackbox --exp_name preactresnet18-cifar10 --cla_model preactresnet18 --cla_network_pkl XXX-cla3-XXX --dataset cifar10 --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/preactresnet18-cifar10/20220703/00001-ompgd-eps-0.05-acc-21.68/attack-cifar10-dataset/latent-attack-samples/test --batch_size 256 --epochs 40 --lr 0.001 >> /home/maggie/mmat/log/CIFAR10/defense/patch-mixup-training/preactresnet18-cifar10-patchmixup-om-pgd-20220812.log 2>&1


# # #--------------------cla-4---------------------
# # preactresnet18 patchmixup beta(1,1) defense for pgd(eps=0.05)
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220812.py run --mode defense --defense_mode patchmixup --mix_mode maskmixup --sample_mode bernoullisampler --attack_mode pgd --attack_eps 0.05 --blackbox --exp_name preactresnet18-cifar10 --cla_model preactresnet18 --cla_network_pkl XXX-cla4-XXX --dataset cifar10 --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/preactresnet18-cifar10/20220624/00001-pgd-eps-0.05-acc-28.93/attack-cifar10-dataset/samples/test --batch_size 256 --epochs 40 --lr 0.001 >> /home/maggie/mmat/log/CIFAR10/defense/patch-mixup-training/preactresnet18-cifar10-patchmixup-pgd-20220812.log 2>&1

# # # preactresnet18 patchmixup beta(1,1) defense for autoattack (eps=0.05)
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220812.py run --mode defense --defense_mode patchmixup --mix_mode maskmixup --sample_mode bernoullisampler --attack_mode autoattack --attack_eps 0.05 --blackbox --exp_name preactresnet18-cifar10 --cla_model preactresnet18 --cla_network_pkl XXX-cla4-XXX --dataset cifar10 --test_adv_dataset /home/data/maggie/result-newhome/attack/autoattack/preactresnet18-cifar10/20220708/00000-autoattack-eps-0.05-acc-7.59/attack-cifar10-dataset/samples/test --batch_size 256 --epochs 40 --lr 0.001 >> /home/maggie/mmat/log/CIFAR10/defense/patch-mixup-training/preactresnet18-cifar10-patchmixup-autoattack-20220812.log 2>&1

# # # preactresnet18 patchmixup beta(1,1) defense for deepfool (eps=0.02)
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220812.py run --mode defense --defense_mode patchmixup --mix_mode maskmixup --sample_mode bernoullisampler --attack_mode deepfool --attack_eps 0.02 --blackbox --exp_name preactresnet18-cifar10 --cla_model preactresnet18 --cla_network_pkl XXX-cla4-XXX --dataset cifar10 --test_adv_dataset /home/data/maggie/result-newhome/attack/deepfool/preactresnet18-cifar10/20220707/00000-deepfool-eps-0.02-acc-10.36/attack-cifar10-dataset/samples/test --batch_size 256 --epochs 40 --lr 0.001 >> /home/maggie/mmat/log/CIFAR10/defense/patch-mixup-training/preactresnet18-cifar10-patchmixup-deepfool-20220812.log 2>&1

# # # preactresnet18 patchmixup beta(1,1) defense for cw (conf=0)
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220812.py run --mode defense --defense_mode patchmixup --mix_mode maskmixup --sample_mode bernoullisampler --attack_mode cw --attack_eps 0.05 --blackbox --exp_name preactresnet18-cifar10 --cla_model preactresnet18 --cla_network_pkl XXX-cla4-XXX --dataset cifar10 --test_adv_dataset /home/data/maggie/result-newhome/attack/cw/preactresnet18-cifar10/20220629/00000-cw-confidence-0.0-acc-2.6/attack-cifar10-dataset/samples/test --batch_size 256 --epochs 40 --lr 0.001 >> /home/maggie/mmat/log/CIFAR10/defense/patch-mixup-training/preactresnet18-cifar10-patchmixup-cw-20220812.log 2>&1

# # # preactresnet18 patchmixup beta(1,1) defense for om-fgsm (eps=0.3)
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220812.py run --mode defense --defense_mode patchmixup --mix_mode maskmixup --sample_mode bernoullisampler --attack_mode om-fgsm --attack_eps 0.3 --blackbox --exp_name preactresnet18-cifar10 --cla_model preactresnet18 --cla_network_pkl XXX-cla4-XXX --dataset cifar10 --test_adv_dataset /home/data/maggie/result-newhome/attack/fgsm/preactresnet18-cifar10/20220702/00004-omfgsm-eps-0.3-acc-27.96/attack-cifar10-dataset/latent-attack-samples/test --batch_size 256 --epochs 40 --lr 0.001 >> /home/maggie/mmat/log/CIFAR10/defense/patch-mixup-training/preactresnet18-cifar10-patchmixup-om-fgsm-20220812.log 2>&1

# # # preactresnet18 patchmixup beta(1,1) defense for om-pgd (eps=0.05)
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220812.py run --mode defense --defense_mode patchmixup --mix_mode maskmixup --sample_mode bernoullisampler --attack_mode om-pgd --attack_eps 0.05 --blackbox --exp_name preactresnet18-cifar10 --cla_model preactresnet18 --cla_network_pkl XXX-cla4-XXX --dataset cifar10 --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/preactresnet18-cifar10/20220703/00001-ompgd-eps-0.05-acc-21.68/attack-cifar10-dataset/latent-attack-samples/test --batch_size 256 --epochs 40 --lr 0.001 >> /home/maggie/mmat/log/CIFAR10/defense/patch-mixup-training/preactresnet18-cifar10-patchmixup-om-pgd-20220812.log 2>&1

# # #--------------------cla-5---------------------
# # preactresnet18 patchmixup beta(1,1) defense for pgd(eps=0.05)
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220812.py run --mode defense --defense_mode patchmixup --mix_mode maskmixup --sample_mode bernoullisampler --attack_mode pgd --attack_eps 0.05 --blackbox --exp_name preactresnet18-cifar10 --cla_model preactresnet18 --cla_network_pkl XXX-cla5-XXX --dataset cifar10 --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/preactresnet18-cifar10/20220624/00001-pgd-eps-0.05-acc-28.93/attack-cifar10-dataset/samples/test --batch_size 256 --epochs 40 --lr 0.001 >> /home/maggie/mmat/log/CIFAR10/defense/patch-mixup-training/preactresnet18-cifar10-patchmixup-pgd-20220812.log 2>&1

# # # preactresnet18 patchmixup beta(1,1) defense for autoattack (eps=0.05)
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220812.py run --mode defense --defense_mode patchmixup --mix_mode maskmixup --sample_mode bernoullisampler --attack_mode autoattack --attack_eps 0.05 --blackbox --exp_name preactresnet18-cifar10 --cla_model preactresnet18 --cla_network_pkl XXX-cla5-XXX --dataset cifar10 --test_adv_dataset /home/data/maggie/result-newhome/attack/autoattack/preactresnet18-cifar10/20220708/00000-autoattack-eps-0.05-acc-7.59/attack-cifar10-dataset/samples/test --batch_size 256 --epochs 40 --lr 0.001 >> /home/maggie/mmat/log/CIFAR10/defense/patch-mixup-training/preactresnet18-cifar10-patchmixup-autoattack-20220812.log 2>&1

# # # preactresnet18 patchmixup beta(1,1) defense for deepfool (eps=0.02)
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220812.py run --mode defense --defense_mode patchmixup --mix_mode maskmixup --sample_mode bernoullisampler --attack_mode deepfool --attack_eps 0.02 --blackbox --exp_name preactresnet18-cifar10 --cla_model preactresnet18 --cla_network_pkl XXX-cla5-XXX --dataset cifar10 --test_adv_dataset /home/data/maggie/result-newhome/attack/deepfool/preactresnet18-cifar10/20220707/00000-deepfool-eps-0.02-acc-10.36/attack-cifar10-dataset/samples/test --batch_size 256 --epochs 40 --lr 0.001 >> /home/maggie/mmat/log/CIFAR10/defense/patch-mixup-training/preactresnet18-cifar10-patchmixup-deepfool-20220812.log 2>&1

# # # preactresnet18 patchmixup beta(1,1) defense for cw (conf=0)
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220812.py run --mode defense --defense_mode patchmixup --mix_mode maskmixup --sample_mode bernoullisampler --attack_mode cw --attack_eps 0.05 --blackbox --exp_name preactresnet18-cifar10 --cla_model preactresnet18 --cla_network_pkl XXX-cla5-XXX --dataset cifar10 --test_adv_dataset /home/data/maggie/result-newhome/attack/cw/preactresnet18-cifar10/20220629/00000-cw-confidence-0.0-acc-2.6/attack-cifar10-dataset/samples/test --batch_size 256 --epochs 40 --lr 0.001 >> /home/maggie/mmat/log/CIFAR10/defense/patch-mixup-training/preactresnet18-cifar10-patchmixup-cw-20220812.log 2>&1

# # # preactresnet18 patchmixup beta(1,1) defense for om-fgsm (eps=0.3)
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220812.py run --mode defense --defense_mode patchmixup --mix_mode maskmixup --sample_mode bernoullisampler --attack_mode om-fgsm --attack_eps 0.3 --blackbox --exp_name preactresnet18-cifar10 --cla_model preactresnet18 --cla_network_pkl XXX-cla5-XXX --dataset cifar10 --test_adv_dataset /home/data/maggie/result-newhome/attack/fgsm/preactresnet18-cifar10/20220702/00004-omfgsm-eps-0.3-acc-27.96/attack-cifar10-dataset/latent-attack-samples/test --batch_size 256 --epochs 40 --lr 0.001 >> /home/maggie/mmat/log/CIFAR10/defense/patch-mixup-training/preactresnet18-cifar10-patchmixup-om-fgsm-20220812.log 2>&1

# # # preactresnet18 patchmixup beta(1,1) defense for om-pgd (eps=0.05)
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220812.py run --mode defense --defense_mode patchmixup --mix_mode maskmixup --sample_mode bernoullisampler --attack_mode om-pgd --attack_eps 0.05 --blackbox --exp_name preactresnet18-cifar10 --cla_model preactresnet18 --cla_network_pkl XXX-cla5-XXX --dataset cifar10 --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/preactresnet18-cifar10/20220703/00001-ompgd-eps-0.05-acc-21.68/attack-cifar10-dataset/latent-attack-samples/test --batch_size 256 --epochs 40 --lr 0.001 >> /home/maggie/mmat/log/CIFAR10/defense/patch-mixup-training/preactresnet18-cifar10-patchmixup-om-pgd-20220812.log 2>&1

# # #--------------------cla-6---------------------
# # preactresnet18 patchmixup beta(1,1) defense for pgd(eps=0.05)
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220812.py run --mode defense --defense_mode patchmixup --mix_mode maskmixup --sample_mode bernoullisampler --attack_mode pgd --attack_eps 0.05 --blackbox --exp_name preactresnet18-cifar10 --cla_model preactresnet18 --cla_network_pkl XXX-cla6-XXX --dataset cifar10 --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/preactresnet18-cifar10/20220624/00001-pgd-eps-0.05-acc-28.93/attack-cifar10-dataset/samples/test --batch_size 256 --epochs 40 --lr 0.001 >> /home/maggie/mmat/log/CIFAR10/defense/patch-mixup-training/preactresnet18-cifar10-patchmixup-pgd-20220812.log 2>&1

# # # preactresnet18 patchmixup beta(1,1) defense for autoattack (eps=0.05)
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220812.py run --mode defense --defense_mode patchmixup --mix_mode maskmixup --sample_mode bernoullisampler --attack_mode autoattack --attack_eps 0.05 --blackbox --exp_name preactresnet18-cifar10 --cla_model preactresnet18 --cla_network_pkl XXX-cla6-XXX --dataset cifar10 --test_adv_dataset /home/data/maggie/result-newhome/attack/autoattack/preactresnet18-cifar10/20220708/00000-autoattack-eps-0.05-acc-7.59/attack-cifar10-dataset/samples/test --batch_size 256 --epochs 40 --lr 0.001 >> /home/maggie/mmat/log/CIFAR10/defense/patch-mixup-training/preactresnet18-cifar10-patchmixup-autoattack-20220812.log 2>&1

# # # preactresnet18 patchmixup beta(1,1) defense for deepfool (eps=0.02)
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220812.py run --mode defense --defense_mode patchmixup --mix_mode maskmixup --sample_mode bernoullisampler --attack_mode deepfool --attack_eps 0.02 --blackbox --exp_name preactresnet18-cifar10 --cla_model preactresnet18 --cla_network_pkl XXX-cla6-XXX --dataset cifar10 --test_adv_dataset /home/data/maggie/result-newhome/attack/deepfool/preactresnet18-cifar10/20220707/00000-deepfool-eps-0.02-acc-10.36/attack-cifar10-dataset/samples/test --batch_size 256 --epochs 40 --lr 0.001 >> /home/maggie/mmat/log/CIFAR10/defense/patch-mixup-training/preactresnet18-cifar10-patchmixup-deepfool-20220812.log 2>&1

# # # preactresnet18 patchmixup beta(1,1) defense for cw (conf=0)
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220812.py run --mode defense --defense_mode patchmixup --mix_mode maskmixup --sample_mode bernoullisampler --attack_mode cw --attack_eps 0.05 --blackbox --exp_name preactresnet18-cifar10 --cla_model preactresnet18 --cla_network_pkl XXX-cla6-XXX --dataset cifar10 --test_adv_dataset /home/data/maggie/result-newhome/attack/cw/preactresnet18-cifar10/20220629/00000-cw-confidence-0.0-acc-2.6/attack-cifar10-dataset/samples/test --batch_size 256 --epochs 40 --lr 0.001 >> /home/maggie/mmat/log/CIFAR10/defense/patch-mixup-training/preactresnet18-cifar10-patchmixup-cw-20220812.log 2>&1

# # # preactresnet18 patchmixup beta(1,1) defense for om-fgsm (eps=0.3)
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220812.py run --mode defense --defense_mode patchmixup --mix_mode maskmixup --sample_mode bernoullisampler --attack_mode om-fgsm --attack_eps 0.3 --blackbox --exp_name preactresnet18-cifar10 --cla_model preactresnet18 --cla_network_pkl XXX-cla6-XXX --dataset cifar10 --test_adv_dataset /home/data/maggie/result-newhome/attack/fgsm/preactresnet18-cifar10/20220702/00004-omfgsm-eps-0.3-acc-27.96/attack-cifar10-dataset/latent-attack-samples/test --batch_size 256 --epochs 40 --lr 0.001 >> /home/maggie/mmat/log/CIFAR10/defense/patch-mixup-training/preactresnet18-cifar10-patchmixup-om-fgsm-20220812.log 2>&1

# # # preactresnet18 patchmixup beta(1,1) defense for om-pgd (eps=0.05)
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220812.py run --mode defense --defense_mode patchmixup --mix_mode maskmixup --sample_mode bernoullisampler --attack_mode om-pgd --attack_eps 0.05 --blackbox --exp_name preactresnet18-cifar10 --cla_model preactresnet18 --cla_network_pkl XXX-cla6-XXX --dataset cifar10 --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/preactresnet18-cifar10/20220703/00001-ompgd-eps-0.05-acc-21.68/attack-cifar10-dataset/latent-attack-samples/test --batch_size 256 --epochs 40 --lr 0.001 >> /home/maggie/mmat/log/CIFAR10/defense/patch-mixup-training/preactresnet18-cifar10-patchmixup-om-pgd-20220812.log 2>&1
