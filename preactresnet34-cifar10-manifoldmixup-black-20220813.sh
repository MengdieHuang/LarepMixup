source ~/.bashrc
source /home/xieyi/anaconda3/bin/activate mmat

# adv50000 cat cle50000
# PreActResNet34 manifoldmixup beta(1,1) against FGSM (eps=0.05)
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220815.py run --mode defense --defense_mode manifoldmixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode fgsm --attack_eps 0.05 --blackbox --exp_name preactresnet34-cifar10 --cla_model preactresnet34 --cla_network_pkl /home/data/maggie/result-newhome/train/cla-train/preactresnet34-cifar10/20210927/00002-testacc-83.57/train-cifar10-dataset/standard-trained-classifier-preactresnet34-on-clean-cifar10-epoch-0021.pkl --dataset cifar10 --test_adv_dataset /home/data/maggie/result-newhome/attack/fgsm/preactresnet34-cifar10/20220627/00001-fgsm-eps-0.05-acc-31.37/attack-cifar10-dataset/samples/test --batch_size 256 --epochs 30 --lr 0.01 >> /home/maggie/mmat/log/CIFAR10/defense/manifold-mixup-training/preactresnet34-cifar10-manifoldmixup-fgsm-20220815.log 2>&1

# PreActResNet34 manifoldmixup beta(1,1) against FGSM (eps=0.05)
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220815.py run --mode defense --defense_mode manifoldmixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode fgsm --attack_eps 0.05 --blackbox --exp_name preactresnet34-cifar10 --cla_model preactresnet34 --cla_network_pkl /home/data/maggie/result-newhome/train/cla-train/preactresnet34-cifar10/20210927/00002-testacc-83.57/train-cifar10-dataset/standard-trained-classifier-preactresnet34-on-clean-cifar10-epoch-0021.pkl --dataset cifar10 --test_adv_dataset /home/data/maggie/result-newhome/attack/fgsm/preactresnet34-cifar10/20220627/00001-fgsm-eps-0.05-acc-31.37/attack-cifar10-dataset/samples/test --batch_size 256 --epochs 30 --lr 0.01 >> /home/maggie/mmat/log/CIFAR10/defense/manifold-mixup-training/preactresnet34-cifar10-manifoldmixup-fgsm-20220815.log 2>&1

# PreActResNet34 manifoldmixup beta(1,1) against FGSM (eps=0.05)
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220815.py run --mode defense --defense_mode manifoldmixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode fgsm --attack_eps 0.05 --blackbox --exp_name preactresnet34-cifar10 --cla_model preactresnet34 --cla_network_pkl /home/data/maggie/result-newhome/train/cla-train/preactresnet34-cifar10/20210927/00002-testacc-83.57/train-cifar10-dataset/standard-trained-classifier-preactresnet34-on-clean-cifar10-epoch-0021.pkl --dataset cifar10 --test_adv_dataset /home/data/maggie/result-newhome/attack/fgsm/preactresnet34-cifar10/20220627/00001-fgsm-eps-0.05-acc-31.37/attack-cifar10-dataset/samples/test --batch_size 256 --epochs 30 --lr 0.01 >> /home/maggie/mmat/log/CIFAR10/defense/manifold-mixup-training/preactresnet34-cifar10-manifoldmixup-fgsm-20220815.log 2>&1

# PreActResNet34 manifoldmixup beta(1,1) against FGSM (eps=0.05)
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220815.py run --mode defense --defense_mode manifoldmixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode fgsm --attack_eps 0.05 --blackbox --exp_name preactresnet34-cifar10 --cla_model preactresnet34 --cla_network_pkl /home/data/maggie/result-newhome/train/cla-train/preactresnet34-cifar10/20210927/00002-testacc-83.57/train-cifar10-dataset/standard-trained-classifier-preactresnet34-on-clean-cifar10-epoch-0021.pkl --dataset cifar10 --test_adv_dataset /home/data/maggie/result-newhome/attack/fgsm/preactresnet34-cifar10/20220627/00001-fgsm-eps-0.05-acc-31.37/attack-cifar10-dataset/samples/test --batch_size 256 --epochs 30 --lr 0.01 >> /home/maggie/mmat/log/CIFAR10/defense/manifold-mixup-training/preactresnet34-cifar10-manifoldmixup-fgsm-20220815.log 2>&1

# PreActResNet34 manifoldmixup beta(1,1) against FGSM (eps=0.05)
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220815.py run --mode defense --defense_mode manifoldmixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode fgsm --attack_eps 0.05 --blackbox --exp_name preactresnet34-cifar10 --cla_model preactresnet34 --cla_network_pkl /home/data/maggie/result-newhome/train/cla-train/preactresnet34-cifar10/20210927/00002-testacc-83.57/train-cifar10-dataset/standard-trained-classifier-preactresnet34-on-clean-cifar10-epoch-0021.pkl --dataset cifar10 --test_adv_dataset /home/data/maggie/result-newhome/attack/fgsm/preactresnet34-cifar10/20220627/00001-fgsm-eps-0.05-acc-31.37/attack-cifar10-dataset/samples/test --batch_size 256 --epochs 30 --lr 0.01 >> /home/maggie/mmat/log/CIFAR10/defense/manifold-mixup-training/preactresnet34-cifar10-manifoldmixup-fgsm-20220815.log 2>&1

# PreActResNet34 manifoldmixup beta(1,1) against FGSM (eps=0.05)
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220815.py run --mode defense --defense_mode manifoldmixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode fgsm --attack_eps 0.05 --blackbox --exp_name preactresnet34-cifar10 --cla_model preactresnet34 --cla_network_pkl /home/data/maggie/result-newhome/train/cla-train/preactresnet34-cifar10/20210927/00002-testacc-83.57/train-cifar10-dataset/standard-trained-classifier-preactresnet34-on-clean-cifar10-epoch-0021.pkl --dataset cifar10 --test_adv_dataset /home/data/maggie/result-newhome/attack/fgsm/preactresnet34-cifar10/20220627/00001-fgsm-eps-0.05-acc-31.37/attack-cifar10-dataset/samples/test --batch_size 256 --epochs 30 --lr 0.01 >> /home/maggie/mmat/log/CIFAR10/defense/manifold-mixup-training/preactresnet34-cifar10-manifoldmixup-fgsm-20220815.log 2>&1

# #-------------以上要存储at训练好的模型-----------
# # #   使用PGD对抗训练后的 cla1
# # PreActResNet34 manifoldmixup beta(1,1) against pgd (eps=0.05)
# CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220815.py run --mode defense --defense_mode manifoldmixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode pgd --attack_eps 0.05 --blackbox --exp_name preactresnet34-cifar10 --cla_model preactresnet34 --cla_network_pkl XXX-cla1-XXX --dataset cifar10 --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/preactresnet34-cifar10/20220624/00001-pgd-eps-0.05-acc-25.71/attack-cifar10-dataset/samples/test --batch_size 256 --epochs 30 --lr 0.01 >> /home/maggie/mmat/log/CIFAR10/defense/manifold-mixup-training/preactresnet34-cifar10-manifoldmixup-pgd-20220815.log 2>&1

# # PreActResNet34 manifoldmixup beta(1,1) against autoattack (eps=0.05)
# CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220815.py run --mode defense --defense_mode manifoldmixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode autoattack --attack_eps 0.05 --blackbox --exp_name preactresnet34-cifar10 --cla_model preactresnet34 --cla_network_pkl XXX-cla1-XXX --dataset cifar10 --test_adv_dataset /home/data/maggie/result-newhome/attack/autoattack/preactresnet34-cifar10/20220712/00000-autoattack-eps-0.05-acc-5.27/attack-cifar10-dataset/samples/test --batch_size 256 --epochs 30 --lr 0.01 >> /home/maggie/mmat/log/CIFAR10/defense/manifold-mixup-training/preactresnet34-cifar10-manifoldmixup-autoattack-20220815.log 2>&1

# # PreActResNet34 manifoldmixup beta(1,1) against deepfool (eps=0.02)
# CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220815.py run --mode defense --defense_mode manifoldmixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode deepfool --attack_eps 0.02 --blackbox --exp_name preactresnet34-cifar10 --cla_model preactresnet34 --cla_network_pkl XXX-cla1-XXX --dataset cifar10 --test_adv_dataset /home/data/maggie/result-newhome/attack/deepfool/preactresnet34-cifar10/20220707/00000-deepfool-eps-0.02-acc-12.27/attack-cifar10-dataset/samples/test --batch_size 256 --epochs 30 --lr 0.01 >> /home/maggie/mmat/log/CIFAR10/defense/manifold-mixup-training/preactresnet34-cifar10-manifoldmixup-deepfool-20220815.log 2>&1

# # PreActResNet34 manifoldmixup beta(1,1) against cw (eps=0.02)
# CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220815.py run --mode defense --defense_mode manifoldmixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode cw --attack_eps 0.05 --blackbox --exp_name preactresnet34-cifar10 --cla_model preactresnet34 --cla_network_pkl XXX-cla1-XXX --dataset cifar10 --test_adv_dataset /home/data/maggie/result-newhome/attack/cw/preactresnet34-cifar10/20220704/00000-cw-confidence-0.0-acc-1.89/attack-cifar10-dataset/samples/test --batch_size 256 --epochs 30 --lr 0.01 >> /home/maggie/mmat/log/CIFAR10/defense/manifold-mixup-training/preactresnet34-cifar10-manifoldmixup-cw-20220815.log 2>&1

# # PreActResNet34 manifoldmixup beta(1,1) against om-fgsm (eps=0.05)
# CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220815.py run --mode defense --defense_mode manifoldmixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode om-fgsm --attack_eps 0.05 --blackbox --exp_name preactresnet34-cifar10 --cla_model preactresnet34 --cla_network_pkl XXX-cla1-XXX --dataset cifar10 --test_adv_dataset /home/data/maggie/result-newhome/attack/fgsm/preactresnet34-cifar10/20220702/00001-omfgsm-eps-0.05-acc-49.23/attack-cifar10-dataset/latent-attack-samples/test --batch_size 256 --epochs 30 --lr 0.01 >> /home/maggie/mmat/log/CIFAR10/defense/manifold-mixup-training/preactresnet34-cifar10-manifoldmixup-om-fgsm-20220815.log 2>&1

# # PreActResNet34 manifoldmixup beta(1,1) against om-pgd (eps=0.05)
# CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220815.py run --mode defense --defense_mode manifoldmixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode om-pgd --attack_eps 0.05 --blackbox --exp_name preactresnet34-cifar10 --cla_model preactresnet34 --cla_network_pkl XXX-cla1-XXX --dataset cifar10 --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/preactresnet34-cifar10/20220703/00001-ompgd-eps-0.05-acc-17.05/attack-cifar10-dataset/latent-attack-samples/test --batch_size 256 --epochs 30 --lr 0.01 >> /home/maggie/mmat/log/CIFAR10/defense/manifold-mixup-training/preactresnet34-cifar10-manifoldmixup-om-pgd-20220815.log 2>&1

# # #   使用PGD对抗训练后的 cla2
# # PreActResNet34 manifoldmixup beta(1,1) against pgd (eps=0.05)
# CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220815.py run --mode defense --defense_mode manifoldmixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode pgd --attack_eps 0.05 --blackbox --exp_name preactresnet34-cifar10 --cla_model preactresnet34 --cla_network_pkl XXX-cla2-XXX --dataset cifar10 --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/preactresnet34-cifar10/20220624/00001-pgd-eps-0.05-acc-25.71/attack-cifar10-dataset/samples/test --batch_size 256 --epochs 30 --lr 0.01 >> /home/maggie/mmat/log/CIFAR10/defense/manifold-mixup-training/preactresnet34-cifar10-manifoldmixup-pgd-20220815.log 2>&1

# # PreActResNet34 manifoldmixup beta(1,1) against autoattack (eps=0.05)
# CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220815.py run --mode defense --defense_mode manifoldmixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode autoattack --attack_eps 0.05 --blackbox --exp_name preactresnet34-cifar10 --cla_model preactresnet34 --cla_network_pkl XXX-cla2-XXX --dataset cifar10 --test_adv_dataset /home/data/maggie/result-newhome/attack/autoattack/preactresnet34-cifar10/20220712/00000-autoattack-eps-0.05-acc-5.27/attack-cifar10-dataset/samples/test --batch_size 256 --epochs 30 --lr 0.01 >> /home/maggie/mmat/log/CIFAR10/defense/manifold-mixup-training/preactresnet34-cifar10-manifoldmixup-autoattack-20220815.log 2>&1

# # PreActResNet34 manifoldmixup beta(1,1) against deepfool (eps=0.02)
# CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220815.py run --mode defense --defense_mode manifoldmixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode deepfool --attack_eps 0.02 --blackbox --exp_name preactresnet34-cifar10 --cla_model preactresnet34 --cla_network_pkl XXX-cla2-XXX --dataset cifar10 --test_adv_dataset /home/data/maggie/result-newhome/attack/deepfool/preactresnet34-cifar10/20220707/00000-deepfool-eps-0.02-acc-12.27/attack-cifar10-dataset/samples/test --batch_size 256 --epochs 30 --lr 0.01 >> /home/maggie/mmat/log/CIFAR10/defense/manifold-mixup-training/preactresnet34-cifar10-manifoldmixup-deepfool-20220815.log 2>&1

# # PreActResNet34 manifoldmixup beta(1,1) against cw (eps=0.02)
# CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220815.py run --mode defense --defense_mode manifoldmixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode cw --attack_eps 0.05 --blackbox --exp_name preactresnet34-cifar10 --cla_model preactresnet34 --cla_network_pkl XXX-cla2-XXX --dataset cifar10 --test_adv_dataset /home/data/maggie/result-newhome/attack/cw/preactresnet34-cifar10/20220704/00000-cw-confidence-0.0-acc-1.89/attack-cifar10-dataset/samples/test --batch_size 256 --epochs 30 --lr 0.01 >> /home/maggie/mmat/log/CIFAR10/defense/manifold-mixup-training/preactresnet34-cifar10-manifoldmixup-cw-20220815.log 2>&1

# # PreActResNet34 manifoldmixup beta(1,1) against om-fgsm (eps=0.05)
# CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220815.py run --mode defense --defense_mode manifoldmixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode om-fgsm --attack_eps 0.05 --blackbox --exp_name preactresnet34-cifar10 --cla_model preactresnet34 --cla_network_pkl XXX-cla2-XXX --dataset cifar10 --test_adv_dataset /home/data/maggie/result-newhome/attack/fgsm/preactresnet34-cifar10/20220702/00001-omfgsm-eps-0.05-acc-49.23/attack-cifar10-dataset/latent-attack-samples/test --batch_size 256 --epochs 30 --lr 0.01 >> /home/maggie/mmat/log/CIFAR10/defense/manifold-mixup-training/preactresnet34-cifar10-manifoldmixup-om-fgsm-20220815.log 2>&1

# # PreActResNet34 manifoldmixup beta(1,1) against om-pgd (eps=0.05)
# CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220815.py run --mode defense --defense_mode manifoldmixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode om-pgd --attack_eps 0.05 --blackbox --exp_name preactresnet34-cifar10 --cla_model preactresnet34 --cla_network_pkl XXX-cla2-XXX --dataset cifar10 --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/preactresnet34-cifar10/20220703/00001-ompgd-eps-0.05-acc-17.05/attack-cifar10-dataset/latent-attack-samples/test --batch_size 256 --epochs 30 --lr 0.01 >> /home/maggie/mmat/log/CIFAR10/defense/manifold-mixup-training/preactresnet34-cifar10-manifoldmixup-om-pgd-20220815.log 2>&1

# # #   使用PGD对抗训练后的 cla3
# # PreActResNet34 manifoldmixup beta(1,1) against pgd (eps=0.05)
# CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220815.py run --mode defense --defense_mode manifoldmixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode pgd --attack_eps 0.05 --blackbox --exp_name preactresnet34-cifar10 --cla_model preactresnet34 --cla_network_pkl XXX-cla3-XXX --dataset cifar10 --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/preactresnet34-cifar10/20220624/00001-pgd-eps-0.05-acc-25.71/attack-cifar10-dataset/samples/test --batch_size 256 --epochs 30 --lr 0.01 >> /home/maggie/mmat/log/CIFAR10/defense/manifold-mixup-training/preactresnet34-cifar10-manifoldmixup-pgd-20220815.log 2>&1

# # PreActResNet34 manifoldmixup beta(1,1) against autoattack (eps=0.05)
# CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220815.py run --mode defense --defense_mode manifoldmixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode autoattack --attack_eps 0.05 --blackbox --exp_name preactresnet34-cifar10 --cla_model preactresnet34 --cla_network_pkl XXX-cla3-XXX --dataset cifar10 --test_adv_dataset /home/data/maggie/result-newhome/attack/autoattack/preactresnet34-cifar10/20220712/00000-autoattack-eps-0.05-acc-5.27/attack-cifar10-dataset/samples/test --batch_size 256 --epochs 30 --lr 0.01 >> /home/maggie/mmat/log/CIFAR10/defense/manifold-mixup-training/preactresnet34-cifar10-manifoldmixup-autoattack-20220815.log 2>&1

# # PreActResNet34 manifoldmixup beta(1,1) against deepfool (eps=0.02)
# CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220815.py run --mode defense --defense_mode manifoldmixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode deepfool --attack_eps 0.02 --blackbox --exp_name preactresnet34-cifar10 --cla_model preactresnet34 --cla_network_pkl XXX-cla3-XXX --dataset cifar10 --test_adv_dataset /home/data/maggie/result-newhome/attack/deepfool/preactresnet34-cifar10/20220707/00000-deepfool-eps-0.02-acc-12.27/attack-cifar10-dataset/samples/test --batch_size 256 --epochs 30 --lr 0.01 >> /home/maggie/mmat/log/CIFAR10/defense/manifold-mixup-training/preactresnet34-cifar10-manifoldmixup-deepfool-20220815.log 2>&1

# # PreActResNet34 manifoldmixup beta(1,1) against cw (eps=0.02)
# CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220815.py run --mode defense --defense_mode manifoldmixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode cw --attack_eps 0.05 --blackbox --exp_name preactresnet34-cifar10 --cla_model preactresnet34 --cla_network_pkl XXX-cla3-XXX --dataset cifar10 --test_adv_dataset /home/data/maggie/result-newhome/attack/cw/preactresnet34-cifar10/20220704/00000-cw-confidence-0.0-acc-1.89/attack-cifar10-dataset/samples/test --batch_size 256 --epochs 30 --lr 0.01 >> /home/maggie/mmat/log/CIFAR10/defense/manifold-mixup-training/preactresnet34-cifar10-manifoldmixup-cw-20220815.log 2>&1

# # PreActResNet34 manifoldmixup beta(1,1) against om-fgsm (eps=0.05)
# CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220815.py run --mode defense --defense_mode manifoldmixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode om-fgsm --attack_eps 0.05 --blackbox --exp_name preactresnet34-cifar10 --cla_model preactresnet34 --cla_network_pkl XXX-cla3-XXX --dataset cifar10 --test_adv_dataset /home/data/maggie/result-newhome/attack/fgsm/preactresnet34-cifar10/20220702/00001-omfgsm-eps-0.05-acc-49.23/attack-cifar10-dataset/latent-attack-samples/test --batch_size 256 --epochs 30 --lr 0.01 >> /home/maggie/mmat/log/CIFAR10/defense/manifold-mixup-training/preactresnet34-cifar10-manifoldmixup-om-fgsm-20220815.log 2>&1

# # PreActResNet34 manifoldmixup beta(1,1) against om-pgd (eps=0.05)
# CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220815.py run --mode defense --defense_mode manifoldmixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode om-pgd --attack_eps 0.05 --blackbox --exp_name preactresnet34-cifar10 --cla_model preactresnet34 --cla_network_pkl XXX-cla3-XXX --dataset cifar10 --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/preactresnet34-cifar10/20220703/00001-ompgd-eps-0.05-acc-17.05/attack-cifar10-dataset/latent-attack-samples/test --batch_size 256 --epochs 30 --lr 0.01 >> /home/maggie/mmat/log/CIFAR10/defense/manifold-mixup-training/preactresnet34-cifar10-manifoldmixup-om-pgd-20220815.log 2>&1

# # #   使用PGD对抗训练后的 cla4
# # PreActResNet34 manifoldmixup beta(1,1) against pgd (eps=0.05)
# CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220815.py run --mode defense --defense_mode manifoldmixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode pgd --attack_eps 0.05 --blackbox --exp_name preactresnet34-cifar10 --cla_model preactresnet34 --cla_network_pkl XXX-cla4-XXX --dataset cifar10 --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/preactresnet34-cifar10/20220624/00001-pgd-eps-0.05-acc-25.71/attack-cifar10-dataset/samples/test --batch_size 256 --epochs 30 --lr 0.01 >> /home/maggie/mmat/log/CIFAR10/defense/manifold-mixup-training/preactresnet34-cifar10-manifoldmixup-pgd-20220815.log 2>&1

# # PreActResNet34 manifoldmixup beta(1,1) against autoattack (eps=0.05)
# CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220815.py run --mode defense --defense_mode manifoldmixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode autoattack --attack_eps 0.05 --blackbox --exp_name preactresnet34-cifar10 --cla_model preactresnet34 --cla_network_pkl XXX-cla4-XXX --dataset cifar10 --test_adv_dataset /home/data/maggie/result-newhome/attack/autoattack/preactresnet34-cifar10/20220712/00000-autoattack-eps-0.05-acc-5.27/attack-cifar10-dataset/samples/test --batch_size 256 --epochs 30 --lr 0.01 >> /home/maggie/mmat/log/CIFAR10/defense/manifold-mixup-training/preactresnet34-cifar10-manifoldmixup-autoattack-20220815.log 2>&1

# # PreActResNet34 manifoldmixup beta(1,1) against deepfool (eps=0.02)
# CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220815.py run --mode defense --defense_mode manifoldmixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode deepfool --attack_eps 0.02 --blackbox --exp_name preactresnet34-cifar10 --cla_model preactresnet34 --cla_network_pkl XXX-cla4-XXX --dataset cifar10 --test_adv_dataset /home/data/maggie/result-newhome/attack/deepfool/preactresnet34-cifar10/20220707/00000-deepfool-eps-0.02-acc-12.27/attack-cifar10-dataset/samples/test --batch_size 256 --epochs 30 --lr 0.01 >> /home/maggie/mmat/log/CIFAR10/defense/manifold-mixup-training/preactresnet34-cifar10-manifoldmixup-deepfool-20220815.log 2>&1

# # PreActResNet34 manifoldmixup beta(1,1) against cw (eps=0.02)
# CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220815.py run --mode defense --defense_mode manifoldmixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode cw --attack_eps 0.05 --blackbox --exp_name preactresnet34-cifar10 --cla_model preactresnet34 --cla_network_pkl XXX-cla4-XXX --dataset cifar10 --test_adv_dataset /home/data/maggie/result-newhome/attack/cw/preactresnet34-cifar10/20220704/00000-cw-confidence-0.0-acc-1.89/attack-cifar10-dataset/samples/test --batch_size 256 --epochs 30 --lr 0.01 >> /home/maggie/mmat/log/CIFAR10/defense/manifold-mixup-training/preactresnet34-cifar10-manifoldmixup-cw-20220815.log 2>&1

# # PreActResNet34 manifoldmixup beta(1,1) against om-fgsm (eps=0.05)
# CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220815.py run --mode defense --defense_mode manifoldmixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode om-fgsm --attack_eps 0.05 --blackbox --exp_name preactresnet34-cifar10 --cla_model preactresnet34 --cla_network_pkl XXX-cla4-XXX --dataset cifar10 --test_adv_dataset /home/data/maggie/result-newhome/attack/fgsm/preactresnet34-cifar10/20220702/00001-omfgsm-eps-0.05-acc-49.23/attack-cifar10-dataset/latent-attack-samples/test --batch_size 256 --epochs 30 --lr 0.01 >> /home/maggie/mmat/log/CIFAR10/defense/manifold-mixup-training/preactresnet34-cifar10-manifoldmixup-om-fgsm-20220815.log 2>&1

# # PreActResNet34 manifoldmixup beta(1,1) against om-pgd (eps=0.05)
# CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220815.py run --mode defense --defense_mode manifoldmixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode om-pgd --attack_eps 0.05 --blackbox --exp_name preactresnet34-cifar10 --cla_model preactresnet34 --cla_network_pkl XXX-cla4-XXX --dataset cifar10 --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/preactresnet34-cifar10/20220703/00001-ompgd-eps-0.05-acc-17.05/attack-cifar10-dataset/latent-attack-samples/test --batch_size 256 --epochs 30 --lr 0.01 >> /home/maggie/mmat/log/CIFAR10/defense/manifold-mixup-training/preactresnet34-cifar10-manifoldmixup-om-pgd-20220815.log 2>&1

# # #   使用PGD对抗训练后的 cla5
# # PreActResNet34 manifoldmixup beta(1,1) against pgd (eps=0.05)
# CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220815.py run --mode defense --defense_mode manifoldmixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode pgd --attack_eps 0.05 --blackbox --exp_name preactresnet34-cifar10 --cla_model preactresnet34 --cla_network_pkl XXX-cla5-XXX --dataset cifar10 --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/preactresnet34-cifar10/20220624/00001-pgd-eps-0.05-acc-25.71/attack-cifar10-dataset/samples/test --batch_size 256 --epochs 30 --lr 0.01 >> /home/maggie/mmat/log/CIFAR10/defense/manifold-mixup-training/preactresnet34-cifar10-manifoldmixup-pgd-20220815.log 2>&1

# # PreActResNet34 manifoldmixup beta(1,1) against autoattack (eps=0.05)
# CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220815.py run --mode defense --defense_mode manifoldmixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode autoattack --attack_eps 0.05 --blackbox --exp_name preactresnet34-cifar10 --cla_model preactresnet34 --cla_network_pkl XXX-cla5-XXX --dataset cifar10 --test_adv_dataset /home/data/maggie/result-newhome/attack/autoattack/preactresnet34-cifar10/20220712/00000-autoattack-eps-0.05-acc-5.27/attack-cifar10-dataset/samples/test --batch_size 256 --epochs 30 --lr 0.01 >> /home/maggie/mmat/log/CIFAR10/defense/manifold-mixup-training/preactresnet34-cifar10-manifoldmixup-autoattack-20220815.log 2>&1

# # PreActResNet34 manifoldmixup beta(1,1) against deepfool (eps=0.02)
# CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220815.py run --mode defense --defense_mode manifoldmixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode deepfool --attack_eps 0.02 --blackbox --exp_name preactresnet34-cifar10 --cla_model preactresnet34 --cla_network_pkl XXX-cla5-XXX --dataset cifar10 --test_adv_dataset /home/data/maggie/result-newhome/attack/deepfool/preactresnet34-cifar10/20220707/00000-deepfool-eps-0.02-acc-12.27/attack-cifar10-dataset/samples/test --batch_size 256 --epochs 30 --lr 0.01 >> /home/maggie/mmat/log/CIFAR10/defense/manifold-mixup-training/preactresnet34-cifar10-manifoldmixup-deepfool-20220815.log 2>&1

# # PreActResNet34 manifoldmixup beta(1,1) against cw (eps=0.02)
# CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220815.py run --mode defense --defense_mode manifoldmixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode cw --attack_eps 0.05 --blackbox --exp_name preactresnet34-cifar10 --cla_model preactresnet34 --cla_network_pkl XXX-cla5-XXX --dataset cifar10 --test_adv_dataset /home/data/maggie/result-newhome/attack/cw/preactresnet34-cifar10/20220704/00000-cw-confidence-0.0-acc-1.89/attack-cifar10-dataset/samples/test --batch_size 256 --epochs 30 --lr 0.01 >> /home/maggie/mmat/log/CIFAR10/defense/manifold-mixup-training/preactresnet34-cifar10-manifoldmixup-cw-20220815.log 2>&1

# # PreActResNet34 manifoldmixup beta(1,1) against om-fgsm (eps=0.05)
# CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220815.py run --mode defense --defense_mode manifoldmixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode om-fgsm --attack_eps 0.05 --blackbox --exp_name preactresnet34-cifar10 --cla_model preactresnet34 --cla_network_pkl XXX-cla5-XXX --dataset cifar10 --test_adv_dataset /home/data/maggie/result-newhome/attack/fgsm/preactresnet34-cifar10/20220702/00001-omfgsm-eps-0.05-acc-49.23/attack-cifar10-dataset/latent-attack-samples/test --batch_size 256 --epochs 30 --lr 0.01 >> /home/maggie/mmat/log/CIFAR10/defense/manifold-mixup-training/preactresnet34-cifar10-manifoldmixup-om-fgsm-20220815.log 2>&1

# # PreActResNet34 manifoldmixup beta(1,1) against om-pgd (eps=0.05)
# CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220815.py run --mode defense --defense_mode manifoldmixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode om-pgd --attack_eps 0.05 --blackbox --exp_name preactresnet34-cifar10 --cla_model preactresnet34 --cla_network_pkl XXX-cla5-XXX --dataset cifar10 --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/preactresnet34-cifar10/20220703/00001-ompgd-eps-0.05-acc-17.05/attack-cifar10-dataset/latent-attack-samples/test --batch_size 256 --epochs 30 --lr 0.01 >> /home/maggie/mmat/log/CIFAR10/defense/manifold-mixup-training/preactresnet34-cifar10-manifoldmixup-om-pgd-20220815.log 2>&1

# # #   使用PGD对抗训练后的 cla6
# # PreActResNet34 manifoldmixup beta(1,1) against pgd (eps=0.05)
# CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220815.py run --mode defense --defense_mode manifoldmixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode pgd --attack_eps 0.05 --blackbox --exp_name preactresnet34-cifar10 --cla_model preactresnet34 --cla_network_pkl XXX-cla6-XXX --dataset cifar10 --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/preactresnet34-cifar10/20220624/00001-pgd-eps-0.05-acc-25.71/attack-cifar10-dataset/samples/test --batch_size 256 --epochs 30 --lr 0.01 >> /home/maggie/mmat/log/CIFAR10/defense/manifold-mixup-training/preactresnet34-cifar10-manifoldmixup-pgd-20220815.log 2>&1

# # PreActResNet34 manifoldmixup beta(1,1) against autoattack (eps=0.05)
# CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220815.py run --mode defense --defense_mode manifoldmixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode autoattack --attack_eps 0.05 --blackbox --exp_name preactresnet34-cifar10 --cla_model preactresnet34 --cla_network_pkl XXX-cla6-XXX --dataset cifar10 --test_adv_dataset /home/data/maggie/result-newhome/attack/autoattack/preactresnet34-cifar10/20220712/00000-autoattack-eps-0.05-acc-5.27/attack-cifar10-dataset/samples/test --batch_size 256 --epochs 30 --lr 0.01 >> /home/maggie/mmat/log/CIFAR10/defense/manifold-mixup-training/preactresnet34-cifar10-manifoldmixup-autoattack-20220815.log 2>&1

# # PreActResNet34 manifoldmixup beta(1,1) against deepfool (eps=0.02)
# CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220815.py run --mode defense --defense_mode manifoldmixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode deepfool --attack_eps 0.02 --blackbox --exp_name preactresnet34-cifar10 --cla_model preactresnet34 --cla_network_pkl XXX-cla6-XXX --dataset cifar10 --test_adv_dataset /home/data/maggie/result-newhome/attack/deepfool/preactresnet34-cifar10/20220707/00000-deepfool-eps-0.02-acc-12.27/attack-cifar10-dataset/samples/test --batch_size 256 --epochs 30 --lr 0.01 >> /home/maggie/mmat/log/CIFAR10/defense/manifold-mixup-training/preactresnet34-cifar10-manifoldmixup-deepfool-20220815.log 2>&1

# # PreActResNet34 manifoldmixup beta(1,1) against cw (eps=0.02)
# CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220815.py run --mode defense --defense_mode manifoldmixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode cw --attack_eps 0.05 --blackbox --exp_name preactresnet34-cifar10 --cla_model preactresnet34 --cla_network_pkl XXX-cla6-XXX --dataset cifar10 --test_adv_dataset /home/data/maggie/result-newhome/attack/cw/preactresnet34-cifar10/20220704/00000-cw-confidence-0.0-acc-1.89/attack-cifar10-dataset/samples/test --batch_size 256 --epochs 30 --lr 0.01 >> /home/maggie/mmat/log/CIFAR10/defense/manifold-mixup-training/preactresnet34-cifar10-manifoldmixup-cw-20220815.log 2>&1

# # PreActResNet34 manifoldmixup beta(1,1) against om-fgsm (eps=0.05)
# CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220815.py run --mode defense --defense_mode manifoldmixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode om-fgsm --attack_eps 0.05 --blackbox --exp_name preactresnet34-cifar10 --cla_model preactresnet34 --cla_network_pkl XXX-cla6-XXX --dataset cifar10 --test_adv_dataset /home/data/maggie/result-newhome/attack/fgsm/preactresnet34-cifar10/20220702/00001-omfgsm-eps-0.05-acc-49.23/attack-cifar10-dataset/latent-attack-samples/test --batch_size 256 --epochs 30 --lr 0.01 >> /home/maggie/mmat/log/CIFAR10/defense/manifold-mixup-training/preactresnet34-cifar10-manifoldmixup-om-fgsm-20220815.log 2>&1

# # PreActResNet34 manifoldmixup beta(1,1) against om-pgd (eps=0.05)
# CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220815.py run --mode defense --defense_mode manifoldmixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode om-pgd --attack_eps 0.05 --blackbox --exp_name preactresnet34-cifar10 --cla_model preactresnet34 --cla_network_pkl XXX-cla6-XXX --dataset cifar10 --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/preactresnet34-cifar10/20220703/00001-ompgd-eps-0.05-acc-17.05/attack-cifar10-dataset/latent-attack-samples/test --batch_size 256 --epochs 30 --lr 0.01 >> /home/maggie/mmat/log/CIFAR10/defense/manifold-mixup-training/preactresnet34-cifar10-manifoldmixup-om-pgd-20220815.log 2>&1