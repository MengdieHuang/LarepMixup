source ~/.bashrc
source /home/xieyi/anaconda3/bin/activate mmat

# adv50000 cat cle50000
# PreActResNet50 manifoldmixup beta(1,1) against FGSM (eps=0.1)
CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220905.py run --mode defense --defense_mode manifoldmixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode fgsm --attack_eps 0.1 --blackbox --exp_name preactresnet50-svhn --cla_model preactresnet50 --cla_network_pkl /home/data/maggie/result-newhome/train/cla-train/preactresnet50-svhn/20220627/00001-testacc-95.76/train-svhn-dataset/standard-trained-classifier-preactresnet50-on-clean-svhn-epoch-0013.pkl --dataset svhn --test_adv_dataset /home/data/maggie/result-newhome/attack/fgsm/preactresnet50-svhn/20220628/00002-fgsm-eps-0.1-acc-60.02/attack-svhn-dataset/samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/SVHN/defense/manifold-mixup-training/preactresnet50-svhn-manifoldmixup-fgsm-20220916.log 2>&1

# PreActResNet50 manifoldmixup beta(1,1) against FGSM (eps=0.1)
CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220905.py run --mode defense --defense_mode manifoldmixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode fgsm --attack_eps 0.1 --blackbox --exp_name preactresnet50-svhn --cla_model preactresnet50 --cla_network_pkl /home/data/maggie/result-newhome/train/cla-train/preactresnet50-svhn/20220627/00001-testacc-95.76/train-svhn-dataset/standard-trained-classifier-preactresnet50-on-clean-svhn-epoch-0013.pkl --dataset svhn --test_adv_dataset /home/data/maggie/result-newhome/attack/fgsm/preactresnet50-svhn/20220628/00002-fgsm-eps-0.1-acc-60.02/attack-svhn-dataset/samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/SVHN/defense/manifold-mixup-training/preactresnet50-svhn-manifoldmixup-fgsm-20220916.log 2>&1

# PreActResNet50 manifoldmixup beta(1,1) against FGSM (eps=0.1)
CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220905.py run --mode defense --defense_mode manifoldmixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode fgsm --attack_eps 0.1 --blackbox --exp_name preactresnet50-svhn --cla_model preactresnet50 --cla_network_pkl /home/data/maggie/result-newhome/train/cla-train/preactresnet50-svhn/20220627/00001-testacc-95.76/train-svhn-dataset/standard-trained-classifier-preactresnet50-on-clean-svhn-epoch-0013.pkl --dataset svhn --test_adv_dataset /home/data/maggie/result-newhome/attack/fgsm/preactresnet50-svhn/20220628/00002-fgsm-eps-0.1-acc-60.02/attack-svhn-dataset/samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/SVHN/defense/manifold-mixup-training/preactresnet50-svhn-manifoldmixup-fgsm-20220916.log 2>&1

# PreActResNet50 manifoldmixup beta(1,1) against FGSM (eps=0.1)
CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220905.py run --mode defense --defense_mode manifoldmixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode fgsm --attack_eps 0.1 --blackbox --exp_name preactresnet50-svhn --cla_model preactresnet50 --cla_network_pkl /home/data/maggie/result-newhome/train/cla-train/preactresnet50-svhn/20220627/00001-testacc-95.76/train-svhn-dataset/standard-trained-classifier-preactresnet50-on-clean-svhn-epoch-0013.pkl --dataset svhn --test_adv_dataset /home/data/maggie/result-newhome/attack/fgsm/preactresnet50-svhn/20220628/00002-fgsm-eps-0.1-acc-60.02/attack-svhn-dataset/samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/SVHN/defense/manifold-mixup-training/preactresnet50-svhn-manifoldmixup-fgsm-20220916.log 2>&1

# PreActResNet50 manifoldmixup beta(1,1) against FGSM (eps=0.1)
CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220905.py run --mode defense --defense_mode manifoldmixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode fgsm --attack_eps 0.1 --blackbox --exp_name preactresnet50-svhn --cla_model preactresnet50 --cla_network_pkl /home/data/maggie/result-newhome/train/cla-train/preactresnet50-svhn/20220627/00001-testacc-95.76/train-svhn-dataset/standard-trained-classifier-preactresnet50-on-clean-svhn-epoch-0013.pkl --dataset svhn --test_adv_dataset /home/data/maggie/result-newhome/attack/fgsm/preactresnet50-svhn/20220628/00002-fgsm-eps-0.1-acc-60.02/attack-svhn-dataset/samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/SVHN/defense/manifold-mixup-training/preactresnet50-svhn-manifoldmixup-fgsm-20220916.log 2>&1

# PreActResNet50 manifoldmixup beta(1,1) against FGSM (eps=0.1)
CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220905.py run --mode defense --defense_mode manifoldmixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode fgsm --attack_eps 0.1 --blackbox --exp_name preactresnet50-svhn --cla_model preactresnet50 --cla_network_pkl /home/data/maggie/result-newhome/train/cla-train/preactresnet50-svhn/20220627/00001-testacc-95.76/train-svhn-dataset/standard-trained-classifier-preactresnet50-on-clean-svhn-epoch-0013.pkl --dataset svhn --test_adv_dataset /home/data/maggie/result-newhome/attack/fgsm/preactresnet50-svhn/20220628/00002-fgsm-eps-0.1-acc-60.02/attack-svhn-dataset/samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/SVHN/defense/manifold-mixup-training/preactresnet50-svhn-manifoldmixup-fgsm-20220916.log 2>&1


# #-------------以上要存储at训练好的模型-----------
# # #   使用PGD对抗训练后的 cla1
# # PreActResNet50 manifoldmixup beta(1,1) against pgd (eps=0.1)
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220905.py run --mode defense --defense_mode manifoldmixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode pgd --attack_eps 0.1 --blackbox --exp_name preactresnet50-svhn --cla_model preactresnet50 --cla_network_pkl /home/data/maggie/result-newhome/defense/manifoldmixup/fgsm/basemixup-betasampler/preactresnet50-svhn/blackbox/20220906/00001/manifoldmixup-svhn-dataset/manifoldmixup-trained-classifier-preactresnet50-on-svhn-epoch-0005-manifoldmix-cla1.pkl --dataset svhn --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/preactresnet50-svhn/20220629/00002-pgd-eps-0.1-acc-35.61/attack-svhn-dataset/samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/SVHN/defense/manifold-mixup-training/preactresnet50-svhn-manifoldmixup-pgd-20220916.log 2>&1

# # PreActResNet50 manifoldmixup beta(1,1) against autoattack (eps=0.1)
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220905.py run --mode defense --defense_mode manifoldmixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode autoattack --attack_eps 0.1 --blackbox --exp_name preactresnet50-svhn --cla_model preactresnet50 --cla_network_pkl /home/data/maggie/result-newhome/defense/manifoldmixup/fgsm/basemixup-betasampler/preactresnet50-svhn/blackbox/20220906/00001/manifoldmixup-svhn-dataset/manifoldmixup-trained-classifier-preactresnet50-on-svhn-epoch-0005-manifoldmix-cla1.pkl --dataset svhn --test_adv_dataset /home/data/maggie/result-newhome/attack/autoattack/preactresnet50-svhn/20220728/00000-autoattack-eps-0.1-acc-29.94/attack-svhn-dataset/samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/SVHN/defense/manifold-mixup-training/preactresnet50-svhn-manifoldmixup-autoattack-20220916.log 2>&1

# # PreActResNet50 manifoldmixup beta(1,1) against deepfool (eps=0.1)
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220905.py run --mode defense --defense_mode manifoldmixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode deepfool --attack_eps 0.1 --blackbox --exp_name preactresnet50-svhn --cla_model preactresnet50 --cla_network_pkl /home/data/maggie/result-newhome/defense/manifoldmixup/fgsm/basemixup-betasampler/preactresnet50-svhn/blackbox/20220906/00001/manifoldmixup-svhn-dataset/manifoldmixup-trained-classifier-preactresnet50-on-svhn-epoch-0005-manifoldmix-cla1.pkl --dataset svhn --test_adv_dataset /home/data/maggie/result-newhome/attack/deepfool/preactresnet50-svhn/20220721/00000-deepfool-eps-0.1-acc-23.72/attack-svhn-dataset/samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/SVHN/defense/manifold-mixup-training/preactresnet50-svhn-manifoldmixup-deepfool-20220916.log 2>&1

# # PreActResNet50 manifoldmixup beta(1,1) against cw (eps=0.1)
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220905.py run --mode defense --defense_mode manifoldmixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode cw --attack_eps 0.1 --blackbox --exp_name preactresnet50-svhn --cla_model preactresnet50 --cla_network_pkl /home/data/maggie/result-newhome/defense/manifoldmixup/fgsm/basemixup-betasampler/preactresnet50-svhn/blackbox/20220906/00001/manifoldmixup-svhn-dataset/manifoldmixup-trained-classifier-preactresnet50-on-svhn-epoch-0005-manifoldmix-cla1.pkl --dataset svhn --test_adv_dataset /home/data/maggie/result-newhome/attack/cw/preactresnet50-svhn/20220807/00000-cw-confidence-0.0-acc-27.09/attack-svhn-dataset/samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/SVHN/defense/manifold-mixup-training/preactresnet50-svhn-manifoldmixup-cw-20220916.log 2>&1

# # PreActResNet50 manifoldmixup beta(1,1) against om-fgsm (eps=0.1)
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220905.py run --mode defense --defense_mode manifoldmixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode om-fgsm --attack_eps 0.1 --blackbox --exp_name preactresnet50-svhn --cla_model preactresnet50 --cla_network_pkl /home/data/maggie/result-newhome/defense/manifoldmixup/fgsm/basemixup-betasampler/preactresnet50-svhn/blackbox/20220906/00001/manifoldmixup-svhn-dataset/manifoldmixup-trained-classifier-preactresnet50-on-svhn-epoch-0005-manifoldmix-cla1.pkl --dataset svhn --test_adv_dataset /home/data/maggie/result-newhome/attack/fgsm/preactresnet50-svhn/20220710/00002-omfgsm-eps-0.1-acc-40.01/attack-svhn-dataset/latent-attack-samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/SVHN/defense/manifold-mixup-training/preactresnet50-svhn-manifoldmixup-om-fgsm-20220916.log 2>&1

# # PreActResNet50 manifoldmixup beta(1,1) against om-pgd (eps=0.1)
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220905.py run --mode defense --defense_mode manifoldmixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode om-pgd --attack_eps 0.1 --blackbox --exp_name preactresnet50-svhn --cla_model preactresnet50 --cla_network_pkl /home/data/maggie/result-newhome/defense/manifoldmixup/fgsm/basemixup-betasampler/preactresnet50-svhn/blackbox/20220906/00001/manifoldmixup-svhn-dataset/manifoldmixup-trained-classifier-preactresnet50-on-svhn-epoch-0005-manifoldmix-cla1.pkl --dataset svhn --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/preactresnet50-svhn/20220713/00002-ompgd-eps-0.1-acc-6.73/attack-svhn-dataset/latent-attack-samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/SVHN/defense/manifold-mixup-training/preactresnet50-svhn-manifoldmixup-om-pgd-20220916.log 2>&1

# # #   使用PGD对抗训练后的 cla2
# # PreActResNet50 manifoldmixup beta(1,1) against pgd (eps=0.1)
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220905.py run --mode defense --defense_mode manifoldmixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode pgd --attack_eps 0.1 --blackbox --exp_name preactresnet50-svhn --cla_model preactresnet50 --cla_network_pkl /home/data/maggie/result-newhome/defense/manifoldmixup/fgsm/basemixup-betasampler/preactresnet50-svhn/blackbox/20220906/00005/manifoldmixup-svhn-dataset/manifoldmixup-trained-classifier-preactresnet50-on-svhn-epoch-0011-manifoldmix-cla2.pkl --dataset svhn --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/preactresnet50-svhn/20220629/00002-pgd-eps-0.1-acc-35.61/attack-svhn-dataset/samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/SVHN/defense/manifold-mixup-training/preactresnet50-svhn-manifoldmixup-pgd-20220916.log 2>&1

# # PreActResNet50 manifoldmixup beta(1,1) against autoattack (eps=0.1)
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220905.py run --mode defense --defense_mode manifoldmixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode autoattack --attack_eps 0.1 --blackbox --exp_name preactresnet50-svhn --cla_model preactresnet50 --cla_network_pkl /home/data/maggie/result-newhome/defense/manifoldmixup/fgsm/basemixup-betasampler/preactresnet50-svhn/blackbox/20220906/00005/manifoldmixup-svhn-dataset/manifoldmixup-trained-classifier-preactresnet50-on-svhn-epoch-0011-manifoldmix-cla2.pkl --dataset svhn --test_adv_dataset /home/data/maggie/result-newhome/attack/autoattack/preactresnet50-svhn/20220728/00000-autoattack-eps-0.1-acc-29.94/attack-svhn-dataset/samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/SVHN/defense/manifold-mixup-training/preactresnet50-svhn-manifoldmixup-autoattack-20220916.log 2>&1

# # PreActResNet50 manifoldmixup beta(1,1) against deepfool (eps=0.1)
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220905.py run --mode defense --defense_mode manifoldmixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode deepfool --attack_eps 0.1 --blackbox --exp_name preactresnet50-svhn --cla_model preactresnet50 --cla_network_pkl /home/data/maggie/result-newhome/defense/manifoldmixup/fgsm/basemixup-betasampler/preactresnet50-svhn/blackbox/20220906/00005/manifoldmixup-svhn-dataset/manifoldmixup-trained-classifier-preactresnet50-on-svhn-epoch-0011-manifoldmix-cla2.pkl --dataset svhn --test_adv_dataset /home/data/maggie/result-newhome/attack/deepfool/preactresnet50-svhn/20220721/00000-deepfool-eps-0.1-acc-23.72/attack-svhn-dataset/samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/SVHN/defense/manifold-mixup-training/preactresnet50-svhn-manifoldmixup-deepfool-20220916.log 2>&1

# # PreActResNet50 manifoldmixup beta(1,1) against cw (eps=0.1)
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220905.py run --mode defense --defense_mode manifoldmixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode cw --attack_eps 0.1 --blackbox --exp_name preactresnet50-svhn --cla_model preactresnet50 --cla_network_pkl /home/data/maggie/result-newhome/defense/manifoldmixup/fgsm/basemixup-betasampler/preactresnet50-svhn/blackbox/20220906/00005/manifoldmixup-svhn-dataset/manifoldmixup-trained-classifier-preactresnet50-on-svhn-epoch-0011-manifoldmix-cla2.pkl --dataset svhn --test_adv_dataset /home/data/maggie/result-newhome/attack/cw/preactresnet50-svhn/20220807/00000-cw-confidence-0.0-acc-27.09/attack-svhn-dataset/samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/SVHN/defense/manifold-mixup-training/preactresnet50-svhn-manifoldmixup-cw-20220916.log 2>&1

# # PreActResNet50 manifoldmixup beta(1,1) against om-fgsm (eps=0.1)
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220905.py run --mode defense --defense_mode manifoldmixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode om-fgsm --attack_eps 0.1 --blackbox --exp_name preactresnet50-svhn --cla_model preactresnet50 --cla_network_pkl /home/data/maggie/result-newhome/defense/manifoldmixup/fgsm/basemixup-betasampler/preactresnet50-svhn/blackbox/20220906/00005/manifoldmixup-svhn-dataset/manifoldmixup-trained-classifier-preactresnet50-on-svhn-epoch-0011-manifoldmix-cla2.pkl --dataset svhn --test_adv_dataset /home/data/maggie/result-newhome/attack/fgsm/preactresnet50-svhn/20220710/00002-omfgsm-eps-0.1-acc-40.01/attack-svhn-dataset/latent-attack-samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/SVHN/defense/manifold-mixup-training/preactresnet50-svhn-manifoldmixup-om-fgsm-20220916.log 2>&1

# # PreActResNet50 manifoldmixup beta(1,1) against om-pgd (eps=0.1)
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220905.py run --mode defense --defense_mode manifoldmixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode om-pgd --attack_eps 0.1 --blackbox --exp_name preactresnet50-svhn --cla_model preactresnet50 --cla_network_pkl /home/data/maggie/result-newhome/defense/manifoldmixup/fgsm/basemixup-betasampler/preactresnet50-svhn/blackbox/20220906/00005/manifoldmixup-svhn-dataset/manifoldmixup-trained-classifier-preactresnet50-on-svhn-epoch-0011-manifoldmix-cla2.pkl --dataset svhn --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/preactresnet50-svhn/20220713/00002-ompgd-eps-0.1-acc-6.73/attack-svhn-dataset/latent-attack-samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/SVHN/defense/manifold-mixup-training/preactresnet50-svhn-manifoldmixup-om-pgd-20220916.log 2>&1

# # #   使用PGD对抗训练后的 cla3
# # PreActResNet50 manifoldmixup beta(1,1) against pgd (eps=0.1)
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220905.py run --mode defense --defense_mode manifoldmixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode pgd --attack_eps 0.1 --blackbox --exp_name preactresnet50-svhn --cla_model preactresnet50 --cla_network_pkl XXX-cla3-XXX --dataset svhn --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/preactresnet50-svhn/20220629/00002-pgd-eps-0.1-acc-35.61/attack-svhn-dataset/samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/SVHN/defense/manifold-mixup-training/preactresnet50-svhn-manifoldmixup-pgd-20220916.log 2>&1

# # PreActResNet50 manifoldmixup beta(1,1) against autoattack (eps=0.1)
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220905.py run --mode defense --defense_mode manifoldmixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode autoattack --attack_eps 0.1 --blackbox --exp_name preactresnet50-svhn --cla_model preactresnet50 --cla_network_pkl XXX-cla3-XXX --dataset svhn --test_adv_dataset /home/data/maggie/result-newhome/attack/autoattack/preactresnet50-svhn/20220728/00000-autoattack-eps-0.1-acc-29.94/attack-svhn-dataset/samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/SVHN/defense/manifold-mixup-training/preactresnet50-svhn-manifoldmixup-autoattack-20220916.log 2>&1

# # PreActResNet50 manifoldmixup beta(1,1) against deepfool (eps=0.1)
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220905.py run --mode defense --defense_mode manifoldmixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode deepfool --attack_eps 0.1 --blackbox --exp_name preactresnet50-svhn --cla_model preactresnet50 --cla_network_pkl XXX-cla3-XXX --dataset svhn --test_adv_dataset /home/data/maggie/result-newhome/attack/deepfool/preactresnet50-svhn/20220721/00000-deepfool-eps-0.1-acc-23.72/attack-svhn-dataset/samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/SVHN/defense/manifold-mixup-training/preactresnet50-svhn-manifoldmixup-deepfool-20220916.log 2>&1

# # PreActResNet50 manifoldmixup beta(1,1) against cw (eps=0.1)
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220905.py run --mode defense --defense_mode manifoldmixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode cw --attack_eps 0.1 --blackbox --exp_name preactresnet50-svhn --cla_model preactresnet50 --cla_network_pkl XXX-cla3-XXX --dataset svhn --test_adv_dataset /home/data/maggie/result-newhome/attack/cw/preactresnet50-svhn/20220807/00000-cw-confidence-0.0-acc-27.09/attack-svhn-dataset/samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/SVHN/defense/manifold-mixup-training/preactresnet50-svhn-manifoldmixup-cw-20220916.log 2>&1

# # PreActResNet50 manifoldmixup beta(1,1) against om-fgsm (eps=0.1)
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220905.py run --mode defense --defense_mode manifoldmixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode om-fgsm --attack_eps 0.1 --blackbox --exp_name preactresnet50-svhn --cla_model preactresnet50 --cla_network_pkl XXX-cla3-XXX --dataset svhn --test_adv_dataset /home/data/maggie/result-newhome/attack/fgsm/preactresnet50-svhn/20220710/00002-omfgsm-eps-0.1-acc-40.01/attack-svhn-dataset/latent-attack-samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/SVHN/defense/manifold-mixup-training/preactresnet50-svhn-manifoldmixup-om-fgsm-20220916.log 2>&1

# # PreActResNet50 manifoldmixup beta(1,1) against om-pgd (eps=0.1)
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220905.py run --mode defense --defense_mode manifoldmixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode om-pgd --attack_eps 0.1 --blackbox --exp_name preactresnet50-svhn --cla_model preactresnet50 --cla_network_pkl XXX-cla3-XXX --dataset svhn --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/preactresnet50-svhn/20220713/00002-ompgd-eps-0.1-acc-6.73/attack-svhn-dataset/latent-attack-samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/SVHN/defense/manifold-mixup-training/preactresnet50-svhn-manifoldmixup-om-pgd-20220916.log 2>&1

# # #   使用PGD对抗训练后的 cla4
# # PreActResNet50 manifoldmixup beta(1,1) against pgd (eps=0.1)
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220905.py run --mode defense --defense_mode manifoldmixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode pgd --attack_eps 0.1 --blackbox --exp_name preactresnet50-svhn --cla_model preactresnet50 --cla_network_pkl XXX-cla4-XXX --dataset svhn --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/preactresnet50-svhn/20220629/00002-pgd-eps-0.1-acc-35.61/attack-svhn-dataset/samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/SVHN/defense/manifold-mixup-training/preactresnet50-svhn-manifoldmixup-pgd-20220916.log 2>&1

# # PreActResNet50 manifoldmixup beta(1,1) against autoattack (eps=0.1)
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220905.py run --mode defense --defense_mode manifoldmixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode autoattack --attack_eps 0.1 --blackbox --exp_name preactresnet50-svhn --cla_model preactresnet50 --cla_network_pkl XXX-cla4-XXX --dataset svhn --test_adv_dataset /home/data/maggie/result-newhome/attack/autoattack/preactresnet50-svhn/20220728/00000-autoattack-eps-0.1-acc-29.94/attack-svhn-dataset/samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/SVHN/defense/manifold-mixup-training/preactresnet50-svhn-manifoldmixup-autoattack-20220916.log 2>&1

# # PreActResNet50 manifoldmixup beta(1,1) against deepfool (eps=0.1)
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220905.py run --mode defense --defense_mode manifoldmixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode deepfool --attack_eps 0.1 --blackbox --exp_name preactresnet50-svhn --cla_model preactresnet50 --cla_network_pkl XXX-cla4-XXX --dataset svhn --test_adv_dataset /home/data/maggie/result-newhome/attack/deepfool/preactresnet50-svhn/20220721/00000-deepfool-eps-0.1-acc-23.72/attack-svhn-dataset/samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/SVHN/defense/manifold-mixup-training/preactresnet50-svhn-manifoldmixup-deepfool-20220916.log 2>&1

# # PreActResNet50 manifoldmixup beta(1,1) against cw (eps=0.1)
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220905.py run --mode defense --defense_mode manifoldmixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode cw --attack_eps 0.1 --blackbox --exp_name preactresnet50-svhn --cla_model preactresnet50 --cla_network_pkl XXX-cla4-XXX --dataset svhn --test_adv_dataset /home/data/maggie/result-newhome/attack/cw/preactresnet50-svhn/20220807/00000-cw-confidence-0.0-acc-27.09/attack-svhn-dataset/samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/SVHN/defense/manifold-mixup-training/preactresnet50-svhn-manifoldmixup-cw-20220916.log 2>&1

# # PreActResNet50 manifoldmixup beta(1,1) against om-fgsm (eps=0.1)
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220905.py run --mode defense --defense_mode manifoldmixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode om-fgsm --attack_eps 0.1 --blackbox --exp_name preactresnet50-svhn --cla_model preactresnet50 --cla_network_pkl XXX-cla4-XXX --dataset svhn --test_adv_dataset /home/data/maggie/result-newhome/attack/fgsm/preactresnet50-svhn/20220710/00002-omfgsm-eps-0.1-acc-40.01/attack-svhn-dataset/latent-attack-samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/SVHN/defense/manifold-mixup-training/preactresnet50-svhn-manifoldmixup-om-fgsm-20220916.log 2>&1

# # PreActResNet50 manifoldmixup beta(1,1) against om-pgd (eps=0.1)
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220905.py run --mode defense --defense_mode manifoldmixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode om-pgd --attack_eps 0.1 --blackbox --exp_name preactresnet50-svhn --cla_model preactresnet50 --cla_network_pkl XXX-cla4-XXX --dataset svhn --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/preactresnet50-svhn/20220713/00002-ompgd-eps-0.1-acc-6.73/attack-svhn-dataset/latent-attack-samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/SVHN/defense/manifold-mixup-training/preactresnet50-svhn-manifoldmixup-om-pgd-20220916.log 2>&1

# # #   使用PGD对抗训练后的 cla5
# # PreActResNet50 manifoldmixup beta(1,1) against pgd (eps=0.1)
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220905.py run --mode defense --defense_mode manifoldmixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode pgd --attack_eps 0.1 --blackbox --exp_name preactresnet50-svhn --cla_model preactresnet50 --cla_network_pkl XXX-cla5-XXX --dataset svhn --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/preactresnet50-svhn/20220629/00002-pgd-eps-0.1-acc-35.61/attack-svhn-dataset/samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/SVHN/defense/manifold-mixup-training/preactresnet50-svhn-manifoldmixup-pgd-20220916.log 2>&1

# # PreActResNet50 manifoldmixup beta(1,1) against autoattack (eps=0.1)
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220905.py run --mode defense --defense_mode manifoldmixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode autoattack --attack_eps 0.1 --blackbox --exp_name preactresnet50-svhn --cla_model preactresnet50 --cla_network_pkl XXX-cla5-XXX --dataset svhn --test_adv_dataset /home/data/maggie/result-newhome/attack/autoattack/preactresnet50-svhn/20220728/00000-autoattack-eps-0.1-acc-29.94/attack-svhn-dataset/samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/SVHN/defense/manifold-mixup-training/preactresnet50-svhn-manifoldmixup-autoattack-20220916.log 2>&1

# # PreActResNet50 manifoldmixup beta(1,1) against deepfool (eps=0.1)
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220905.py run --mode defense --defense_mode manifoldmixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode deepfool --attack_eps 0.1 --blackbox --exp_name preactresnet50-svhn --cla_model preactresnet50 --cla_network_pkl XXX-cla5-XXX --dataset svhn --test_adv_dataset /home/data/maggie/result-newhome/attack/deepfool/preactresnet50-svhn/20220721/00000-deepfool-eps-0.1-acc-23.72/attack-svhn-dataset/samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/SVHN/defense/manifold-mixup-training/preactresnet50-svhn-manifoldmixup-deepfool-20220916.log 2>&1

# # PreActResNet50 manifoldmixup beta(1,1) against cw (eps=0.1)
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220905.py run --mode defense --defense_mode manifoldmixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode cw --attack_eps 0.1 --blackbox --exp_name preactresnet50-svhn --cla_model preactresnet50 --cla_network_pkl XXX-cla5-XXX --dataset svhn --test_adv_dataset /home/data/maggie/result-newhome/attack/cw/preactresnet50-svhn/20220807/00000-cw-confidence-0.0-acc-27.09/attack-svhn-dataset/samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/SVHN/defense/manifold-mixup-training/preactresnet50-svhn-manifoldmixup-cw-20220916.log 2>&1

# # PreActResNet50 manifoldmixup beta(1,1) against om-fgsm (eps=0.1)
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220905.py run --mode defense --defense_mode manifoldmixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode om-fgsm --attack_eps 0.1 --blackbox --exp_name preactresnet50-svhn --cla_model preactresnet50 --cla_network_pkl XXX-cla5-XXX --dataset svhn --test_adv_dataset /home/data/maggie/result-newhome/attack/fgsm/preactresnet50-svhn/20220710/00002-omfgsm-eps-0.1-acc-40.01/attack-svhn-dataset/latent-attack-samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/SVHN/defense/manifold-mixup-training/preactresnet50-svhn-manifoldmixup-om-fgsm-20220916.log 2>&1

# # PreActResNet50 manifoldmixup beta(1,1) against om-pgd (eps=0.1)
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220905.py run --mode defense --defense_mode manifoldmixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode om-pgd --attack_eps 0.1 --blackbox --exp_name preactresnet50-svhn --cla_model preactresnet50 --cla_network_pkl XXX-cla5-XXX --dataset svhn --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/preactresnet50-svhn/20220713/00002-ompgd-eps-0.1-acc-6.73/attack-svhn-dataset/latent-attack-samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/SVHN/defense/manifold-mixup-training/preactresnet50-svhn-manifoldmixup-om-pgd-20220916.log 2>&1

# # #   使用PGD对抗训练后的 cla6
# # PreActResNet50 manifoldmixup beta(1,1) against pgd (eps=0.1)
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220905.py run --mode defense --defense_mode manifoldmixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode pgd --attack_eps 0.1 --blackbox --exp_name preactresnet50-svhn --cla_model preactresnet50 --cla_network_pkl XXX-cla6-XXX --dataset svhn --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/preactresnet50-svhn/20220629/00002-pgd-eps-0.1-acc-35.61/attack-svhn-dataset/samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/SVHN/defense/manifold-mixup-training/preactresnet50-svhn-manifoldmixup-pgd-20220916.log 2>&1

# # PreActResNet50 manifoldmixup beta(1,1) against autoattack (eps=0.1)
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220905.py run --mode defense --defense_mode manifoldmixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode autoattack --attack_eps 0.1 --blackbox --exp_name preactresnet50-svhn --cla_model preactresnet50 --cla_network_pkl XXX-cla6-XXX --dataset svhn --test_adv_dataset /home/data/maggie/result-newhome/attack/autoattack/preactresnet50-svhn/20220728/00000-autoattack-eps-0.1-acc-29.94/attack-svhn-dataset/samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/SVHN/defense/manifold-mixup-training/preactresnet50-svhn-manifoldmixup-autoattack-20220916.log 2>&1

# # PreActResNet50 manifoldmixup beta(1,1) against deepfool (eps=0.1)
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220905.py run --mode defense --defense_mode manifoldmixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode deepfool --attack_eps 0.1 --blackbox --exp_name preactresnet50-svhn --cla_model preactresnet50 --cla_network_pkl XXX-cla6-XXX --dataset svhn --test_adv_dataset /home/data/maggie/result-newhome/attack/deepfool/preactresnet50-svhn/20220721/00000-deepfool-eps-0.1-acc-23.72/attack-svhn-dataset/samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/SVHN/defense/manifold-mixup-training/preactresnet50-svhn-manifoldmixup-deepfool-20220916.log 2>&1

# # PreActResNet50 manifoldmixup beta(1,1) against cw (eps=0.1)
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220905.py run --mode defense --defense_mode manifoldmixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode cw --attack_eps 0.1 --blackbox --exp_name preactresnet50-svhn --cla_model preactresnet50 --cla_network_pkl XXX-cla6-XXX --dataset svhn --test_adv_dataset /home/data/maggie/result-newhome/attack/cw/preactresnet50-svhn/20220807/00000-cw-confidence-0.0-acc-27.09/attack-svhn-dataset/samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/SVHN/defense/manifold-mixup-training/preactresnet50-svhn-manifoldmixup-cw-20220916.log 2>&1

# # PreActResNet50 manifoldmixup beta(1,1) against om-fgsm (eps=0.1)
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220905.py run --mode defense --defense_mode manifoldmixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode om-fgsm --attack_eps 0.1 --blackbox --exp_name preactresnet50-svhn --cla_model preactresnet50 --cla_network_pkl XXX-cla6-XXX --dataset svhn --test_adv_dataset /home/data/maggie/result-newhome/attack/fgsm/preactresnet50-svhn/20220710/00002-omfgsm-eps-0.1-acc-40.01/attack-svhn-dataset/latent-attack-samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/SVHN/defense/manifold-mixup-training/preactresnet50-svhn-manifoldmixup-om-fgsm-20220916.log 2>&1

# # PreActResNet50 manifoldmixup beta(1,1) against om-pgd (eps=0.1)
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220905.py run --mode defense --defense_mode manifoldmixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode om-pgd --attack_eps 0.1 --blackbox --exp_name preactresnet50-svhn --cla_model preactresnet50 --cla_network_pkl XXX-cla6-XXX --dataset svhn --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/preactresnet50-svhn/20220713/00002-ompgd-eps-0.1-acc-6.73/attack-svhn-dataset/latent-attack-samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/SVHN/defense/manifold-mixup-training/preactresnet50-svhn-manifoldmixup-om-pgd-20220916.log 2>&1