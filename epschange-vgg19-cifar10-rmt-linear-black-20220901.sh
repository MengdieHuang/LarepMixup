source ~/.bashrc
source /home/xieyi/anaconda3/bin/activate mmat

# adv50000 cat cle50000
# VGG19 linear rmt beta(1,1) against PGD (eps=0.02)
CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220901.py run --mode defense --defense_mode rmt --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode pgd --attack_eps 0.02 --blackbox --exp_name vgg19-cifar10 --cla_model vgg19 --cla_network_pkl /home/data/maggie/result-newhome/train/cla-train/vgg19-cifar10/20210908/00002-testacc-0.8705/train-cifar10-dataset/standard-trained-classifier-vgg19-on-clean-cifar10-epoch-0013.pkl --gen_model stylegan2ada --gen_network_pkl /home/data/maggie/result-newhome/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_dataset /home/data/maggie/result-newhome/project/stylegan2ada-cifar10/20220716/00000/project-cifar10-trainset --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/vgg19-cifar10/20220625/00000-pgd-eps-0.02-acc-58.30/attack-cifar10-dataset/samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/CIFAR10/defense/representation-mixup-training/vgg19-cifar10-rmt-linear-pgd-20220901.log 2>&1

# VGG19 linear rmt beta(1,1) against PGD (eps=0.02)
CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220901.py run --mode defense --defense_mode rmt --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode pgd --attack_eps 0.02 --blackbox --exp_name vgg19-cifar10 --cla_model vgg19 --cla_network_pkl /home/data/maggie/result-newhome/train/cla-train/vgg19-cifar10/20210908/00002-testacc-0.8705/train-cifar10-dataset/standard-trained-classifier-vgg19-on-clean-cifar10-epoch-0013.pkl --gen_model stylegan2ada --gen_network_pkl /home/data/maggie/result-newhome/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_dataset /home/data/maggie/result-newhome/project/stylegan2ada-cifar10/20220716/00000/project-cifar10-trainset --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/vgg19-cifar10/20220625/00000-pgd-eps-0.02-acc-58.30/attack-cifar10-dataset/samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/CIFAR10/defense/representation-mixup-training/vgg19-cifar10-rmt-linear-pgd-20220901.log 2>&1

# VGG19 linear rmt beta(1,1) against PGD (eps=0.02)
CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220901.py run --mode defense --defense_mode rmt --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode pgd --attack_eps 0.02 --blackbox --exp_name vgg19-cifar10 --cla_model vgg19 --cla_network_pkl /home/data/maggie/result-newhome/train/cla-train/vgg19-cifar10/20210908/00002-testacc-0.8705/train-cifar10-dataset/standard-trained-classifier-vgg19-on-clean-cifar10-epoch-0013.pkl --gen_model stylegan2ada --gen_network_pkl /home/data/maggie/result-newhome/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_dataset /home/data/maggie/result-newhome/project/stylegan2ada-cifar10/20220716/00000/project-cifar10-trainset --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/vgg19-cifar10/20220625/00000-pgd-eps-0.02-acc-58.30/attack-cifar10-dataset/samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/CIFAR10/defense/representation-mixup-training/vgg19-cifar10-rmt-linear-pgd-20220901.log 2>&1

# #-------------以上要存储at训练好的模型-----------
# # #   混合训练得到cla1防御PGD 
# # VGG19 linear rmt beta(1,1) against pgd (eps=0.05)
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220901.py run --mode defense --defense_mode rmt --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode pgd --attack_eps 0.05 --blackbox --exp_name vgg19-cifar10 --cla_model vgg19 --cla_network_pkl XXX-cla1-XXX --gen_model stylegan2ada --gen_network_pkl /home/data/maggie/result-newhome/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_dataset /home/data/maggie/result-newhome/project/stylegan2ada-cifar10/20220716/00000/project-cifar10-trainset --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/vgg19-cifar10/20220626/00000-pgd-eps-0.05-acc-35.92/attack-cifar10-dataset/samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/CIFAR10/defense/representation-mixup-training/vgg19-cifar10-rmt-linear-pgd-20220901.log 2>&1

# # VGG19 linear rmt beta(1,1) against pgd (eps=0.1)
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220901.py run --mode defense --defense_mode rmt --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode autoattack --attack_eps 0.1 --blackbox --exp_name vgg19-cifar10 --cla_model vgg19 --cla_network_pkl XXX-cla1-XXX --gen_model stylegan2ada --gen_network_pkl /home/data/maggie/result-newhome/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_dataset /home/data/maggie/result-newhome/project/stylegan2ada-cifar10/20220716/00000/project-cifar10-trainset --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/vgg19-cifar10/20220626/00001-pgd-eps-0.1-acc-5.99/attack-cifar10-dataset/samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/CIFAR10/defense/representation-mixup-training/vgg19-cifar10-rmt-linear-pgd-20220901.log 2>&1

# # VGG19 linear rmt beta(1,1) against pgd (eps=0.2)
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220901.py run --mode defense --defense_mode rmt --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode deepfool --attack_eps 0.2 --blackbox --exp_name vgg19-cifar10 --cla_model vgg19 --cla_network_pkl XXX-cla1-XXX --gen_model stylegan2ada --gen_network_pkl /home/data/maggie/result-newhome/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_dataset /home/data/maggie/result-newhome/project/stylegan2ada-cifar10/20220716/00000/project-cifar10-trainset --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/vgg19-cifar10/20220626/00002-pgd-eps-0.2-acc-1.25/attack-cifar10-dataset/samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/CIFAR10/defense/representation-mixup-training/vgg19-cifar10-rmt-linear-pgd-20220901.log 2>&1

# # VGG19 linear rmt beta(1,1) against pgd (eps=0.3)
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220901.py run --mode defense --defense_mode rmt --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode cw --attack_eps 0.3 --blackbox --exp_name vgg19-cifar10 --cla_model vgg19 --cla_network_pkl XXX-cla1-XXX --gen_model stylegan2ada --gen_network_pkl /home/data/maggie/result-newhome/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_dataset /home/data/maggie/result-newhome/project/stylegan2ada-cifar10/20220716/00000/project-cifar10-trainset --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/vgg19-cifar10/20220626/00003-pgd-eps-0.3-acc-0.25/attack-cifar10-dataset/samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/CIFAR10/defense/representation-mixup-training/vgg19-cifar10-rmt-linear-pgd-20220901.log 2>&1


# # #   混合训练得到cla1防御OM-PGD 
# # VGG19 linear rmt beta(1,1) against om-pgd (eps=0.02)
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220901.py run --mode defense --defense_mode rmt --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode om-pgd --attack_eps 0.02 --blackbox --exp_name vgg19-cifar10 --cla_model vgg19 --cla_network_pkl XXX-cla1-XXX --gen_model stylegan2ada --gen_network_pkl /home/data/maggie/result-newhome/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_dataset /home/data/maggie/result-newhome/project/stylegan2ada-cifar10/20220716/00000/project-cifar10-trainset --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/vgg19-cifar10/20220705/00000-ompgd-eps-0.02-acc-54.71/attack-cifar10-dataset/latent-attack-samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/CIFAR10/defense/representation-mixup-training/vgg19-cifar10-rmt-linear-om-pgd-20220901.log 2>&1

# # VGG19 linear rmt beta(1,1) against om-pgd (eps=0.05)
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220901.py run --mode defense --defense_mode rmt --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode om-pgd --attack_eps 0.05 --blackbox --exp_name vgg19-cifar10 --cla_model vgg19 --cla_network_pkl XXX-cla1-XXX --gen_model stylegan2ada --gen_network_pkl /home/data/maggie/result-newhome/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_dataset /home/data/maggie/result-newhome/project/stylegan2ada-cifar10/20220716/00000/project-cifar10-trainset --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/vgg19-cifar10/20220706/00000-ompgd-eps-0.05-acc-22.54/attack-cifar10-dataset/latent-attack-samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/CIFAR10/defense/representation-mixup-training/vgg19-cifar10-rmt-linear-om-pgd-20220901.log 2>&1

# # VGG19 linear rmt beta(1,1) against om-pgd (eps=0.1)
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220901.py run --mode defense --defense_mode rmt --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode om-pgd --attack_eps 0.1 --blackbox --exp_name vgg19-cifar10 --cla_model vgg19 --cla_network_pkl XXX-cla1-XXX --gen_model stylegan2ada --gen_network_pkl /home/data/maggie/result-newhome/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_dataset /home/data/maggie/result-newhome/project/stylegan2ada-cifar10/20220716/00000/project-cifar10-trainset --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/vgg19-cifar10/20220706/00001-ompgd-eps-0.1-acc-3.76/attack-cifar10-dataset/latent-attack-samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/CIFAR10/defense/representation-mixup-training/vgg19-cifar10-rmt-linear-om-pgd-20220901.log 2>&1

# # VGG19 linear rmt beta(1,1) against om-pgd (eps=0.2)
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220901.py run --mode defense --defense_mode rmt --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode om-pgd --attack_eps 0.2 --blackbox --exp_name vgg19-cifar10 --cla_model vgg19 --cla_network_pkl XXX-cla1-XXX --gen_model stylegan2ada --gen_network_pkl /home/data/maggie/result-newhome/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_dataset /home/data/maggie/result-newhome/project/stylegan2ada-cifar10/20220716/00000/project-cifar10-trainset --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/vgg19-cifar10/20220706/00002-ompgd-eps-0.2-acc-0.98/attack-cifar10-dataset/latent-attack-samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/CIFAR10/defense/representation-mixup-training/vgg19-cifar10-rmt-linear-om-pgd-20220901.log 2>&1

# # VGG19 linear rmt beta(1,1) against om-pgd (eps=0.3)
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220901.py run --mode defense --defense_mode rmt --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode om-pgd --attack_eps 0.3 --blackbox --exp_name vgg19-cifar10 --cla_model vgg19 --cla_network_pkl XXX-cla1-XXX --gen_model stylegan2ada --gen_network_pkl /home/data/maggie/result-newhome/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_dataset /home/data/maggie/result-newhome/project/stylegan2ada-cifar10/20220716/00000/project-cifar10-trainset --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/vgg19-cifar10/20220706/00003-ompgd-eps-0.3-acc-0.61/attack-cifar10-dataset/latent-attack-samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/CIFAR10/defense/representation-mixup-training/vgg19-cifar10-rmt-linear-om-pgd-20220901.log 2>&1

#-----------------
# # #   混合训练得到cla2防御PGD 
# # VGG19 linear rmt beta(1,1) against pgd (eps=0.05)
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220901.py run --mode defense --defense_mode rmt --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode pgd --attack_eps 0.05 --blackbox --exp_name vgg19-cifar10 --cla_model vgg19 --cla_network_pkl XXX-cla2-XXX --gen_model stylegan2ada --gen_network_pkl /home/data/maggie/result-newhome/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_dataset /home/data/maggie/result-newhome/project/stylegan2ada-cifar10/20220716/00000/project-cifar10-trainset --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/vgg19-cifar10/20220626/00000-pgd-eps-0.05-acc-35.92/attack-cifar10-dataset/samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/CIFAR10/defense/representation-mixup-training/vgg19-cifar10-rmt-linear-pgd-20220901.log 2>&1

# # VGG19 linear rmt beta(1,1) against pgd (eps=0.1)
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220901.py run --mode defense --defense_mode rmt --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode autoattack --attack_eps 0.1 --blackbox --exp_name vgg19-cifar10 --cla_model vgg19 --cla_network_pkl XXX-cla2-XXX --gen_model stylegan2ada --gen_network_pkl /home/data/maggie/result-newhome/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_dataset /home/data/maggie/result-newhome/project/stylegan2ada-cifar10/20220716/00000/project-cifar10-trainset --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/vgg19-cifar10/20220626/00001-pgd-eps-0.1-acc-5.99/attack-cifar10-dataset/samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/CIFAR10/defense/representation-mixup-training/vgg19-cifar10-rmt-linear-pgd-20220901.log 2>&1

# # VGG19 linear rmt beta(1,1) against pgd (eps=0.2)
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220901.py run --mode defense --defense_mode rmt --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode deepfool --attack_eps 0.2 --blackbox --exp_name vgg19-cifar10 --cla_model vgg19 --cla_network_pkl XXX-cla2-XXX --gen_model stylegan2ada --gen_network_pkl /home/data/maggie/result-newhome/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_dataset /home/data/maggie/result-newhome/project/stylegan2ada-cifar10/20220716/00000/project-cifar10-trainset --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/vgg19-cifar10/20220626/00002-pgd-eps-0.2-acc-1.25/attack-cifar10-dataset/samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/CIFAR10/defense/representation-mixup-training/vgg19-cifar10-rmt-linear-pgd-20220901.log 2>&1

# # VGG19 linear rmt beta(1,1) against pgd (eps=0.3)
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220901.py run --mode defense --defense_mode rmt --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode cw --attack_eps 0.3 --blackbox --exp_name vgg19-cifar10 --cla_model vgg19 --cla_network_pkl XXX-cla2-XXX --gen_model stylegan2ada --gen_network_pkl /home/data/maggie/result-newhome/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_dataset /home/data/maggie/result-newhome/project/stylegan2ada-cifar10/20220716/00000/project-cifar10-trainset --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/vgg19-cifar10/20220626/00003-pgd-eps-0.3-acc-0.25/attack-cifar10-dataset/samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/CIFAR10/defense/representation-mixup-training/vgg19-cifar10-rmt-linear-pgd-20220901.log 2>&1

# # #   混合训练得到cla2防御OM-PGD 
# # VGG19 linear rmt beta(1,1) against om-pgd (eps=0.02)
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220901.py run --mode defense --defense_mode rmt --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode om-pgd --attack_eps 0.02 --blackbox --exp_name vgg19-cifar10 --cla_model vgg19 --cla_network_pkl XXX-cla2-XXX --gen_model stylegan2ada --gen_network_pkl /home/data/maggie/result-newhome/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_dataset /home/data/maggie/result-newhome/project/stylegan2ada-cifar10/20220716/00000/project-cifar10-trainset --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/vgg19-cifar10/20220705/00000-ompgd-eps-0.02-acc-54.71/attack-cifar10-dataset/latent-attack-samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/CIFAR10/defense/representation-mixup-training/vgg19-cifar10-rmt-linear-om-pgd-20220901.log 2>&1

# # VGG19 linear rmt beta(1,1) against om-pgd (eps=0.05)
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220901.py run --mode defense --defense_mode rmt --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode om-pgd --attack_eps 0.05 --blackbox --exp_name vgg19-cifar10 --cla_model vgg19 --cla_network_pkl XXX-cla2-XXX --gen_model stylegan2ada --gen_network_pkl /home/data/maggie/result-newhome/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_dataset /home/data/maggie/result-newhome/project/stylegan2ada-cifar10/20220716/00000/project-cifar10-trainset --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/vgg19-cifar10/20220706/00000-ompgd-eps-0.05-acc-22.54/attack-cifar10-dataset/latent-attack-samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/CIFAR10/defense/representation-mixup-training/vgg19-cifar10-rmt-linear-om-pgd-20220901.log 2>&1

# # VGG19 linear rmt beta(1,1) against om-pgd (eps=0.1)
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220901.py run --mode defense --defense_mode rmt --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode om-pgd --attack_eps 0.1 --blackbox --exp_name vgg19-cifar10 --cla_model vgg19 --cla_network_pkl XXX-cla2-XXX --gen_model stylegan2ada --gen_network_pkl /home/data/maggie/result-newhome/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_dataset /home/data/maggie/result-newhome/project/stylegan2ada-cifar10/20220716/00000/project-cifar10-trainset --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/vgg19-cifar10/20220706/00001-ompgd-eps-0.1-acc-3.76/attack-cifar10-dataset/latent-attack-samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/CIFAR10/defense/representation-mixup-training/vgg19-cifar10-rmt-linear-om-pgd-20220901.log 2>&1

# # VGG19 linear rmt beta(1,1) against om-pgd (eps=0.2)
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220901.py run --mode defense --defense_mode rmt --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode om-pgd --attack_eps 0.2 --blackbox --exp_name vgg19-cifar10 --cla_model vgg19 --cla_network_pkl XXX-cla2-XXX --gen_model stylegan2ada --gen_network_pkl /home/data/maggie/result-newhome/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_dataset /home/data/maggie/result-newhome/project/stylegan2ada-cifar10/20220716/00000/project-cifar10-trainset --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/vgg19-cifar10/20220706/00002-ompgd-eps-0.2-acc-0.98/attack-cifar10-dataset/latent-attack-samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/CIFAR10/defense/representation-mixup-training/vgg19-cifar10-rmt-linear-om-pgd-20220901.log 2>&1

# # VGG19 linear rmt beta(1,1) against om-pgd (eps=0.3)
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220901.py run --mode defense --defense_mode rmt --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode om-pgd --attack_eps 0.3 --blackbox --exp_name vgg19-cifar10 --cla_model vgg19 --cla_network_pkl XXX-cla2-XXX --gen_model stylegan2ada --gen_network_pkl /home/data/maggie/result-newhome/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_dataset /home/data/maggie/result-newhome/project/stylegan2ada-cifar10/20220716/00000/project-cifar10-trainset --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/vgg19-cifar10/20220706/00003-ompgd-eps-0.3-acc-0.61/attack-cifar10-dataset/latent-attack-samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/CIFAR10/defense/representation-mixup-training/vgg19-cifar10-rmt-linear-om-pgd-20220901.log 2>&1


#-----------------
# # #   混合训练得到cla3防御PGD 
# # VGG19 linear rmt beta(1,1) against pgd (eps=0.05)
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220901.py run --mode defense --defense_mode rmt --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode pgd --attack_eps 0.05 --blackbox --exp_name vgg19-cifar10 --cla_model vgg19 --cla_network_pkl XXX-cla3-XXX --gen_model stylegan2ada --gen_network_pkl /home/data/maggie/result-newhome/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_dataset /home/data/maggie/result-newhome/project/stylegan2ada-cifar10/20220716/00000/project-cifar10-trainset --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/vgg19-cifar10/20220626/00000-pgd-eps-0.05-acc-35.92/attack-cifar10-dataset/samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/CIFAR10/defense/representation-mixup-training/vgg19-cifar10-rmt-linear-pgd-20220901.log 2>&1

# # VGG19 linear rmt beta(1,1) against pgd (eps=0.1)
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220901.py run --mode defense --defense_mode rmt --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode autoattack --attack_eps 0.1 --blackbox --exp_name vgg19-cifar10 --cla_model vgg19 --cla_network_pkl XXX-cla3-XXX --gen_model stylegan2ada --gen_network_pkl /home/data/maggie/result-newhome/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_dataset /home/data/maggie/result-newhome/project/stylegan2ada-cifar10/20220716/00000/project-cifar10-trainset --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/vgg19-cifar10/20220626/00001-pgd-eps-0.1-acc-5.99/attack-cifar10-dataset/samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/CIFAR10/defense/representation-mixup-training/vgg19-cifar10-rmt-linear-pgd-20220901.log 2>&1

# # VGG19 linear rmt beta(1,1) against pgd (eps=0.2)
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220901.py run --mode defense --defense_mode rmt --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode deepfool --attack_eps 0.2 --blackbox --exp_name vgg19-cifar10 --cla_model vgg19 --cla_network_pkl XXX-cla3-XXX --gen_model stylegan2ada --gen_network_pkl /home/data/maggie/result-newhome/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_dataset /home/data/maggie/result-newhome/project/stylegan2ada-cifar10/20220716/00000/project-cifar10-trainset --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/vgg19-cifar10/20220626/00002-pgd-eps-0.2-acc-1.25/attack-cifar10-dataset/samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/CIFAR10/defense/representation-mixup-training/vgg19-cifar10-rmt-linear-pgd-20220901.log 2>&1

# # VGG19 linear rmt beta(1,1) against pgd (eps=0.3)
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220901.py run --mode defense --defense_mode rmt --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode cw --attack_eps 0.3 --blackbox --exp_name vgg19-cifar10 --cla_model vgg19 --cla_network_pkl XXX-cla3-XXX --gen_model stylegan2ada --gen_network_pkl /home/data/maggie/result-newhome/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_dataset /home/data/maggie/result-newhome/project/stylegan2ada-cifar10/20220716/00000/project-cifar10-trainset --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/vgg19-cifar10/20220626/00003-pgd-eps-0.3-acc-0.25/attack-cifar10-dataset/samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/CIFAR10/defense/representation-mixup-training/vgg19-cifar10-rmt-linear-pgd-20220901.log 2>&1

# # #   混合训练得到cla3防御OM-PGD 
# # VGG19 linear rmt beta(1,1) against om-pgd (eps=0.02)
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220901.py run --mode defense --defense_mode rmt --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode om-pgd --attack_eps 0.02 --blackbox --exp_name vgg19-cifar10 --cla_model vgg19 --cla_network_pkl XXX-cla3-XXX --gen_model stylegan2ada --gen_network_pkl /home/data/maggie/result-newhome/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_dataset /home/data/maggie/result-newhome/project/stylegan2ada-cifar10/20220716/00000/project-cifar10-trainset --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/vgg19-cifar10/20220705/00000-ompgd-eps-0.02-acc-54.71/attack-cifar10-dataset/latent-attack-samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/CIFAR10/defense/representation-mixup-training/vgg19-cifar10-rmt-linear-om-pgd-20220901.log 2>&1

# # VGG19 linear rmt beta(1,1) against om-pgd (eps=0.05)
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220901.py run --mode defense --defense_mode rmt --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode om-pgd --attack_eps 0.05 --blackbox --exp_name vgg19-cifar10 --cla_model vgg19 --cla_network_pkl XXX-cla3-XXX --gen_model stylegan2ada --gen_network_pkl /home/data/maggie/result-newhome/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_dataset /home/data/maggie/result-newhome/project/stylegan2ada-cifar10/20220716/00000/project-cifar10-trainset --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/vgg19-cifar10/20220706/00000-ompgd-eps-0.05-acc-22.54/attack-cifar10-dataset/latent-attack-samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/CIFAR10/defense/representation-mixup-training/vgg19-cifar10-rmt-linear-om-pgd-20220901.log 2>&1

# # VGG19 linear rmt beta(1,1) against om-pgd (eps=0.1)
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220901.py run --mode defense --defense_mode rmt --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode om-pgd --attack_eps 0.1 --blackbox --exp_name vgg19-cifar10 --cla_model vgg19 --cla_network_pkl XXX-cla3-XXX --gen_model stylegan2ada --gen_network_pkl /home/data/maggie/result-newhome/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_dataset /home/data/maggie/result-newhome/project/stylegan2ada-cifar10/20220716/00000/project-cifar10-trainset --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/vgg19-cifar10/20220706/00001-ompgd-eps-0.1-acc-3.76/attack-cifar10-dataset/latent-attack-samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/CIFAR10/defense/representation-mixup-training/vgg19-cifar10-rmt-linear-om-pgd-20220901.log 2>&1

# # VGG19 linear rmt beta(1,1) against om-pgd (eps=0.2)
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220901.py run --mode defense --defense_mode rmt --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode om-pgd --attack_eps 0.2 --blackbox --exp_name vgg19-cifar10 --cla_model vgg19 --cla_network_pkl XXX-cla3-XXX --gen_model stylegan2ada --gen_network_pkl /home/data/maggie/result-newhome/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_dataset /home/data/maggie/result-newhome/project/stylegan2ada-cifar10/20220716/00000/project-cifar10-trainset --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/vgg19-cifar10/20220706/00002-ompgd-eps-0.2-acc-0.98/attack-cifar10-dataset/latent-attack-samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/CIFAR10/defense/representation-mixup-training/vgg19-cifar10-rmt-linear-om-pgd-20220901.log 2>&1

# # VGG19 linear rmt beta(1,1) against om-pgd (eps=0.3)
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220901.py run --mode defense --defense_mode rmt --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode om-pgd --attack_eps 0.3 --blackbox --exp_name vgg19-cifar10 --cla_model vgg19 --cla_network_pkl XXX-cla3-XXX --gen_model stylegan2ada --gen_network_pkl /home/data/maggie/result-newhome/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_dataset /home/data/maggie/result-newhome/project/stylegan2ada-cifar10/20220716/00000/project-cifar10-trainset --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/vgg19-cifar10/20220706/00003-ompgd-eps-0.3-acc-0.61/attack-cifar10-dataset/latent-attack-samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/CIFAR10/defense/representation-mixup-training/vgg19-cifar10-rmt-linear-om-pgd-20220901.log 2>&1