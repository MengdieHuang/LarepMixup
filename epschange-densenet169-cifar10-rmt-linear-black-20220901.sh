source ~/.bashrc
source /home/xieyi/anaconda3/bin/activate mmat

# # adv50000 cat cle50000
# # DenseNet169 linear rmt beta(1,1) against PGD (eps=0.02)
# CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220901.py run --mode defense --defense_mode rmt --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode pgd --attack_eps 0.02 --blackbox --exp_name densenet169-cifar10 --cla_model densenet169 --cla_network_pkl /home/data/maggie/result-newhome/train/cla-train/densenet169-cifar10/20210909/00003-testacc-0.7929/train-cifar10-dataset/standard-trained-classifier-densenet169-on-clean-cifar10-epoch-0012.pkl --gen_model stylegan2ada --gen_network_pkl /home/data/maggie/result-newhome/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_dataset /home/data/maggie/result-newhome/project/stylegan2ada-cifar10/20220716/00000/project-cifar10-trainset --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/densenet169-cifar10/20220624/00000-pgd-eps-0.02-acc-56.48/attack-cifar10-dataset/samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/CIFAR10/defense/representation-mixup-training/densenet169-cifar10-rmt-linear-pgd-20220905.log 2>&1

# # DenseNet169 linear rmt beta(1,1) against PGD (eps=0.02)
# CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220901.py run --mode defense --defense_mode rmt --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode pgd --attack_eps 0.02 --blackbox --exp_name densenet169-cifar10 --cla_model densenet169 --cla_network_pkl /home/data/maggie/result-newhome/train/cla-train/densenet169-cifar10/20210909/00003-testacc-0.7929/train-cifar10-dataset/standard-trained-classifier-densenet169-on-clean-cifar10-epoch-0012.pkl --gen_model stylegan2ada --gen_network_pkl /home/data/maggie/result-newhome/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_dataset /home/data/maggie/result-newhome/project/stylegan2ada-cifar10/20220716/00000/project-cifar10-trainset --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/densenet169-cifar10/20220624/00000-pgd-eps-0.02-acc-56.48/attack-cifar10-dataset/samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/CIFAR10/defense/representation-mixup-training/densenet169-cifar10-rmt-linear-pgd-20220905.log 2>&1

# # DenseNet169 linear rmt beta(1,1) against PGD (eps=0.02)
# CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220901.py run --mode defense --defense_mode rmt --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode pgd --attack_eps 0.02 --blackbox --exp_name densenet169-cifar10 --cla_model densenet169 --cla_network_pkl /home/data/maggie/result-newhome/train/cla-train/densenet169-cifar10/20210909/00003-testacc-0.7929/train-cifar10-dataset/standard-trained-classifier-densenet169-on-clean-cifar10-epoch-0012.pkl --gen_model stylegan2ada --gen_network_pkl /home/data/maggie/result-newhome/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_dataset /home/data/maggie/result-newhome/project/stylegan2ada-cifar10/20220716/00000/project-cifar10-trainset --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/densenet169-cifar10/20220624/00000-pgd-eps-0.02-acc-56.48/attack-cifar10-dataset/samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/CIFAR10/defense/representation-mixup-training/densenet169-cifar10-rmt-linear-pgd-20220905.log 2>&1

#-------------以上要存储at训练好的模型-----------
# #   混合训练得到cla1防御PGD 
# DenseNet169 linear rmt beta(1,1) against pgd (eps=0.05)
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220901.py run --mode defense --defense_mode rmt --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode pgd --attack_eps 0.05 --blackbox --exp_name densenet169-cifar10 --cla_model densenet169 --cla_network_pkl /home/data/maggie/result-newhome/defense/rmt/pgd/basemixup-betasampler/densenet169-cifar10/blackbox/20220906/00000/rmt-cifar10-dataset/rmt-trained-classifier-densenet169-on-cifar10-epoch-0031-rmt-linear-cla1.pkl --gen_model stylegan2ada --gen_network_pkl /home/data/maggie/result-newhome/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_dataset /home/data/maggie/result-newhome/project/stylegan2ada-cifar10/20220716/00000/project-cifar10-trainset --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/densenet169-cifar10/20220625/00000-pgd-eps-0.05-acc-32.23/attack-cifar10-dataset/samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/CIFAR10/defense/representation-mixup-training/densenet169-cifar10-rmt-linear-pgd-20220905.log 2>&1

# DenseNet169 linear rmt beta(1,1) against pgd (eps=0.1)
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220901.py run --mode defense --defense_mode rmt --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode autoattack --attack_eps 0.1 --blackbox --exp_name densenet169-cifar10 --cla_model densenet169 --cla_network_pkl /home/data/maggie/result-newhome/defense/rmt/pgd/basemixup-betasampler/densenet169-cifar10/blackbox/20220906/00000/rmt-cifar10-dataset/rmt-trained-classifier-densenet169-on-cifar10-epoch-0031-rmt-linear-cla1.pkl --gen_model stylegan2ada --gen_network_pkl /home/data/maggie/result-newhome/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_dataset /home/data/maggie/result-newhome/project/stylegan2ada-cifar10/20220716/00000/project-cifar10-trainset --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/densenet169-cifar10/20220625/00001-pgd-eps-0.1-acc-6.29/attack-cifar10-dataset/samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/CIFAR10/defense/representation-mixup-training/densenet169-cifar10-rmt-linear-pgd-20220905.log 2>&1

# DenseNet169 linear rmt beta(1,1) against pgd (eps=0.2)
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220901.py run --mode defense --defense_mode rmt --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode deepfool --attack_eps 0.2 --blackbox --exp_name densenet169-cifar10 --cla_model densenet169 --cla_network_pkl /home/data/maggie/result-newhome/defense/rmt/pgd/basemixup-betasampler/densenet169-cifar10/blackbox/20220906/00000/rmt-cifar10-dataset/rmt-trained-classifier-densenet169-on-cifar10-epoch-0031-rmt-linear-cla1.pkl --gen_model stylegan2ada --gen_network_pkl /home/data/maggie/result-newhome/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_dataset /home/data/maggie/result-newhome/project/stylegan2ada-cifar10/20220716/00000/project-cifar10-trainset --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/densenet169-cifar10/20220625/00002-pgd-eps-0.2-acc-0.78/attack-cifar10-dataset/samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/CIFAR10/defense/representation-mixup-training/densenet169-cifar10-rmt-linear-pgd-20220905.log 2>&1

# DenseNet169 linear rmt beta(1,1) against pgd (eps=0.3)
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220901.py run --mode defense --defense_mode rmt --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode cw --attack_eps 0.3 --blackbox --exp_name densenet169-cifar10 --cla_model densenet169 --cla_network_pkl /home/data/maggie/result-newhome/defense/rmt/pgd/basemixup-betasampler/densenet169-cifar10/blackbox/20220906/00000/rmt-cifar10-dataset/rmt-trained-classifier-densenet169-on-cifar10-epoch-0031-rmt-linear-cla1.pkl --gen_model stylegan2ada --gen_network_pkl /home/data/maggie/result-newhome/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_dataset /home/data/maggie/result-newhome/project/stylegan2ada-cifar10/20220716/00000/project-cifar10-trainset --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/densenet169-cifar10/20220626/00000-pgd-eps-0.3-acc-0.21/attack-cifar10-dataset/samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/CIFAR10/defense/representation-mixup-training/densenet169-cifar10-rmt-linear-pgd-20220905.log 2>&1


# #   混合训练得到cla1防御OM-PGD 
# DenseNet169 linear rmt beta(1,1) against om-pgd (eps=0.02)
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220901.py run --mode defense --defense_mode rmt --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode om-pgd --attack_eps 0.02 --blackbox --exp_name densenet169-cifar10 --cla_model densenet169 --cla_network_pkl /home/data/maggie/result-newhome/defense/rmt/pgd/basemixup-betasampler/densenet169-cifar10/blackbox/20220906/00000/rmt-cifar10-dataset/rmt-trained-classifier-densenet169-on-cifar10-epoch-0031-rmt-linear-cla1.pkl --gen_model stylegan2ada --gen_network_pkl /home/data/maggie/result-newhome/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_dataset /home/data/maggie/result-newhome/project/stylegan2ada-cifar10/20220716/00000/project-cifar10-trainset --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/densenet169-cifar10/20220706/00000-ompgd-eps-0.02-acc-48.92/attack-cifar10-dataset/latent-attack-samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/CIFAR10/defense/representation-mixup-training/densenet169-cifar10-rmt-linear-om-pgd-20220905.log 2>&1

# DenseNet169 linear rmt beta(1,1) against om-pgd (eps=0.05)
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220901.py run --mode defense --defense_mode rmt --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode om-pgd --attack_eps 0.05 --blackbox --exp_name densenet169-cifar10 --cla_model densenet169 --cla_network_pkl /home/data/maggie/result-newhome/defense/rmt/pgd/basemixup-betasampler/densenet169-cifar10/blackbox/20220906/00000/rmt-cifar10-dataset/rmt-trained-classifier-densenet169-on-cifar10-epoch-0031-rmt-linear-cla1.pkl --gen_model stylegan2ada --gen_network_pkl /home/data/maggie/result-newhome/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_dataset /home/data/maggie/result-newhome/project/stylegan2ada-cifar10/20220716/00000/project-cifar10-trainset --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/densenet169-cifar10/20220706/00001-ompgd-eps-0.05-acc-14.25/attack-cifar10-dataset/latent-attack-samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/CIFAR10/defense/representation-mixup-training/densenet169-cifar10-rmt-linear-om-pgd-20220905.log 2>&1

# DenseNet169 linear rmt beta(1,1) against om-pgd (eps=0.1)
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220901.py run --mode defense --defense_mode rmt --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode om-pgd --attack_eps 0.1 --blackbox --exp_name densenet169-cifar10 --cla_model densenet169 --cla_network_pkl /home/data/maggie/result-newhome/defense/rmt/pgd/basemixup-betasampler/densenet169-cifar10/blackbox/20220906/00000/rmt-cifar10-dataset/rmt-trained-classifier-densenet169-on-cifar10-epoch-0031-rmt-linear-cla1.pkl --gen_model stylegan2ada --gen_network_pkl /home/data/maggie/result-newhome/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_dataset /home/data/maggie/result-newhome/project/stylegan2ada-cifar10/20220716/00000/project-cifar10-trainset --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/densenet169-cifar10/20220706/00002-ompgd-eps-0.1-acc-1.33/attack-cifar10-dataset/latent-attack-samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/CIFAR10/defense/representation-mixup-training/densenet169-cifar10-rmt-linear-om-pgd-20220905.log 2>&1

# DenseNet169 linear rmt beta(1,1) against om-pgd (eps=0.2)
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220901.py run --mode defense --defense_mode rmt --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode om-pgd --attack_eps 0.2 --blackbox --exp_name densenet169-cifar10 --cla_model densenet169 --cla_network_pkl /home/data/maggie/result-newhome/defense/rmt/pgd/basemixup-betasampler/densenet169-cifar10/blackbox/20220906/00000/rmt-cifar10-dataset/rmt-trained-classifier-densenet169-on-cifar10-epoch-0031-rmt-linear-cla1.pkl --gen_model stylegan2ada --gen_network_pkl /home/data/maggie/result-newhome/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_dataset /home/data/maggie/result-newhome/project/stylegan2ada-cifar10/20220716/00000/project-cifar10-trainset --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/densenet169-cifar10/20220706/00003-ompgd-eps-0.2-acc-0.18/attack-cifar10-dataset/latent-attack-samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/CIFAR10/defense/representation-mixup-training/densenet169-cifar10-rmt-linear-om-pgd-20220905.log 2>&1

# DenseNet169 linear rmt beta(1,1) against om-pgd (eps=0.3)
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220901.py run --mode defense --defense_mode rmt --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode om-pgd --attack_eps 0.3 --blackbox --exp_name densenet169-cifar10 --cla_model densenet169 --cla_network_pkl /home/data/maggie/result-newhome/defense/rmt/pgd/basemixup-betasampler/densenet169-cifar10/blackbox/20220906/00000/rmt-cifar10-dataset/rmt-trained-classifier-densenet169-on-cifar10-epoch-0031-rmt-linear-cla1.pkl --gen_model stylegan2ada --gen_network_pkl /home/data/maggie/result-newhome/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_dataset /home/data/maggie/result-newhome/project/stylegan2ada-cifar10/20220716/00000/project-cifar10-trainset --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/densenet169-cifar10/20220706/00004-ompgd-eps-0.3-acc-0.13/attack-cifar10-dataset/latent-attack-samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/CIFAR10/defense/representation-mixup-training/densenet169-cifar10-rmt-linear-om-pgd-20220905.log 2>&1

-----------------
# #   混合训练得到cla2防御PGD 
# DenseNet169 linear rmt beta(1,1) against pgd (eps=0.05)
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220901.py run --mode defense --defense_mode rmt --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode pgd --attack_eps 0.05 --blackbox --exp_name densenet169-cifar10 --cla_model densenet169 --cla_network_pkl /home/data/maggie/result-newhome/defense/rmt/pgd/basemixup-betasampler/densenet169-cifar10/blackbox/20220907/00000/rmt-cifar10-dataset/rmt-trained-classifier-densenet169-on-cifar10-epoch-0016-rmt-linear-cla2.pkl --gen_model stylegan2ada --gen_network_pkl /home/data/maggie/result-newhome/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_dataset /home/data/maggie/result-newhome/project/stylegan2ada-cifar10/20220716/00000/project-cifar10-trainset --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/densenet169-cifar10/20220625/00000-pgd-eps-0.05-acc-32.23/attack-cifar10-dataset/samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/CIFAR10/defense/representation-mixup-training/densenet169-cifar10-rmt-linear-pgd-20220905.log 2>&1

# DenseNet169 linear rmt beta(1,1) against pgd (eps=0.1)
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220901.py run --mode defense --defense_mode rmt --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode autoattack --attack_eps 0.1 --blackbox --exp_name densenet169-cifar10 --cla_model densenet169 --cla_network_pkl /home/data/maggie/result-newhome/defense/rmt/pgd/basemixup-betasampler/densenet169-cifar10/blackbox/20220907/00000/rmt-cifar10-dataset/rmt-trained-classifier-densenet169-on-cifar10-epoch-0016-rmt-linear-cla2.pkl --gen_model stylegan2ada --gen_network_pkl /home/data/maggie/result-newhome/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_dataset /home/data/maggie/result-newhome/project/stylegan2ada-cifar10/20220716/00000/project-cifar10-trainset --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/densenet169-cifar10/20220625/00001-pgd-eps-0.1-acc-6.29/attack-cifar10-dataset/samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/CIFAR10/defense/representation-mixup-training/densenet169-cifar10-rmt-linear-pgd-20220905.log 2>&1

# DenseNet169 linear rmt beta(1,1) against pgd (eps=0.2)
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220901.py run --mode defense --defense_mode rmt --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode deepfool --attack_eps 0.2 --blackbox --exp_name densenet169-cifar10 --cla_model densenet169 --cla_network_pkl /home/data/maggie/result-newhome/defense/rmt/pgd/basemixup-betasampler/densenet169-cifar10/blackbox/20220907/00000/rmt-cifar10-dataset/rmt-trained-classifier-densenet169-on-cifar10-epoch-0016-rmt-linear-cla2.pkl --gen_model stylegan2ada --gen_network_pkl /home/data/maggie/result-newhome/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_dataset /home/data/maggie/result-newhome/project/stylegan2ada-cifar10/20220716/00000/project-cifar10-trainset --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/densenet169-cifar10/20220625/00002-pgd-eps-0.2-acc-0.78/attack-cifar10-dataset/samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/CIFAR10/defense/representation-mixup-training/densenet169-cifar10-rmt-linear-pgd-20220905.log 2>&1

# DenseNet169 linear rmt beta(1,1) against pgd (eps=0.3)
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220901.py run --mode defense --defense_mode rmt --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode cw --attack_eps 0.3 --blackbox --exp_name densenet169-cifar10 --cla_model densenet169 --cla_network_pkl /home/data/maggie/result-newhome/defense/rmt/pgd/basemixup-betasampler/densenet169-cifar10/blackbox/20220907/00000/rmt-cifar10-dataset/rmt-trained-classifier-densenet169-on-cifar10-epoch-0016-rmt-linear-cla2.pkl --gen_model stylegan2ada --gen_network_pkl /home/data/maggie/result-newhome/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_dataset /home/data/maggie/result-newhome/project/stylegan2ada-cifar10/20220716/00000/project-cifar10-trainset --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/densenet169-cifar10/20220626/00000-pgd-eps-0.3-acc-0.21/attack-cifar10-dataset/samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/CIFAR10/defense/representation-mixup-training/densenet169-cifar10-rmt-linear-pgd-20220905.log 2>&1

# #   混合训练得到cla2防御OM-PGD 
# DenseNet169 linear rmt beta(1,1) against om-pgd (eps=0.02)
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220901.py run --mode defense --defense_mode rmt --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode om-pgd --attack_eps 0.02 --blackbox --exp_name densenet169-cifar10 --cla_model densenet169 --cla_network_pkl /home/data/maggie/result-newhome/defense/rmt/pgd/basemixup-betasampler/densenet169-cifar10/blackbox/20220907/00000/rmt-cifar10-dataset/rmt-trained-classifier-densenet169-on-cifar10-epoch-0016-rmt-linear-cla2.pkl --gen_model stylegan2ada --gen_network_pkl /home/data/maggie/result-newhome/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_dataset /home/data/maggie/result-newhome/project/stylegan2ada-cifar10/20220716/00000/project-cifar10-trainset --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/densenet169-cifar10/20220706/00000-ompgd-eps-0.02-acc-48.92/attack-cifar10-dataset/latent-attack-samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/CIFAR10/defense/representation-mixup-training/densenet169-cifar10-rmt-linear-om-pgd-20220905.log 2>&1

# DenseNet169 linear rmt beta(1,1) against om-pgd (eps=0.05)
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220901.py run --mode defense --defense_mode rmt --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode om-pgd --attack_eps 0.05 --blackbox --exp_name densenet169-cifar10 --cla_model densenet169 --cla_network_pkl /home/data/maggie/result-newhome/defense/rmt/pgd/basemixup-betasampler/densenet169-cifar10/blackbox/20220907/00000/rmt-cifar10-dataset/rmt-trained-classifier-densenet169-on-cifar10-epoch-0016-rmt-linear-cla2.pkl --gen_model stylegan2ada --gen_network_pkl /home/data/maggie/result-newhome/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_dataset /home/data/maggie/result-newhome/project/stylegan2ada-cifar10/20220716/00000/project-cifar10-trainset --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/densenet169-cifar10/20220706/00001-ompgd-eps-0.05-acc-14.25/attack-cifar10-dataset/latent-attack-samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/CIFAR10/defense/representation-mixup-training/densenet169-cifar10-rmt-linear-om-pgd-20220905.log 2>&1

# DenseNet169 linear rmt beta(1,1) against om-pgd (eps=0.1)
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220901.py run --mode defense --defense_mode rmt --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode om-pgd --attack_eps 0.1 --blackbox --exp_name densenet169-cifar10 --cla_model densenet169 --cla_network_pkl /home/data/maggie/result-newhome/defense/rmt/pgd/basemixup-betasampler/densenet169-cifar10/blackbox/20220907/00000/rmt-cifar10-dataset/rmt-trained-classifier-densenet169-on-cifar10-epoch-0016-rmt-linear-cla2.pkl --gen_model stylegan2ada --gen_network_pkl /home/data/maggie/result-newhome/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_dataset /home/data/maggie/result-newhome/project/stylegan2ada-cifar10/20220716/00000/project-cifar10-trainset --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/densenet169-cifar10/20220706/00002-ompgd-eps-0.1-acc-1.33/attack-cifar10-dataset/latent-attack-samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/CIFAR10/defense/representation-mixup-training/densenet169-cifar10-rmt-linear-om-pgd-20220905.log 2>&1

# DenseNet169 linear rmt beta(1,1) against om-pgd (eps=0.2)
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220901.py run --mode defense --defense_mode rmt --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode om-pgd --attack_eps 0.2 --blackbox --exp_name densenet169-cifar10 --cla_model densenet169 --cla_network_pkl /home/data/maggie/result-newhome/defense/rmt/pgd/basemixup-betasampler/densenet169-cifar10/blackbox/20220907/00000/rmt-cifar10-dataset/rmt-trained-classifier-densenet169-on-cifar10-epoch-0016-rmt-linear-cla2.pkl --gen_model stylegan2ada --gen_network_pkl /home/data/maggie/result-newhome/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_dataset /home/data/maggie/result-newhome/project/stylegan2ada-cifar10/20220716/00000/project-cifar10-trainset --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/densenet169-cifar10/20220706/00003-ompgd-eps-0.2-acc-0.18/attack-cifar10-dataset/latent-attack-samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/CIFAR10/defense/representation-mixup-training/densenet169-cifar10-rmt-linear-om-pgd-20220905.log 2>&1

# DenseNet169 linear rmt beta(1,1) against om-pgd (eps=0.3)
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220901.py run --mode defense --defense_mode rmt --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode om-pgd --attack_eps 0.3 --blackbox --exp_name densenet169-cifar10 --cla_model densenet169 --cla_network_pkl /home/data/maggie/result-newhome/defense/rmt/pgd/basemixup-betasampler/densenet169-cifar10/blackbox/20220907/00000/rmt-cifar10-dataset/rmt-trained-classifier-densenet169-on-cifar10-epoch-0016-rmt-linear-cla2.pkl --gen_model stylegan2ada --gen_network_pkl /home/data/maggie/result-newhome/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_dataset /home/data/maggie/result-newhome/project/stylegan2ada-cifar10/20220716/00000/project-cifar10-trainset --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/densenet169-cifar10/20220706/00004-ompgd-eps-0.3-acc-0.13/attack-cifar10-dataset/latent-attack-samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/CIFAR10/defense/representation-mixup-training/densenet169-cifar10-rmt-linear-om-pgd-20220905.log 2>&1


-----------------
# #   混合训练得到cla3防御PGD 
# DenseNet169 linear rmt beta(1,1) against pgd (eps=0.05)
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220901.py run --mode defense --defense_mode rmt --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode pgd --attack_eps 0.05 --blackbox --exp_name densenet169-cifar10 --cla_model densenet169 --cla_network_pkl /home/data/maggie/result-newhome/defense/rmt/pgd/basemixup-betasampler/densenet169-cifar10/blackbox/20220907/00001/rmt-cifar10-dataset/rmt-trained-classifier-densenet169-on-cifar10-epoch-0031-rmt-linear-cla3.pkl --gen_model stylegan2ada --gen_network_pkl /home/data/maggie/result-newhome/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_dataset /home/data/maggie/result-newhome/project/stylegan2ada-cifar10/20220716/00000/project-cifar10-trainset --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/densenet169-cifar10/20220625/00000-pgd-eps-0.05-acc-32.23/attack-cifar10-dataset/samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/CIFAR10/defense/representation-mixup-training/densenet169-cifar10-rmt-linear-pgd-20220905.log 2>&1

# DenseNet169 linear rmt beta(1,1) against pgd (eps=0.1)
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220901.py run --mode defense --defense_mode rmt --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode autoattack --attack_eps 0.1 --blackbox --exp_name densenet169-cifar10 --cla_model densenet169 --cla_network_pkl /home/data/maggie/result-newhome/defense/rmt/pgd/basemixup-betasampler/densenet169-cifar10/blackbox/20220907/00001/rmt-cifar10-dataset/rmt-trained-classifier-densenet169-on-cifar10-epoch-0031-rmt-linear-cla3.pkl --gen_model stylegan2ada --gen_network_pkl /home/data/maggie/result-newhome/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_dataset /home/data/maggie/result-newhome/project/stylegan2ada-cifar10/20220716/00000/project-cifar10-trainset --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/densenet169-cifar10/20220625/00001-pgd-eps-0.1-acc-6.29/attack-cifar10-dataset/samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/CIFAR10/defense/representation-mixup-training/densenet169-cifar10-rmt-linear-pgd-20220905.log 2>&1

# DenseNet169 linear rmt beta(1,1) against pgd (eps=0.2)
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220901.py run --mode defense --defense_mode rmt --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode deepfool --attack_eps 0.2 --blackbox --exp_name densenet169-cifar10 --cla_model densenet169 --cla_network_pkl /home/data/maggie/result-newhome/defense/rmt/pgd/basemixup-betasampler/densenet169-cifar10/blackbox/20220907/00001/rmt-cifar10-dataset/rmt-trained-classifier-densenet169-on-cifar10-epoch-0031-rmt-linear-cla3.pkl --gen_model stylegan2ada --gen_network_pkl /home/data/maggie/result-newhome/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_dataset /home/data/maggie/result-newhome/project/stylegan2ada-cifar10/20220716/00000/project-cifar10-trainset --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/densenet169-cifar10/20220625/00002-pgd-eps-0.2-acc-0.78/attack-cifar10-dataset/samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/CIFAR10/defense/representation-mixup-training/densenet169-cifar10-rmt-linear-pgd-20220905.log 2>&1

# DenseNet169 linear rmt beta(1,1) against pgd (eps=0.3)
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220901.py run --mode defense --defense_mode rmt --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode cw --attack_eps 0.3 --blackbox --exp_name densenet169-cifar10 --cla_model densenet169 --cla_network_pkl /home/data/maggie/result-newhome/defense/rmt/pgd/basemixup-betasampler/densenet169-cifar10/blackbox/20220907/00001/rmt-cifar10-dataset/rmt-trained-classifier-densenet169-on-cifar10-epoch-0031-rmt-linear-cla3.pkl --gen_model stylegan2ada --gen_network_pkl /home/data/maggie/result-newhome/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_dataset /home/data/maggie/result-newhome/project/stylegan2ada-cifar10/20220716/00000/project-cifar10-trainset --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/densenet169-cifar10/20220626/00000-pgd-eps-0.3-acc-0.21/attack-cifar10-dataset/samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/CIFAR10/defense/representation-mixup-training/densenet169-cifar10-rmt-linear-pgd-20220905.log 2>&1

# #   混合训练得到cla3防御OM-PGD 
# DenseNet169 linear rmt beta(1,1) against om-pgd (eps=0.02)
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220901.py run --mode defense --defense_mode rmt --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode om-pgd --attack_eps 0.02 --blackbox --exp_name densenet169-cifar10 --cla_model densenet169 --cla_network_pkl /home/data/maggie/result-newhome/defense/rmt/pgd/basemixup-betasampler/densenet169-cifar10/blackbox/20220907/00001/rmt-cifar10-dataset/rmt-trained-classifier-densenet169-on-cifar10-epoch-0031-rmt-linear-cla3.pkl --gen_model stylegan2ada --gen_network_pkl /home/data/maggie/result-newhome/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_dataset /home/data/maggie/result-newhome/project/stylegan2ada-cifar10/20220716/00000/project-cifar10-trainset --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/densenet169-cifar10/20220706/00000-ompgd-eps-0.02-acc-48.92/attack-cifar10-dataset/latent-attack-samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/CIFAR10/defense/representation-mixup-training/densenet169-cifar10-rmt-linear-om-pgd-20220905.log 2>&1

# DenseNet169 linear rmt beta(1,1) against om-pgd (eps=0.05)
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220901.py run --mode defense --defense_mode rmt --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode om-pgd --attack_eps 0.05 --blackbox --exp_name densenet169-cifar10 --cla_model densenet169 --cla_network_pkl /home/data/maggie/result-newhome/defense/rmt/pgd/basemixup-betasampler/densenet169-cifar10/blackbox/20220907/00001/rmt-cifar10-dataset/rmt-trained-classifier-densenet169-on-cifar10-epoch-0031-rmt-linear-cla3.pkl --gen_model stylegan2ada --gen_network_pkl /home/data/maggie/result-newhome/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_dataset /home/data/maggie/result-newhome/project/stylegan2ada-cifar10/20220716/00000/project-cifar10-trainset --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/densenet169-cifar10/20220706/00001-ompgd-eps-0.05-acc-14.25/attack-cifar10-dataset/latent-attack-samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/CIFAR10/defense/representation-mixup-training/densenet169-cifar10-rmt-linear-om-pgd-20220905.log 2>&1

# DenseNet169 linear rmt beta(1,1) against om-pgd (eps=0.1)
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220901.py run --mode defense --defense_mode rmt --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode om-pgd --attack_eps 0.1 --blackbox --exp_name densenet169-cifar10 --cla_model densenet169 --cla_network_pkl /home/data/maggie/result-newhome/defense/rmt/pgd/basemixup-betasampler/densenet169-cifar10/blackbox/20220907/00001/rmt-cifar10-dataset/rmt-trained-classifier-densenet169-on-cifar10-epoch-0031-rmt-linear-cla3.pkl --gen_model stylegan2ada --gen_network_pkl /home/data/maggie/result-newhome/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_dataset /home/data/maggie/result-newhome/project/stylegan2ada-cifar10/20220716/00000/project-cifar10-trainset --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/densenet169-cifar10/20220706/00002-ompgd-eps-0.1-acc-1.33/attack-cifar10-dataset/latent-attack-samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/CIFAR10/defense/representation-mixup-training/densenet169-cifar10-rmt-linear-om-pgd-20220905.log 2>&1

# DenseNet169 linear rmt beta(1,1) against om-pgd (eps=0.2)
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220901.py run --mode defense --defense_mode rmt --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode om-pgd --attack_eps 0.2 --blackbox --exp_name densenet169-cifar10 --cla_model densenet169 --cla_network_pkl /home/data/maggie/result-newhome/defense/rmt/pgd/basemixup-betasampler/densenet169-cifar10/blackbox/20220907/00001/rmt-cifar10-dataset/rmt-trained-classifier-densenet169-on-cifar10-epoch-0031-rmt-linear-cla3.pkl --gen_model stylegan2ada --gen_network_pkl /home/data/maggie/result-newhome/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_dataset /home/data/maggie/result-newhome/project/stylegan2ada-cifar10/20220716/00000/project-cifar10-trainset --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/densenet169-cifar10/20220706/00003-ompgd-eps-0.2-acc-0.18/attack-cifar10-dataset/latent-attack-samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/CIFAR10/defense/representation-mixup-training/densenet169-cifar10-rmt-linear-om-pgd-20220905.log 2>&1

# DenseNet169 linear rmt beta(1,1) against om-pgd (eps=0.3)
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220901.py run --mode defense --defense_mode rmt --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode om-pgd --attack_eps 0.3 --blackbox --exp_name densenet169-cifar10 --cla_model densenet169 --cla_network_pkl /home/data/maggie/result-newhome/defense/rmt/pgd/basemixup-betasampler/densenet169-cifar10/blackbox/20220907/00001/rmt-cifar10-dataset/rmt-trained-classifier-densenet169-on-cifar10-epoch-0031-rmt-linear-cla3.pkl --gen_model stylegan2ada --gen_network_pkl /home/data/maggie/result-newhome/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_dataset /home/data/maggie/result-newhome/project/stylegan2ada-cifar10/20220716/00000/project-cifar10-trainset --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/densenet169-cifar10/20220706/00004-ompgd-eps-0.3-acc-0.13/attack-cifar10-dataset/latent-attack-samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/CIFAR10/defense/representation-mixup-training/densenet169-cifar10-rmt-linear-om-pgd-20220905.log 2>&1
