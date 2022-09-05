source ~/.bashrc
source /home/xieyi/anaconda3/bin/activate mmat

# # adv50000 cat cle50000
# # ResNet18 linear rmt beta(1,1) against PGD (eps=0.02)
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220901.py run --mode defense --defense_mode rmt --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode pgd --attack_eps 0.02 --blackbox --exp_name resnet18-cifar10 --cla_model resnet18 --cla_network_pkl /home/data/maggie/result-newhome/train/cla-train/resnet18-cifar10/20210908/00017-testacc-0.7779/train-cifar10-dataset/standard-trained-classifier-resnet18-on-clean-cifar10-finished.pkl --gen_model stylegan2ada --gen_network_pkl /home/data/maggie/result-newhome/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_dataset /home/data/maggie/result-newhome/project/stylegan2ada-cifar10/20220716/00000/project-cifar10-trainset --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/resnet18-cifar10/20220624/00000-pgd-eps-0.02-acc-57.81/attack-cifar10-dataset/samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/CIFAR10/defense/representation-mixup-training/resnet18-cifar10-rmt-linear-pgd-20220901.log 2>&1

# # ResNet18 linear rmt beta(1,1) against PGD (eps=0.02)
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220901.py run --mode defense --defense_mode rmt --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode pgd --attack_eps 0.02 --blackbox --exp_name resnet18-cifar10 --cla_model resnet18 --cla_network_pkl /home/data/maggie/result-newhome/train/cla-train/resnet18-cifar10/20210908/00017-testacc-0.7779/train-cifar10-dataset/standard-trained-classifier-resnet18-on-clean-cifar10-finished.pkl --gen_model stylegan2ada --gen_network_pkl /home/data/maggie/result-newhome/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_dataset /home/data/maggie/result-newhome/project/stylegan2ada-cifar10/20220716/00000/project-cifar10-trainset --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/resnet18-cifar10/20220624/00000-pgd-eps-0.02-acc-57.81/attack-cifar10-dataset/samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/CIFAR10/defense/representation-mixup-training/resnet18-cifar10-rmt-linear-pgd-20220901.log 2>&1

# # ResNet18 linear rmt beta(1,1) against PGD (eps=0.02)
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220901.py run --mode defense --defense_mode rmt --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode pgd --attack_eps 0.02 --blackbox --exp_name resnet18-cifar10 --cla_model resnet18 --cla_network_pkl /home/data/maggie/result-newhome/train/cla-train/resnet18-cifar10/20210908/00017-testacc-0.7779/train-cifar10-dataset/standard-trained-classifier-resnet18-on-clean-cifar10-finished.pkl --gen_model stylegan2ada --gen_network_pkl /home/data/maggie/result-newhome/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_dataset /home/data/maggie/result-newhome/project/stylegan2ada-cifar10/20220716/00000/project-cifar10-trainset --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/resnet18-cifar10/20220624/00000-pgd-eps-0.02-acc-57.81/attack-cifar10-dataset/samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/CIFAR10/defense/representation-mixup-training/resnet18-cifar10-rmt-linear-pgd-20220901.log 2>&1

#-------------以上要存储at训练好的模型-----------
# #   混合训练得到cla1防御PGD 
# ResNet18 linear rmt beta(1,1) against pgd (eps=0.05)
CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220901.py run --mode defense --defense_mode rmt --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode pgd --attack_eps 0.05 --blackbox --exp_name resnet18-cifar10 --cla_model resnet18 --cla_network_pkl /home/data/maggie/result-newhome/defense/rmt/pgd/basemixup-betasampler/resnet18-cifar10/blackbox/20220903/00000/rmt-cifar10-dataset/rmt-trained-classifier-resnet18-on-cifar10-epoch-0018-rmt-linear-cla1.pkl --gen_model stylegan2ada --gen_network_pkl /home/data/maggie/result-newhome/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_dataset /home/data/maggie/result-newhome/project/stylegan2ada-cifar10/20220716/00000/project-cifar10-trainset --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/resnet18-cifar10/20220624/00001-pgd-eps-0.05-acc-34.87/attack-cifar10-dataset/samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/CIFAR10/defense/representation-mixup-training/resnet18-cifar10-rmt-linear-pgd-20220901.log 2>&1

# ResNet18 linear rmt beta(1,1) against pgd (eps=0.1)
CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220901.py run --mode defense --defense_mode rmt --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode autoattack --attack_eps 0.1 --blackbox --exp_name resnet18-cifar10 --cla_model resnet18 --cla_network_pkl /home/data/maggie/result-newhome/defense/rmt/pgd/basemixup-betasampler/resnet18-cifar10/blackbox/20220903/00000/rmt-cifar10-dataset/rmt-trained-classifier-resnet18-on-cifar10-epoch-0018-rmt-linear-cla1.pkl --gen_model stylegan2ada --gen_network_pkl /home/data/maggie/result-newhome/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_dataset /home/data/maggie/result-newhome/project/stylegan2ada-cifar10/20220716/00000/project-cifar10-trainset --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/resnet18-cifar10/20220624/00002-pgd-eps-0.1-acc-7.83/attack-cifar10-dataset/samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/CIFAR10/defense/representation-mixup-training/resnet18-cifar10-rmt-linear-pgd-20220901.log 2>&1

# ResNet18 linear rmt beta(1,1) against pgd (eps=0.2)
CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220901.py run --mode defense --defense_mode rmt --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode deepfool --attack_eps 0.2 --blackbox --exp_name resnet18-cifar10 --cla_model resnet18 --cla_network_pkl /home/data/maggie/result-newhome/defense/rmt/pgd/basemixup-betasampler/resnet18-cifar10/blackbox/20220903/00000/rmt-cifar10-dataset/rmt-trained-classifier-resnet18-on-cifar10-epoch-0018-rmt-linear-cla1.pkl --gen_model stylegan2ada --gen_network_pkl /home/data/maggie/result-newhome/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_dataset /home/data/maggie/result-newhome/project/stylegan2ada-cifar10/20220716/00000/project-cifar10-trainset --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/resnet18-cifar10/20220624/00003-pgd-eps-0.2-acc-1.04/attack-cifar10-dataset/samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/CIFAR10/defense/representation-mixup-training/resnet18-cifar10-rmt-linear-pgd-20220901.log 2>&1

# ResNet18 linear rmt beta(1,1) against pgd (eps=0.3)
CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220901.py run --mode defense --defense_mode rmt --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode cw --attack_eps 0.3 --blackbox --exp_name resnet18-cifar10 --cla_model resnet18 --cla_network_pkl /home/data/maggie/result-newhome/defense/rmt/pgd/basemixup-betasampler/resnet18-cifar10/blackbox/20220903/00000/rmt-cifar10-dataset/rmt-trained-classifier-resnet18-on-cifar10-epoch-0018-rmt-linear-cla1.pkl --gen_model stylegan2ada --gen_network_pkl /home/data/maggie/result-newhome/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_dataset /home/data/maggie/result-newhome/project/stylegan2ada-cifar10/20220716/00000/project-cifar10-trainset --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/resnet18-cifar10/20220624/00004-pgd-eps-0.3-acc-0.23/attack-cifar10-dataset/samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/CIFAR10/defense/representation-mixup-training/resnet18-cifar10-rmt-linear-pgd-20220901.log 2>&1


# #   混合训练得到cla1防御OM-PGD 
# ResNet18 linear rmt beta(1,1) against om-pgd (eps=0.02)
CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220901.py run --mode defense --defense_mode rmt --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode om-pgd --attack_eps 0.02 --blackbox --exp_name resnet18-cifar10 --cla_model resnet18 --cla_network_pkl /home/data/maggie/result-newhome/defense/rmt/pgd/basemixup-betasampler/resnet18-cifar10/blackbox/20220903/00000/rmt-cifar10-dataset/rmt-trained-classifier-resnet18-on-cifar10-epoch-0018-rmt-linear-cla1.pkl --gen_model stylegan2ada --gen_network_pkl /home/data/maggie/result-newhome/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_dataset /home/data/maggie/result-newhome/project/stylegan2ada-cifar10/20220716/00000/project-cifar10-trainset --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/resnet18-cifar10/20220704/00000-ompgd-eps-0.02-acc-48.3/attack-cifar10-dataset/latent-attack-samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/CIFAR10/defense/representation-mixup-training/resnet18-cifar10-rmt-linear-om-pgd-20220901.log 2>&1

# ResNet18 linear rmt beta(1,1) against om-pgd (eps=0.05)
CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220901.py run --mode defense --defense_mode rmt --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode om-pgd --attack_eps 0.05 --blackbox --exp_name resnet18-cifar10 --cla_model resnet18 --cla_network_pkl /home/data/maggie/result-newhome/defense/rmt/pgd/basemixup-betasampler/resnet18-cifar10/blackbox/20220903/00000/rmt-cifar10-dataset/rmt-trained-classifier-resnet18-on-cifar10-epoch-0018-rmt-linear-cla1.pkl --gen_model stylegan2ada --gen_network_pkl /home/data/maggie/result-newhome/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_dataset /home/data/maggie/result-newhome/project/stylegan2ada-cifar10/20220716/00000/project-cifar10-trainset --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/resnet18-cifar10/20220704/00001-ompgd-eps-0.05-acc-15.02/attack-cifar10-dataset/latent-attack-samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/CIFAR10/defense/representation-mixup-training/resnet18-cifar10-rmt-linear-om-pgd-20220901.log 2>&1

# ResNet18 linear rmt beta(1,1) against om-pgd (eps=0.1)
CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220901.py run --mode defense --defense_mode rmt --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode om-pgd --attack_eps 0.1 --blackbox --exp_name resnet18-cifar10 --cla_model resnet18 --cla_network_pkl /home/data/maggie/result-newhome/defense/rmt/pgd/basemixup-betasampler/resnet18-cifar10/blackbox/20220903/00000/rmt-cifar10-dataset/rmt-trained-classifier-resnet18-on-cifar10-epoch-0018-rmt-linear-cla1.pkl --gen_model stylegan2ada --gen_network_pkl /home/data/maggie/result-newhome/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_dataset /home/data/maggie/result-newhome/project/stylegan2ada-cifar10/20220716/00000/project-cifar10-trainset --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/resnet18-cifar10/20220705/00000-ompgd-eps-0.1-acc-1.35/attack-cifar10-dataset/latent-attack-samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/CIFAR10/defense/representation-mixup-training/resnet18-cifar10-rmt-linear-om-pgd-20220901.log 2>&1

# ResNet18 linear rmt beta(1,1) against om-pgd (eps=0.2)
CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220901.py run --mode defense --defense_mode rmt --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode om-pgd --attack_eps 0.2 --blackbox --exp_name resnet18-cifar10 --cla_model resnet18 --cla_network_pkl /home/data/maggie/result-newhome/defense/rmt/pgd/basemixup-betasampler/resnet18-cifar10/blackbox/20220903/00000/rmt-cifar10-dataset/rmt-trained-classifier-resnet18-on-cifar10-epoch-0018-rmt-linear-cla1.pkl --gen_model stylegan2ada --gen_network_pkl /home/data/maggie/result-newhome/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_dataset /home/data/maggie/result-newhome/project/stylegan2ada-cifar10/20220716/00000/project-cifar10-trainset --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/resnet18-cifar10/20220705/00001-ompgd-eps-0.2-acc-0.17/attack-cifar10-dataset/latent-attack-samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/CIFAR10/defense/representation-mixup-training/resnet18-cifar10-rmt-linear-om-pgd-20220901.log 2>&1

# ResNet18 linear rmt beta(1,1) against om-pgd (eps=0.3)
CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220901.py run --mode defense --defense_mode rmt --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode om-pgd --attack_eps 0.3 --blackbox --exp_name resnet18-cifar10 --cla_model resnet18 --cla_network_pkl /home/data/maggie/result-newhome/defense/rmt/pgd/basemixup-betasampler/resnet18-cifar10/blackbox/20220903/00000/rmt-cifar10-dataset/rmt-trained-classifier-resnet18-on-cifar10-epoch-0018-rmt-linear-cla1.pkl --gen_model stylegan2ada --gen_network_pkl /home/data/maggie/result-newhome/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_dataset /home/data/maggie/result-newhome/project/stylegan2ada-cifar10/20220716/00000/project-cifar10-trainset --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/resnet18-cifar10/20220705/00002-ompgd-eps-0.3-acc-0.10/attack-cifar10-dataset/latent-attack-samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/CIFAR10/defense/representation-mixup-training/resnet18-cifar10-rmt-linear-om-pgd-20220901.log 2>&1

---------------
# #   混合训练得到cla2防御PGD 
# ResNet18 linear rmt beta(1,1) against pgd (eps=0.05)
CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220901.py run --mode defense --defense_mode rmt --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode pgd --attack_eps 0.05 --blackbox --exp_name resnet18-cifar10 --cla_model resnet18 --cla_network_pkl /home/data/maggie/result-newhome/defense/rmt/pgd/basemixup-betasampler/resnet18-cifar10/blackbox/20220903/00001/rmt-cifar10-dataset/rmt-trained-classifier-resnet18-on-cifar10-epoch-0021-rmt-linear-cla2.pkl --gen_model stylegan2ada --gen_network_pkl /home/data/maggie/result-newhome/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_dataset /home/data/maggie/result-newhome/project/stylegan2ada-cifar10/20220716/00000/project-cifar10-trainset --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/resnet18-cifar10/20220624/00001-pgd-eps-0.05-acc-34.87/attack-cifar10-dataset/samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/CIFAR10/defense/representation-mixup-training/resnet18-cifar10-rmt-linear-pgd-20220901.log 2>&1

# ResNet18 linear rmt beta(1,1) against pgd (eps=0.1)
CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220901.py run --mode defense --defense_mode rmt --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode autoattack --attack_eps 0.1 --blackbox --exp_name resnet18-cifar10 --cla_model resnet18 --cla_network_pkl /home/data/maggie/result-newhome/defense/rmt/pgd/basemixup-betasampler/resnet18-cifar10/blackbox/20220903/00001/rmt-cifar10-dataset/rmt-trained-classifier-resnet18-on-cifar10-epoch-0021-rmt-linear-cla2.pkl --gen_model stylegan2ada --gen_network_pkl /home/data/maggie/result-newhome/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_dataset /home/data/maggie/result-newhome/project/stylegan2ada-cifar10/20220716/00000/project-cifar10-trainset --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/resnet18-cifar10/20220624/00002-pgd-eps-0.1-acc-7.83/attack-cifar10-dataset/samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/CIFAR10/defense/representation-mixup-training/resnet18-cifar10-rmt-linear-pgd-20220901.log 2>&1

# ResNet18 linear rmt beta(1,1) against pgd (eps=0.2)
CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220901.py run --mode defense --defense_mode rmt --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode deepfool --attack_eps 0.2 --blackbox --exp_name resnet18-cifar10 --cla_model resnet18 --cla_network_pkl /home/data/maggie/result-newhome/defense/rmt/pgd/basemixup-betasampler/resnet18-cifar10/blackbox/20220903/00001/rmt-cifar10-dataset/rmt-trained-classifier-resnet18-on-cifar10-epoch-0021-rmt-linear-cla2.pkl --gen_model stylegan2ada --gen_network_pkl /home/data/maggie/result-newhome/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_dataset /home/data/maggie/result-newhome/project/stylegan2ada-cifar10/20220716/00000/project-cifar10-trainset --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/resnet18-cifar10/20220624/00003-pgd-eps-0.2-acc-1.04/attack-cifar10-dataset/samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/CIFAR10/defense/representation-mixup-training/resnet18-cifar10-rmt-linear-pgd-20220901.log 2>&1

# ResNet18 linear rmt beta(1,1) against pgd (eps=0.3)
CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220901.py run --mode defense --defense_mode rmt --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode cw --attack_eps 0.3 --blackbox --exp_name resnet18-cifar10 --cla_model resnet18 --cla_network_pkl /home/data/maggie/result-newhome/defense/rmt/pgd/basemixup-betasampler/resnet18-cifar10/blackbox/20220903/00001/rmt-cifar10-dataset/rmt-trained-classifier-resnet18-on-cifar10-epoch-0021-rmt-linear-cla2.pkl --gen_model stylegan2ada --gen_network_pkl /home/data/maggie/result-newhome/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_dataset /home/data/maggie/result-newhome/project/stylegan2ada-cifar10/20220716/00000/project-cifar10-trainset --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/resnet18-cifar10/20220624/00004-pgd-eps-0.3-acc-0.23/attack-cifar10-dataset/samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/CIFAR10/defense/representation-mixup-training/resnet18-cifar10-rmt-linear-pgd-20220901.log 2>&1

# #   混合训练得到cla2防御OM-PGD 
# ResNet18 linear rmt beta(1,1) against om-pgd (eps=0.02)
CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220901.py run --mode defense --defense_mode rmt --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode om-pgd --attack_eps 0.02 --blackbox --exp_name resnet18-cifar10 --cla_model resnet18 --cla_network_pkl /home/data/maggie/result-newhome/defense/rmt/pgd/basemixup-betasampler/resnet18-cifar10/blackbox/20220903/00001/rmt-cifar10-dataset/rmt-trained-classifier-resnet18-on-cifar10-epoch-0021-rmt-linear-cla2.pkl --gen_model stylegan2ada --gen_network_pkl /home/data/maggie/result-newhome/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_dataset /home/data/maggie/result-newhome/project/stylegan2ada-cifar10/20220716/00000/project-cifar10-trainset --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/resnet18-cifar10/20220704/00000-ompgd-eps-0.02-acc-48.3/attack-cifar10-dataset/latent-attack-samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/CIFAR10/defense/representation-mixup-training/resnet18-cifar10-rmt-linear-om-pgd-20220901.log 2>&1

# ResNet18 linear rmt beta(1,1) against om-pgd (eps=0.05)
CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220901.py run --mode defense --defense_mode rmt --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode om-pgd --attack_eps 0.05 --blackbox --exp_name resnet18-cifar10 --cla_model resnet18 --cla_network_pkl /home/data/maggie/result-newhome/defense/rmt/pgd/basemixup-betasampler/resnet18-cifar10/blackbox/20220903/00001/rmt-cifar10-dataset/rmt-trained-classifier-resnet18-on-cifar10-epoch-0021-rmt-linear-cla2.pkl --gen_model stylegan2ada --gen_network_pkl /home/data/maggie/result-newhome/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_dataset /home/data/maggie/result-newhome/project/stylegan2ada-cifar10/20220716/00000/project-cifar10-trainset --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/resnet18-cifar10/20220704/00001-ompgd-eps-0.05-acc-15.02/attack-cifar10-dataset/latent-attack-samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/CIFAR10/defense/representation-mixup-training/resnet18-cifar10-rmt-linear-om-pgd-20220901.log 2>&1

# ResNet18 linear rmt beta(1,1) against om-pgd (eps=0.1)
CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220901.py run --mode defense --defense_mode rmt --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode om-pgd --attack_eps 0.1 --blackbox --exp_name resnet18-cifar10 --cla_model resnet18 --cla_network_pkl /home/data/maggie/result-newhome/defense/rmt/pgd/basemixup-betasampler/resnet18-cifar10/blackbox/20220903/00001/rmt-cifar10-dataset/rmt-trained-classifier-resnet18-on-cifar10-epoch-0021-rmt-linear-cla2.pkl --gen_model stylegan2ada --gen_network_pkl /home/data/maggie/result-newhome/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_dataset /home/data/maggie/result-newhome/project/stylegan2ada-cifar10/20220716/00000/project-cifar10-trainset --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/resnet18-cifar10/20220705/00000-ompgd-eps-0.1-acc-1.35/attack-cifar10-dataset/latent-attack-samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/CIFAR10/defense/representation-mixup-training/resnet18-cifar10-rmt-linear-om-pgd-20220901.log 2>&1

# ResNet18 linear rmt beta(1,1) against om-pgd (eps=0.2)
CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220901.py run --mode defense --defense_mode rmt --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode om-pgd --attack_eps 0.2 --blackbox --exp_name resnet18-cifar10 --cla_model resnet18 --cla_network_pkl /home/data/maggie/result-newhome/defense/rmt/pgd/basemixup-betasampler/resnet18-cifar10/blackbox/20220903/00001/rmt-cifar10-dataset/rmt-trained-classifier-resnet18-on-cifar10-epoch-0021-rmt-linear-cla2.pkl --gen_model stylegan2ada --gen_network_pkl /home/data/maggie/result-newhome/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_dataset /home/data/maggie/result-newhome/project/stylegan2ada-cifar10/20220716/00000/project-cifar10-trainset --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/resnet18-cifar10/20220705/00001-ompgd-eps-0.2-acc-0.17/attack-cifar10-dataset/latent-attack-samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/CIFAR10/defense/representation-mixup-training/resnet18-cifar10-rmt-linear-om-pgd-20220901.log 2>&1

# ResNet18 linear rmt beta(1,1) against om-pgd (eps=0.3)
CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220901.py run --mode defense --defense_mode rmt --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode om-pgd --attack_eps 0.3 --blackbox --exp_name resnet18-cifar10 --cla_model resnet18 --cla_network_pkl /home/data/maggie/result-newhome/defense/rmt/pgd/basemixup-betasampler/resnet18-cifar10/blackbox/20220903/00001/rmt-cifar10-dataset/rmt-trained-classifier-resnet18-on-cifar10-epoch-0021-rmt-linear-cla2.pkl --gen_model stylegan2ada --gen_network_pkl /home/data/maggie/result-newhome/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_dataset /home/data/maggie/result-newhome/project/stylegan2ada-cifar10/20220716/00000/project-cifar10-trainset --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/resnet18-cifar10/20220705/00002-ompgd-eps-0.3-acc-0.10/attack-cifar10-dataset/latent-attack-samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/CIFAR10/defense/representation-mixup-training/resnet18-cifar10-rmt-linear-om-pgd-20220901.log 2>&1

---------------
# #   混合训练得到cla3防御PGD 
# ResNet18 linear rmt beta(1,1) against pgd (eps=0.05)
CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220901.py run --mode defense --defense_mode rmt --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode pgd --attack_eps 0.05 --blackbox --exp_name resnet18-cifar10 --cla_model resnet18 --cla_network_pkl /home/data/maggie/result-newhome/defense/rmt/pgd/basemixup-betasampler/resnet18-cifar10/blackbox/20220904/00000/rmt-cifar10-dataset/rmt-trained-classifier-resnet18-on-cifar10-epoch-0027-rmt-linear-cla3.pkl --gen_model stylegan2ada --gen_network_pkl /home/data/maggie/result-newhome/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_dataset /home/data/maggie/result-newhome/project/stylegan2ada-cifar10/20220716/00000/project-cifar10-trainset --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/resnet18-cifar10/20220624/00001-pgd-eps-0.05-acc-34.87/attack-cifar10-dataset/samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/CIFAR10/defense/representation-mixup-training/resnet18-cifar10-rmt-linear-pgd-20220901.log 2>&1

# ResNet18 linear rmt beta(1,1) against pgd (eps=0.1)
CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220901.py run --mode defense --defense_mode rmt --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode autoattack --attack_eps 0.1 --blackbox --exp_name resnet18-cifar10 --cla_model resnet18 --cla_network_pkl /home/data/maggie/result-newhome/defense/rmt/pgd/basemixup-betasampler/resnet18-cifar10/blackbox/20220904/00000/rmt-cifar10-dataset/rmt-trained-classifier-resnet18-on-cifar10-epoch-0027-rmt-linear-cla3.pkl --gen_model stylegan2ada --gen_network_pkl /home/data/maggie/result-newhome/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_dataset /home/data/maggie/result-newhome/project/stylegan2ada-cifar10/20220716/00000/project-cifar10-trainset --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/resnet18-cifar10/20220624/00002-pgd-eps-0.1-acc-7.83/attack-cifar10-dataset/samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/CIFAR10/defense/representation-mixup-training/resnet18-cifar10-rmt-linear-pgd-20220901.log 2>&1

# ResNet18 linear rmt beta(1,1) against pgd (eps=0.2)
CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220901.py run --mode defense --defense_mode rmt --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode deepfool --attack_eps 0.2 --blackbox --exp_name resnet18-cifar10 --cla_model resnet18 --cla_network_pkl /home/data/maggie/result-newhome/defense/rmt/pgd/basemixup-betasampler/resnet18-cifar10/blackbox/20220904/00000/rmt-cifar10-dataset/rmt-trained-classifier-resnet18-on-cifar10-epoch-0027-rmt-linear-cla3.pkl --gen_model stylegan2ada --gen_network_pkl /home/data/maggie/result-newhome/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_dataset /home/data/maggie/result-newhome/project/stylegan2ada-cifar10/20220716/00000/project-cifar10-trainset --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/resnet18-cifar10/20220624/00003-pgd-eps-0.2-acc-1.04/attack-cifar10-dataset/samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/CIFAR10/defense/representation-mixup-training/resnet18-cifar10-rmt-linear-pgd-20220901.log 2>&1

# ResNet18 linear rmt beta(1,1) against pgd (eps=0.3)
CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220901.py run --mode defense --defense_mode rmt --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode cw --attack_eps 0.3 --blackbox --exp_name resnet18-cifar10 --cla_model resnet18 --cla_network_pkl /home/data/maggie/result-newhome/defense/rmt/pgd/basemixup-betasampler/resnet18-cifar10/blackbox/20220904/00000/rmt-cifar10-dataset/rmt-trained-classifier-resnet18-on-cifar10-epoch-0027-rmt-linear-cla3.pkl --gen_model stylegan2ada --gen_network_pkl /home/data/maggie/result-newhome/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_dataset /home/data/maggie/result-newhome/project/stylegan2ada-cifar10/20220716/00000/project-cifar10-trainset --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/resnet18-cifar10/20220624/00004-pgd-eps-0.3-acc-0.23/attack-cifar10-dataset/samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/CIFAR10/defense/representation-mixup-training/resnet18-cifar10-rmt-linear-pgd-20220901.log 2>&1

# #   混合训练得到cla3防御OM-PGD 
# ResNet18 linear rmt beta(1,1) against om-pgd (eps=0.02)
CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220901.py run --mode defense --defense_mode rmt --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode om-pgd --attack_eps 0.02 --blackbox --exp_name resnet18-cifar10 --cla_model resnet18 --cla_network_pkl /home/data/maggie/result-newhome/defense/rmt/pgd/basemixup-betasampler/resnet18-cifar10/blackbox/20220904/00000/rmt-cifar10-dataset/rmt-trained-classifier-resnet18-on-cifar10-epoch-0027-rmt-linear-cla3.pkl --gen_model stylegan2ada --gen_network_pkl /home/data/maggie/result-newhome/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_dataset /home/data/maggie/result-newhome/project/stylegan2ada-cifar10/20220716/00000/project-cifar10-trainset --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/resnet18-cifar10/20220704/00000-ompgd-eps-0.02-acc-48.3/attack-cifar10-dataset/latent-attack-samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/CIFAR10/defense/representation-mixup-training/resnet18-cifar10-rmt-linear-om-pgd-20220901.log 2>&1

# ResNet18 linear rmt beta(1,1) against om-pgd (eps=0.05)
CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220901.py run --mode defense --defense_mode rmt --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode om-pgd --attack_eps 0.05 --blackbox --exp_name resnet18-cifar10 --cla_model resnet18 --cla_network_pkl /home/data/maggie/result-newhome/defense/rmt/pgd/basemixup-betasampler/resnet18-cifar10/blackbox/20220904/00000/rmt-cifar10-dataset/rmt-trained-classifier-resnet18-on-cifar10-epoch-0027-rmt-linear-cla3.pkl --gen_model stylegan2ada --gen_network_pkl /home/data/maggie/result-newhome/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_dataset /home/data/maggie/result-newhome/project/stylegan2ada-cifar10/20220716/00000/project-cifar10-trainset --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/resnet18-cifar10/20220704/00001-ompgd-eps-0.05-acc-15.02/attack-cifar10-dataset/latent-attack-samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/CIFAR10/defense/representation-mixup-training/resnet18-cifar10-rmt-linear-om-pgd-20220901.log 2>&1

# ResNet18 linear rmt beta(1,1) against om-pgd (eps=0.1)
CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220901.py run --mode defense --defense_mode rmt --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode om-pgd --attack_eps 0.1 --blackbox --exp_name resnet18-cifar10 --cla_model resnet18 --cla_network_pkl /home/data/maggie/result-newhome/defense/rmt/pgd/basemixup-betasampler/resnet18-cifar10/blackbox/20220904/00000/rmt-cifar10-dataset/rmt-trained-classifier-resnet18-on-cifar10-epoch-0027-rmt-linear-cla3.pkl --gen_model stylegan2ada --gen_network_pkl /home/data/maggie/result-newhome/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_dataset /home/data/maggie/result-newhome/project/stylegan2ada-cifar10/20220716/00000/project-cifar10-trainset --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/resnet18-cifar10/20220705/00000-ompgd-eps-0.1-acc-1.35/attack-cifar10-dataset/latent-attack-samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/CIFAR10/defense/representation-mixup-training/resnet18-cifar10-rmt-linear-om-pgd-20220901.log 2>&1

# ResNet18 linear rmt beta(1,1) against om-pgd (eps=0.2)
CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220901.py run --mode defense --defense_mode rmt --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode om-pgd --attack_eps 0.2 --blackbox --exp_name resnet18-cifar10 --cla_model resnet18 --cla_network_pkl /home/data/maggie/result-newhome/defense/rmt/pgd/basemixup-betasampler/resnet18-cifar10/blackbox/20220904/00000/rmt-cifar10-dataset/rmt-trained-classifier-resnet18-on-cifar10-epoch-0027-rmt-linear-cla3.pkl --gen_model stylegan2ada --gen_network_pkl /home/data/maggie/result-newhome/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_dataset /home/data/maggie/result-newhome/project/stylegan2ada-cifar10/20220716/00000/project-cifar10-trainset --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/resnet18-cifar10/20220705/00001-ompgd-eps-0.2-acc-0.17/attack-cifar10-dataset/latent-attack-samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/CIFAR10/defense/representation-mixup-training/resnet18-cifar10-rmt-linear-om-pgd-20220901.log 2>&1

# ResNet18 linear rmt beta(1,1) against om-pgd (eps=0.3)
CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220901.py run --mode defense --defense_mode rmt --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode om-pgd --attack_eps 0.3 --blackbox --exp_name resnet18-cifar10 --cla_model resnet18 --cla_network_pkl /home/data/maggie/result-newhome/defense/rmt/pgd/basemixup-betasampler/resnet18-cifar10/blackbox/20220904/00000/rmt-cifar10-dataset/rmt-trained-classifier-resnet18-on-cifar10-epoch-0027-rmt-linear-cla3.pkl --gen_model stylegan2ada --gen_network_pkl /home/data/maggie/result-newhome/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_dataset /home/data/maggie/result-newhome/project/stylegan2ada-cifar10/20220716/00000/project-cifar10-trainset --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/resnet18-cifar10/20220705/00002-ompgd-eps-0.3-acc-0.10/attack-cifar10-dataset/latent-attack-samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/CIFAR10/defense/representation-mixup-training/resnet18-cifar10-rmt-linear-om-pgd-20220901.log 2>&1
