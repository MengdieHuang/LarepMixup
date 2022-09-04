source ~/.bashrc
source /home/xieyi/anaconda3/bin/activate mmat

# adv50000 cat cle50000
# GoogleNet linear rmt beta(1,1) against PGD (eps=0.02)
CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220901.py run --mode defense --defense_mode rmt --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode pgd --attack_eps 0.02 --blackbox --exp_name googlenet-svhn --cla_model googlenet --cla_network_pkl /home/data/maggie/result-newhome/train/cla-train/googlenet-svhn/20210910/00001-testacc-0.9382/train-svhn-dataset/standard-trained-classifier-googlenet-on-clean-svhn-epoch-0011.pkl --gen_model stylegan2ada --gen_network_pkl /home/data/maggie/result-newhome/train/gen-train/stylegan2ada-svhn/20210718/00000/svhn-auto1-batch64-ada-bgc-noresume/network-snapshot-025000.pkl --dataset svhn --projected_dataset /home/data/maggie/result-newhome/project/stylegan2ada-svhn/20220715/00000/project-svhn-trainset --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/googlenet-svhn/20220703/00000-pgd-eps-0.02-acc-78.45/attack-svhn-dataset/samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/SVHN/defense/representation-mixup-training/googlenet-svhn-rmt-linear-pgd-20220901.log 2>&1

# GoogleNet linear rmt beta(1,1) against PGD (eps=0.02)
CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220901.py run --mode defense --defense_mode rmt --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode pgd --attack_eps 0.02 --blackbox --exp_name googlenet-svhn --cla_model googlenet --cla_network_pkl /home/data/maggie/result-newhome/train/cla-train/googlenet-svhn/20210910/00001-testacc-0.9382/train-svhn-dataset/standard-trained-classifier-googlenet-on-clean-svhn-epoch-0011.pkl --gen_model stylegan2ada --gen_network_pkl /home/data/maggie/result-newhome/train/gen-train/stylegan2ada-svhn/20210718/00000/svhn-auto1-batch64-ada-bgc-noresume/network-snapshot-025000.pkl --dataset svhn --projected_dataset /home/data/maggie/result-newhome/project/stylegan2ada-svhn/20220715/00000/project-svhn-trainset --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/googlenet-svhn/20220703/00000-pgd-eps-0.02-acc-78.45/attack-svhn-dataset/samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/SVHN/defense/representation-mixup-training/googlenet-svhn-rmt-linear-pgd-20220901.log 2>&1

# GoogleNet linear rmt beta(1,1) against PGD (eps=0.02)
CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220901.py run --mode defense --defense_mode rmt --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode pgd --attack_eps 0.02 --blackbox --exp_name googlenet-svhn --cla_model googlenet --cla_network_pkl /home/data/maggie/result-newhome/train/cla-train/googlenet-svhn/20210910/00001-testacc-0.9382/train-svhn-dataset/standard-trained-classifier-googlenet-on-clean-svhn-epoch-0011.pkl --gen_model stylegan2ada --gen_network_pkl /home/data/maggie/result-newhome/train/gen-train/stylegan2ada-svhn/20210718/00000/svhn-auto1-batch64-ada-bgc-noresume/network-snapshot-025000.pkl --dataset svhn --projected_dataset /home/data/maggie/result-newhome/project/stylegan2ada-svhn/20220715/00000/project-svhn-trainset --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/googlenet-svhn/20220703/00000-pgd-eps-0.02-acc-78.45/attack-svhn-dataset/samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/SVHN/defense/representation-mixup-training/googlenet-svhn-rmt-linear-pgd-20220901.log 2>&1

# #-------------以上要存储at训练好的模型-----------
# # #   混合训练得到cla1防御PGD 
# # GoogleNet linear rmt beta(1,1) against pgd (eps=0.05)
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220901.py run --mode defense --defense_mode rmt --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode pgd --attack_eps 0.05 --blackbox --exp_name googlenet-svhn --cla_model googlenet --cla_network_pkl XXX-cla1-XXX --gen_model stylegan2ada --gen_network_pkl /home/data/maggie/result-newhome/train/gen-train/stylegan2ada-svhn/20210718/00000/svhn-auto1-batch64-ada-bgc-noresume/network-snapshot-025000.pkl --dataset svhn --projected_dataset /home/data/maggie/result-newhome/project/stylegan2ada-svhn/20220715/00000/project-svhn-trainset --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/googlenet-svhn/20220703/00001-pgd-eps-0.05-acc-59.30/attack-svhn-dataset/samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/SVHN/defense/representation-mixup-training/googlenet-svhn-rmt-linear-pgd-20220901.log 2>&1

# # GoogleNet linear rmt beta(1,1) against pgd (eps=0.1)
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220901.py run --mode defense --defense_mode rmt --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode autoattack --attack_eps 0.1 --blackbox --exp_name googlenet-svhn --cla_model googlenet --cla_network_pkl XXX-cla1-XXX --gen_model stylegan2ada --gen_network_pkl /home/data/maggie/result-newhome/train/gen-train/stylegan2ada-svhn/20210718/00000/svhn-auto1-batch64-ada-bgc-noresume/network-snapshot-025000.pkl --dataset svhn --projected_dataset /home/data/maggie/result-newhome/project/stylegan2ada-svhn/20220715/00000/project-svhn-trainset --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/googlenet-svhn/20220703/00002-pgd-eps-0.1-acc-31.94/attack-svhn-dataset/samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/SVHN/defense/representation-mixup-training/googlenet-svhn-rmt-linear-pgd-20220901.log 2>&1

# # GoogleNet linear rmt beta(1,1) against pgd (eps=0.2)
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220901.py run --mode defense --defense_mode rmt --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode deepfool --attack_eps 0.2 --blackbox --exp_name googlenet-svhn --cla_model googlenet --cla_network_pkl XXX-cla1-XXX --gen_model stylegan2ada --gen_network_pkl /home/data/maggie/result-newhome/train/gen-train/stylegan2ada-svhn/20210718/00000/svhn-auto1-batch64-ada-bgc-noresume/network-snapshot-025000.pkl --dataset svhn --projected_dataset /home/data/maggie/result-newhome/project/stylegan2ada-svhn/20220715/00000/project-svhn-trainset --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/googlenet-svhn/20220703/00003-pgd-eps-0.2-acc-17.92/attack-svhn-dataset/samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/SVHN/defense/representation-mixup-training/googlenet-svhn-rmt-linear-pgd-20220901.log 2>&1

# # GoogleNet linear rmt beta(1,1) against pgd (eps=0.3)
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220901.py run --mode defense --defense_mode rmt --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode cw --attack_eps 0.3 --blackbox --exp_name googlenet-svhn --cla_model googlenet --cla_network_pkl XXX-cla1-XXX --gen_model stylegan2ada --gen_network_pkl /home/data/maggie/result-newhome/train/gen-train/stylegan2ada-svhn/20210718/00000/svhn-auto1-batch64-ada-bgc-noresume/network-snapshot-025000.pkl --dataset svhn --projected_dataset /home/data/maggie/result-newhome/project/stylegan2ada-svhn/20220715/00000/project-svhn-trainset --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/googlenet-svhn/20220704/00000-pgd-eps-0.3-acc-10.89/attack-svhn-dataset/samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/SVHN/defense/representation-mixup-training/googlenet-svhn-rmt-linear-pgd-20220901.log 2>&1

# # #   混合训练得到cla1防御OM-PGD 
# # GoogleNet linear rmt beta(1,1) against om-pgd (eps=0.02)
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220901.py run --mode defense --defense_mode rmt --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode om-pgd --attack_eps 0.02 --blackbox --exp_name googlenet-svhn --cla_model googlenet --cla_network_pkl XXX-cla1-XXX --gen_model stylegan2ada --gen_network_pkl /home/data/maggie/result-newhome/train/gen-train/stylegan2ada-svhn/20210718/00000/svhn-auto1-batch64-ada-bgc-noresume/network-snapshot-025000.pkl --dataset svhn --projected_dataset /home/data/maggie/result-newhome/project/stylegan2ada-svhn/20220715/00000/project-svhn-trainset --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/googlenet-svhn/20220716/00000-ompgd-eps-0.02-acc-60.93/attack-svhn-dataset/latent-attack-samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/SVHN/defense/representation-mixup-training/googlenet-svhn-rmt-linear-om-pgd-20220901.log 2>&1

# # GoogleNet linear rmt beta(1,1) against om-pgd (eps=0.05)
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220901.py run --mode defense --defense_mode rmt --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode om-pgd --attack_eps 0.05 --blackbox --exp_name googlenet-svhn --cla_model googlenet --cla_network_pkl XXX-cla1-XXX --gen_model stylegan2ada --gen_network_pkl /home/data/maggie/result-newhome/train/gen-train/stylegan2ada-svhn/20210718/00000/svhn-auto1-batch64-ada-bgc-noresume/network-snapshot-025000.pkl --dataset svhn --projected_dataset /home/data/maggie/result-newhome/project/stylegan2ada-svhn/20220715/00000/project-svhn-trainset --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/googlenet-svhn/20220716/00001-ompgd-eps-0.05-acc-24.56/attack-svhn-dataset/latent-attack-samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/SVHN/defense/representation-mixup-training/googlenet-svhn-rmt-linear-om-pgd-20220901.log 2>&1

# # GoogleNet linear rmt beta(1,1) against om-pgd (eps=0.1)
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220901.py run --mode defense --defense_mode rmt --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode om-pgd --attack_eps 0.1 --blackbox --exp_name googlenet-svhn --cla_model googlenet --cla_network_pkl XXX-cla1-XXX --gen_model stylegan2ada --gen_network_pkl /home/data/maggie/result-newhome/train/gen-train/stylegan2ada-svhn/20210718/00000/svhn-auto1-batch64-ada-bgc-noresume/network-snapshot-025000.pkl --dataset svhn --projected_dataset /home/data/maggie/result-newhome/project/stylegan2ada-svhn/20220715/00000/project-svhn-trainset --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/googlenet-svhn/20220716/00002-ompgd-eps-0.1-acc-5.31/attack-svhn-dataset/latent-attack-samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/SVHN/defense/representation-mixup-training/googlenet-svhn-rmt-linear-om-pgd-20220901.log 2>&1

# # GoogleNet linear rmt beta(1,1) against om-pgd (eps=0.2)
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220901.py run --mode defense --defense_mode rmt --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode om-pgd --attack_eps 0.2 --blackbox --exp_name googlenet-svhn --cla_model googlenet --cla_network_pkl XXX-cla1-XXX --gen_model stylegan2ada --gen_network_pkl /home/data/maggie/result-newhome/train/gen-train/stylegan2ada-svhn/20210718/00000/svhn-auto1-batch64-ada-bgc-noresume/network-snapshot-025000.pkl --dataset svhn --projected_dataset /home/data/maggie/result-newhome/project/stylegan2ada-svhn/20220715/00000/project-svhn-trainset --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/googlenet-svhn/20220716/00003-ompgd-eps-0.2-acc-0.90/attack-svhn-dataset/latent-attack-samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/SVHN/defense/representation-mixup-training/googlenet-svhn-rmt-linear-om-pgd-20220901.log 2>&1

# # GoogleNet linear rmt beta(1,1) against om-pgd (eps=0.3)
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220901.py run --mode defense --defense_mode rmt --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode om-pgd --attack_eps 0.3 --blackbox --exp_name googlenet-svhn --cla_model googlenet --cla_network_pkl XXX-cla1-XXX --gen_model stylegan2ada --gen_network_pkl /home/data/maggie/result-newhome/train/gen-train/stylegan2ada-svhn/20210718/00000/svhn-auto1-batch64-ada-bgc-noresume/network-snapshot-025000.pkl --dataset svhn --projected_dataset /home/data/maggie/result-newhome/project/stylegan2ada-svhn/20220715/00000/project-svhn-trainset --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/googlenet-svhn/20220716/00004-ompgd-eps-0.3-acc-0.45/attack-svhn-dataset/latent-attack-samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/SVHN/defense/representation-mixup-training/googlenet-svhn-rmt-linear-om-pgd-20220901.log 2>&1

#-------------------------------
# # #   混合训练得到cla2防御PGD 
# # GoogleNet linear rmt beta(1,1) against pgd (eps=0.05)
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220901.py run --mode defense --defense_mode rmt --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode pgd --attack_eps 0.05 --blackbox --exp_name googlenet-svhn --cla_model googlenet --cla_network_pkl XXX-cla2-XXX --gen_model stylegan2ada --gen_network_pkl /home/data/maggie/result-newhome/train/gen-train/stylegan2ada-svhn/20210718/00000/svhn-auto1-batch64-ada-bgc-noresume/network-snapshot-025000.pkl --dataset svhn --projected_dataset /home/data/maggie/result-newhome/project/stylegan2ada-svhn/20220715/00000/project-svhn-trainset --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/googlenet-svhn/20220703/00001-pgd-eps-0.05-acc-59.30/attack-svhn-dataset/samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/SVHN/defense/representation-mixup-training/googlenet-svhn-rmt-linear-pgd-20220901.log 2>&1

# # GoogleNet linear rmt beta(1,1) against pgd (eps=0.1)
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220901.py run --mode defense --defense_mode rmt --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode autoattack --attack_eps 0.1 --blackbox --exp_name googlenet-svhn --cla_model googlenet --cla_network_pkl XXX-cla2-XXX --gen_model stylegan2ada --gen_network_pkl /home/data/maggie/result-newhome/train/gen-train/stylegan2ada-svhn/20210718/00000/svhn-auto1-batch64-ada-bgc-noresume/network-snapshot-025000.pkl --dataset svhn --projected_dataset /home/data/maggie/result-newhome/project/stylegan2ada-svhn/20220715/00000/project-svhn-trainset --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/googlenet-svhn/20220703/00002-pgd-eps-0.1-acc-31.94/attack-svhn-dataset/samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/SVHN/defense/representation-mixup-training/googlenet-svhn-rmt-linear-pgd-20220901.log 2>&1

# # GoogleNet linear rmt beta(1,1) against pgd (eps=0.2)
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220901.py run --mode defense --defense_mode rmt --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode deepfool --attack_eps 0.2 --blackbox --exp_name googlenet-svhn --cla_model googlenet --cla_network_pkl XXX-cla2-XXX --gen_model stylegan2ada --gen_network_pkl /home/data/maggie/result-newhome/train/gen-train/stylegan2ada-svhn/20210718/00000/svhn-auto1-batch64-ada-bgc-noresume/network-snapshot-025000.pkl --dataset svhn --projected_dataset /home/data/maggie/result-newhome/project/stylegan2ada-svhn/20220715/00000/project-svhn-trainset --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/googlenet-svhn/20220703/00003-pgd-eps-0.2-acc-17.92/attack-svhn-dataset/samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/SVHN/defense/representation-mixup-training/googlenet-svhn-rmt-linear-pgd-20220901.log 2>&1

# # GoogleNet linear rmt beta(1,1) against pgd (eps=0.3)
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220901.py run --mode defense --defense_mode rmt --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode cw --attack_eps 0.3 --blackbox --exp_name googlenet-svhn --cla_model googlenet --cla_network_pkl XXX-cla2-XXX --gen_model stylegan2ada --gen_network_pkl /home/data/maggie/result-newhome/train/gen-train/stylegan2ada-svhn/20210718/00000/svhn-auto1-batch64-ada-bgc-noresume/network-snapshot-025000.pkl --dataset svhn --projected_dataset /home/data/maggie/result-newhome/project/stylegan2ada-svhn/20220715/00000/project-svhn-trainset --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/googlenet-svhn/20220704/00000-pgd-eps-0.3-acc-10.89/attack-svhn-dataset/samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/SVHN/defense/representation-mixup-training/googlenet-svhn-rmt-linear-pgd-20220901.log 2>&1

# # #   混合训练得到cla2防御OM-PGD 
# # GoogleNet linear rmt beta(1,1) against om-pgd (eps=0.02)
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220901.py run --mode defense --defense_mode rmt --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode om-pgd --attack_eps 0.02 --blackbox --exp_name googlenet-svhn --cla_model googlenet --cla_network_pkl XXX-cla2-XXX --gen_model stylegan2ada --gen_network_pkl /home/data/maggie/result-newhome/train/gen-train/stylegan2ada-svhn/20210718/00000/svhn-auto1-batch64-ada-bgc-noresume/network-snapshot-025000.pkl --dataset svhn --projected_dataset /home/data/maggie/result-newhome/project/stylegan2ada-svhn/20220715/00000/project-svhn-trainset --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/googlenet-svhn/20220716/00000-ompgd-eps-0.02-acc-60.93/attack-svhn-dataset/latent-attack-samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/SVHN/defense/representation-mixup-training/googlenet-svhn-rmt-linear-om-pgd-20220901.log 2>&1

# # GoogleNet linear rmt beta(1,1) against om-pgd (eps=0.05)
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220901.py run --mode defense --defense_mode rmt --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode om-pgd --attack_eps 0.05 --blackbox --exp_name googlenet-svhn --cla_model googlenet --cla_network_pkl XXX-cla2-XXX --gen_model stylegan2ada --gen_network_pkl /home/data/maggie/result-newhome/train/gen-train/stylegan2ada-svhn/20210718/00000/svhn-auto1-batch64-ada-bgc-noresume/network-snapshot-025000.pkl --dataset svhn --projected_dataset /home/data/maggie/result-newhome/project/stylegan2ada-svhn/20220715/00000/project-svhn-trainset --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/googlenet-svhn/20220716/00001-ompgd-eps-0.05-acc-24.56/attack-svhn-dataset/latent-attack-samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/SVHN/defense/representation-mixup-training/googlenet-svhn-rmt-linear-om-pgd-20220901.log 2>&1

# # GoogleNet linear rmt beta(1,1) against om-pgd (eps=0.1)
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220901.py run --mode defense --defense_mode rmt --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode om-pgd --attack_eps 0.1 --blackbox --exp_name googlenet-svhn --cla_model googlenet --cla_network_pkl XXX-cla2-XXX --gen_model stylegan2ada --gen_network_pkl /home/data/maggie/result-newhome/train/gen-train/stylegan2ada-svhn/20210718/00000/svhn-auto1-batch64-ada-bgc-noresume/network-snapshot-025000.pkl --dataset svhn --projected_dataset /home/data/maggie/result-newhome/project/stylegan2ada-svhn/20220715/00000/project-svhn-trainset --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/googlenet-svhn/20220716/00002-ompgd-eps-0.1-acc-5.31/attack-svhn-dataset/latent-attack-samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/SVHN/defense/representation-mixup-training/googlenet-svhn-rmt-linear-om-pgd-20220901.log 2>&1

# # GoogleNet linear rmt beta(1,1) against om-pgd (eps=0.2)
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220901.py run --mode defense --defense_mode rmt --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode om-pgd --attack_eps 0.2 --blackbox --exp_name googlenet-svhn --cla_model googlenet --cla_network_pkl XXX-cla2-XXX --gen_model stylegan2ada --gen_network_pkl /home/data/maggie/result-newhome/train/gen-train/stylegan2ada-svhn/20210718/00000/svhn-auto1-batch64-ada-bgc-noresume/network-snapshot-025000.pkl --dataset svhn --projected_dataset /home/data/maggie/result-newhome/project/stylegan2ada-svhn/20220715/00000/project-svhn-trainset --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/googlenet-svhn/20220716/00003-ompgd-eps-0.2-acc-0.90/attack-svhn-dataset/latent-attack-samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/SVHN/defense/representation-mixup-training/googlenet-svhn-rmt-linear-om-pgd-20220901.log 2>&1

# # GoogleNet linear rmt beta(1,1) against om-pgd (eps=0.3)
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220901.py run --mode defense --defense_mode rmt --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode om-pgd --attack_eps 0.3 --blackbox --exp_name googlenet-svhn --cla_model googlenet --cla_network_pkl XXX-cla2-XXX --gen_model stylegan2ada --gen_network_pkl /home/data/maggie/result-newhome/train/gen-train/stylegan2ada-svhn/20210718/00000/svhn-auto1-batch64-ada-bgc-noresume/network-snapshot-025000.pkl --dataset svhn --projected_dataset /home/data/maggie/result-newhome/project/stylegan2ada-svhn/20220715/00000/project-svhn-trainset --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/googlenet-svhn/20220716/00004-ompgd-eps-0.3-acc-0.45/attack-svhn-dataset/latent-attack-samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/SVHN/defense/representation-mixup-training/googlenet-svhn-rmt-linear-om-pgd-20220901.log 2>&1


#-------------------------------
# # #   混合训练得到cla3防御PGD 
# # GoogleNet linear rmt beta(1,1) against pgd (eps=0.05)
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220901.py run --mode defense --defense_mode rmt --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode pgd --attack_eps 0.05 --blackbox --exp_name googlenet-svhn --cla_model googlenet --cla_network_pkl XXX-cla3-XXX --gen_model stylegan2ada --gen_network_pkl /home/data/maggie/result-newhome/train/gen-train/stylegan2ada-svhn/20210718/00000/svhn-auto1-batch64-ada-bgc-noresume/network-snapshot-025000.pkl --dataset svhn --projected_dataset /home/data/maggie/result-newhome/project/stylegan2ada-svhn/20220715/00000/project-svhn-trainset --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/googlenet-svhn/20220703/00001-pgd-eps-0.05-acc-59.30/attack-svhn-dataset/samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/SVHN/defense/representation-mixup-training/googlenet-svhn-rmt-linear-pgd-20220901.log 2>&1

# # GoogleNet linear rmt beta(1,1) against pgd (eps=0.1)
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220901.py run --mode defense --defense_mode rmt --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode autoattack --attack_eps 0.1 --blackbox --exp_name googlenet-svhn --cla_model googlenet --cla_network_pkl XXX-cla3-XXX --gen_model stylegan2ada --gen_network_pkl /home/data/maggie/result-newhome/train/gen-train/stylegan2ada-svhn/20210718/00000/svhn-auto1-batch64-ada-bgc-noresume/network-snapshot-025000.pkl --dataset svhn --projected_dataset /home/data/maggie/result-newhome/project/stylegan2ada-svhn/20220715/00000/project-svhn-trainset --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/googlenet-svhn/20220703/00002-pgd-eps-0.1-acc-31.94/attack-svhn-dataset/samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/SVHN/defense/representation-mixup-training/googlenet-svhn-rmt-linear-pgd-20220901.log 2>&1

# # GoogleNet linear rmt beta(1,1) against pgd (eps=0.2)
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220901.py run --mode defense --defense_mode rmt --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode deepfool --attack_eps 0.2 --blackbox --exp_name googlenet-svhn --cla_model googlenet --cla_network_pkl XXX-cla3-XXX --gen_model stylegan2ada --gen_network_pkl /home/data/maggie/result-newhome/train/gen-train/stylegan2ada-svhn/20210718/00000/svhn-auto1-batch64-ada-bgc-noresume/network-snapshot-025000.pkl --dataset svhn --projected_dataset /home/data/maggie/result-newhome/project/stylegan2ada-svhn/20220715/00000/project-svhn-trainset --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/googlenet-svhn/20220703/00003-pgd-eps-0.2-acc-17.92/attack-svhn-dataset/samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/SVHN/defense/representation-mixup-training/googlenet-svhn-rmt-linear-pgd-20220901.log 2>&1

# # GoogleNet linear rmt beta(1,1) against pgd (eps=0.3)
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220901.py run --mode defense --defense_mode rmt --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode cw --attack_eps 0.3 --blackbox --exp_name googlenet-svhn --cla_model googlenet --cla_network_pkl XXX-cla3-XXX --gen_model stylegan2ada --gen_network_pkl /home/data/maggie/result-newhome/train/gen-train/stylegan2ada-svhn/20210718/00000/svhn-auto1-batch64-ada-bgc-noresume/network-snapshot-025000.pkl --dataset svhn --projected_dataset /home/data/maggie/result-newhome/project/stylegan2ada-svhn/20220715/00000/project-svhn-trainset --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/googlenet-svhn/20220704/00000-pgd-eps-0.3-acc-10.89/attack-svhn-dataset/samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/SVHN/defense/representation-mixup-training/googlenet-svhn-rmt-linear-pgd-20220901.log 2>&1

# # #   混合训练得到cla3防御OM-PGD 
# # GoogleNet linear rmt beta(1,1) against om-pgd (eps=0.02)
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220901.py run --mode defense --defense_mode rmt --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode om-pgd --attack_eps 0.02 --blackbox --exp_name googlenet-svhn --cla_model googlenet --cla_network_pkl XXX-cla3-XXX --gen_model stylegan2ada --gen_network_pkl /home/data/maggie/result-newhome/train/gen-train/stylegan2ada-svhn/20210718/00000/svhn-auto1-batch64-ada-bgc-noresume/network-snapshot-025000.pkl --dataset svhn --projected_dataset /home/data/maggie/result-newhome/project/stylegan2ada-svhn/20220715/00000/project-svhn-trainset --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/googlenet-svhn/20220716/00000-ompgd-eps-0.02-acc-60.93/attack-svhn-dataset/latent-attack-samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/SVHN/defense/representation-mixup-training/googlenet-svhn-rmt-linear-om-pgd-20220901.log 2>&1

# # GoogleNet linear rmt beta(1,1) against om-pgd (eps=0.05)
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220901.py run --mode defense --defense_mode rmt --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode om-pgd --attack_eps 0.05 --blackbox --exp_name googlenet-svhn --cla_model googlenet --cla_network_pkl XXX-cla3-XXX --gen_model stylegan2ada --gen_network_pkl /home/data/maggie/result-newhome/train/gen-train/stylegan2ada-svhn/20210718/00000/svhn-auto1-batch64-ada-bgc-noresume/network-snapshot-025000.pkl --dataset svhn --projected_dataset /home/data/maggie/result-newhome/project/stylegan2ada-svhn/20220715/00000/project-svhn-trainset --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/googlenet-svhn/20220716/00001-ompgd-eps-0.05-acc-24.56/attack-svhn-dataset/latent-attack-samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/SVHN/defense/representation-mixup-training/googlenet-svhn-rmt-linear-om-pgd-20220901.log 2>&1

# # GoogleNet linear rmt beta(1,1) against om-pgd (eps=0.1)
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220901.py run --mode defense --defense_mode rmt --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode om-pgd --attack_eps 0.1 --blackbox --exp_name googlenet-svhn --cla_model googlenet --cla_network_pkl XXX-cla3-XXX --gen_model stylegan2ada --gen_network_pkl /home/data/maggie/result-newhome/train/gen-train/stylegan2ada-svhn/20210718/00000/svhn-auto1-batch64-ada-bgc-noresume/network-snapshot-025000.pkl --dataset svhn --projected_dataset /home/data/maggie/result-newhome/project/stylegan2ada-svhn/20220715/00000/project-svhn-trainset --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/googlenet-svhn/20220716/00002-ompgd-eps-0.1-acc-5.31/attack-svhn-dataset/latent-attack-samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/SVHN/defense/representation-mixup-training/googlenet-svhn-rmt-linear-om-pgd-20220901.log 2>&1

# # GoogleNet linear rmt beta(1,1) against om-pgd (eps=0.2)
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220901.py run --mode defense --defense_mode rmt --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode om-pgd --attack_eps 0.2 --blackbox --exp_name googlenet-svhn --cla_model googlenet --cla_network_pkl XXX-cla3-XXX --gen_model stylegan2ada --gen_network_pkl /home/data/maggie/result-newhome/train/gen-train/stylegan2ada-svhn/20210718/00000/svhn-auto1-batch64-ada-bgc-noresume/network-snapshot-025000.pkl --dataset svhn --projected_dataset /home/data/maggie/result-newhome/project/stylegan2ada-svhn/20220715/00000/project-svhn-trainset --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/googlenet-svhn/20220716/00003-ompgd-eps-0.2-acc-0.90/attack-svhn-dataset/latent-attack-samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/SVHN/defense/representation-mixup-training/googlenet-svhn-rmt-linear-om-pgd-20220901.log 2>&1

# # GoogleNet linear rmt beta(1,1) against om-pgd (eps=0.3)
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220901.py run --mode defense --defense_mode rmt --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode om-pgd --attack_eps 0.3 --blackbox --exp_name googlenet-svhn --cla_model googlenet --cla_network_pkl XXX-cla3-XXX --gen_model stylegan2ada --gen_network_pkl /home/data/maggie/result-newhome/train/gen-train/stylegan2ada-svhn/20210718/00000/svhn-auto1-batch64-ada-bgc-noresume/network-snapshot-025000.pkl --dataset svhn --projected_dataset /home/data/maggie/result-newhome/project/stylegan2ada-svhn/20220715/00000/project-svhn-trainset --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/googlenet-svhn/20220716/00004-ompgd-eps-0.3-acc-0.45/attack-svhn-dataset/latent-attack-samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/SVHN/defense/representation-mixup-training/googlenet-svhn-rmt-linear-om-pgd-20220901.log 2>&1