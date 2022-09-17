source ~/.bashrc
source /home/xieyi/anaconda3/bin/activate mmat

# CIFAR10 adv50000 cat cle50000

#--------rmt训练好的cla1-----------
# densenet169 linear rmt beta(1,1) against fog (eps=0.2)
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220905.py run --mode defense --defense_mode rmt --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode fog --perceptualattack --attack_eps 0.2 --blackbox --exp_name densenet169-svhn --cla_model densenet169 --cla_network_pkl /home/data/maggie/result-newhome/defense/rmt/pgd/basemixup-betasampler/densenet169-svhn/blackbox/20220907/00000/rmt-svhn-dataset/rmt-trained-classifier-densenet169-on-svhn-epoch-0011-rmt-linear-cla1.pkl --gen_model stylegan2ada --gen_network_pkl /home/data/maggie/result-newhome/train/gen-train/stylegan2ada-svhn/20210718/00000/svhn-auto1-batch64-ada-bgc-noresume/network-snapshot-025000.pkl --dataset svhn --projected_dataset /home/data/maggie/result-newhome/project/stylegan2ada-svhn/20220715/00000/project-svhn-trainset --test_adv_dataset /home/data/maggie/result-newhome/attack/fog/densenet169-svhn/20220907/00000-eps-0.2-acc-4.68/attack-svhn-dataset/samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/SVHN/defense/representation-mixup-training/densenet169-svhn-rmt-linear-fog-20220905.log 2>&1

# densenet169 linear rmt beta(1,1) against snow (eps=0.2)
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220905.py run --mode defense --defense_mode rmt --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode snow --perceptualattack --attack_eps 0.2 --blackbox --exp_name densenet169-svhn --cla_model densenet169 --cla_network_pkl /home/data/maggie/result-newhome/defense/rmt/pgd/basemixup-betasampler/densenet169-svhn/blackbox/20220907/00000/rmt-svhn-dataset/rmt-trained-classifier-densenet169-on-svhn-epoch-0011-rmt-linear-cla1.pkl --gen_model stylegan2ada --gen_network_pkl /home/data/maggie/result-newhome/train/gen-train/stylegan2ada-svhn/20210718/00000/svhn-auto1-batch64-ada-bgc-noresume/network-snapshot-025000.pkl --dataset svhn --projected_dataset /home/data/maggie/result-newhome/project/stylegan2ada-svhn/20220715/00000/project-svhn-trainset --test_adv_dataset /home/data/maggie/result-newhome/attack/snow/densenet169-svhn/20220907/00000-eps-0.2-acc-60.00/attack-svhn-dataset/samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/SVHN/defense/representation-mixup-training/densenet169-svhn-rmt-linear-snow-20220905.log 2>&1

# densenet169 linear rmt beta(1,1) against elastic (eps=0.2)
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220905.py run --mode defense --defense_mode rmt --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode elastic --perceptualattack --attack_eps 0.2 --blackbox --exp_name densenet169-svhn --cla_model densenet169 --cla_network_pkl /home/data/maggie/result-newhome/defense/rmt/pgd/basemixup-betasampler/densenet169-svhn/blackbox/20220907/00000/rmt-svhn-dataset/rmt-trained-classifier-densenet169-on-svhn-epoch-0011-rmt-linear-cla1.pkl --gen_model stylegan2ada --gen_network_pkl /home/data/maggie/result-newhome/train/gen-train/stylegan2ada-svhn/20210718/00000/svhn-auto1-batch64-ada-bgc-noresume/network-snapshot-025000.pkl --dataset svhn --projected_dataset /home/data/maggie/result-newhome/project/stylegan2ada-svhn/20220715/00000/project-svhn-trainset --test_adv_dataset /home/data/maggie/result-newhome/attack/elastic/densenet169-svhn/20220908/00000-eps-0.2-acc-22.63/attack-svhn-dataset/samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/SVHN/defense/representation-mixup-training/densenet169-svhn-rmt-linear-elastic-20220905.log 2>&1

# densenet169 linear rmt beta(1,1) against jpeg (eps=0.2)
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220905.py run --mode defense --defense_mode rmt --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode jpeg --perceptualattack --attack_eps 0.2 --blackbox --exp_name densenet169-svhn --cla_model densenet169 --cla_network_pkl /home/data/maggie/result-newhome/defense/rmt/pgd/basemixup-betasampler/densenet169-svhn/blackbox/20220907/00000/rmt-svhn-dataset/rmt-trained-classifier-densenet169-on-svhn-epoch-0011-rmt-linear-cla1.pkl --gen_model stylegan2ada --gen_network_pkl /home/data/maggie/result-newhome/train/gen-train/stylegan2ada-svhn/20210718/00000/svhn-auto1-batch64-ada-bgc-noresume/network-snapshot-025000.pkl --dataset svhn --projected_dataset /home/data/maggie/result-newhome/project/stylegan2ada-svhn/20220715/00000/project-svhn-trainset --test_adv_dataset /home/data/maggie/result-newhome/attack/jpeg/densenet169-svhn/20220908/00000-eps-0.2-acc-0.93/attack-svhn-dataset/samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/SVHN/defense/representation-mixup-training/densenet169-svhn-rmt-linear-jpeg-20220905.log 2>&1

#--------rmt训练好的cla2-----------
# densenet169 linear rmt beta(1,1) against fog (eps=0.2)
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220905.py run --mode defense --defense_mode rmt --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode fog --perceptualattack --attack_eps 0.2 --blackbox --exp_name densenet169-svhn --cla_model densenet169 --cla_network_pkl /home/data/maggie/result-newhome/defense/rmt/pgd/basemixup-betasampler/densenet169-svhn/blackbox/20220907/00001/rmt-svhn-dataset/rmt-trained-classifier-densenet169-on-svhn-epoch-0037-rmt-linear-cla2.pkl --gen_model stylegan2ada --gen_network_pkl /home/data/maggie/result-newhome/train/gen-train/stylegan2ada-svhn/20210718/00000/svhn-auto1-batch64-ada-bgc-noresume/network-snapshot-025000.pkl --dataset svhn --projected_dataset /home/data/maggie/result-newhome/project/stylegan2ada-svhn/20220715/00000/project-svhn-trainset --test_adv_dataset /home/data/maggie/result-newhome/attack/fog/densenet169-svhn/20220907/00000-eps-0.2-acc-4.68/attack-svhn-dataset/samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/SVHN/defense/representation-mixup-training/densenet169-svhn-rmt-linear-fog-20220905.log 2>&1

# densenet169 linear rmt beta(1,1) against snow (eps=0.2)
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220905.py run --mode defense --defense_mode rmt --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode snow --perceptualattack --attack_eps 0.2 --blackbox --exp_name densenet169-svhn --cla_model densenet169 --cla_network_pkl /home/data/maggie/result-newhome/defense/rmt/pgd/basemixup-betasampler/densenet169-svhn/blackbox/20220907/00001/rmt-svhn-dataset/rmt-trained-classifier-densenet169-on-svhn-epoch-0037-rmt-linear-cla2.pkl --gen_model stylegan2ada --gen_network_pkl /home/data/maggie/result-newhome/train/gen-train/stylegan2ada-svhn/20210718/00000/svhn-auto1-batch64-ada-bgc-noresume/network-snapshot-025000.pkl --dataset svhn --projected_dataset /home/data/maggie/result-newhome/project/stylegan2ada-svhn/20220715/00000/project-svhn-trainset --test_adv_dataset /home/data/maggie/result-newhome/attack/snow/densenet169-svhn/20220907/00000-eps-0.2-acc-60.00/attack-svhn-dataset/samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/SVHN/defense/representation-mixup-training/densenet169-svhn-rmt-linear-snow-20220905.log 2>&1

# densenet169 linear rmt beta(1,1) against elastic (eps=0.2)
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220905.py run --mode defense --defense_mode rmt --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode elastic --perceptualattack --attack_eps 0.2 --blackbox --exp_name densenet169-svhn --cla_model densenet169 --cla_network_pkl /home/data/maggie/result-newhome/defense/rmt/pgd/basemixup-betasampler/densenet169-svhn/blackbox/20220907/00001/rmt-svhn-dataset/rmt-trained-classifier-densenet169-on-svhn-epoch-0037-rmt-linear-cla2.pkl --gen_model stylegan2ada --gen_network_pkl /home/data/maggie/result-newhome/train/gen-train/stylegan2ada-svhn/20210718/00000/svhn-auto1-batch64-ada-bgc-noresume/network-snapshot-025000.pkl --dataset svhn --projected_dataset /home/data/maggie/result-newhome/project/stylegan2ada-svhn/20220715/00000/project-svhn-trainset --test_adv_dataset /home/data/maggie/result-newhome/attack/elastic/densenet169-svhn/20220908/00000-eps-0.2-acc-22.63/attack-svhn-dataset/samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/SVHN/defense/representation-mixup-training/densenet169-svhn-rmt-linear-elastic-20220905.log 2>&1

# densenet169 linear rmt beta(1,1) against jpeg (eps=0.2)
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220905.py run --mode defense --defense_mode rmt --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode jpeg --perceptualattack --attack_eps 0.2 --blackbox --exp_name densenet169-svhn --cla_model densenet169 --cla_network_pkl /home/data/maggie/result-newhome/defense/rmt/pgd/basemixup-betasampler/densenet169-svhn/blackbox/20220907/00001/rmt-svhn-dataset/rmt-trained-classifier-densenet169-on-svhn-epoch-0037-rmt-linear-cla2.pkl --gen_model stylegan2ada --gen_network_pkl /home/data/maggie/result-newhome/train/gen-train/stylegan2ada-svhn/20210718/00000/svhn-auto1-batch64-ada-bgc-noresume/network-snapshot-025000.pkl --dataset svhn --projected_dataset /home/data/maggie/result-newhome/project/stylegan2ada-svhn/20220715/00000/project-svhn-trainset --test_adv_dataset /home/data/maggie/result-newhome/attack/jpeg/densenet169-svhn/20220908/00000-eps-0.2-acc-0.93/attack-svhn-dataset/samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/SVHN/defense/representation-mixup-training/densenet169-svhn-rmt-linear-jpeg-20220905.log 2>&1

#--------rmt训练好的cla3-----------
# densenet169 linear rmt beta(1,1) against fog (eps=0.2)
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220905.py run --mode defense --defense_mode rmt --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode fog --perceptualattack --attack_eps 0.2 --blackbox --exp_name densenet169-svhn --cla_model densenet169 --cla_network_pkl /home/data/maggie/result-newhome/defense/rmt/pgd/basemixup-betasampler/densenet169-svhn/blackbox/20220907/00002/rmt-svhn-dataset/rmt-trained-classifier-densenet169-on-svhn-epoch-0011-rmt-linear-cla3.pkl --gen_model stylegan2ada --gen_network_pkl /home/data/maggie/result-newhome/train/gen-train/stylegan2ada-svhn/20210718/00000/svhn-auto1-batch64-ada-bgc-noresume/network-snapshot-025000.pkl --dataset svhn --projected_dataset /home/data/maggie/result-newhome/project/stylegan2ada-svhn/20220715/00000/project-svhn-trainset --test_adv_dataset /home/data/maggie/result-newhome/attack/fog/densenet169-svhn/20220907/00000-eps-0.2-acc-4.68/attack-svhn-dataset/samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/SVHN/defense/representation-mixup-training/densenet169-svhn-rmt-linear-fog-20220905.log 2>&1

# densenet169 linear rmt beta(1,1) against snow (eps=0.2)
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220905.py run --mode defense --defense_mode rmt --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode snow --perceptualattack --attack_eps 0.2 --blackbox --exp_name densenet169-svhn --cla_model densenet169 --cla_network_pkl /home/data/maggie/result-newhome/defense/rmt/pgd/basemixup-betasampler/densenet169-svhn/blackbox/20220907/00002/rmt-svhn-dataset/rmt-trained-classifier-densenet169-on-svhn-epoch-0011-rmt-linear-cla3.pkl --gen_model stylegan2ada --gen_network_pkl /home/data/maggie/result-newhome/train/gen-train/stylegan2ada-svhn/20210718/00000/svhn-auto1-batch64-ada-bgc-noresume/network-snapshot-025000.pkl --dataset svhn --projected_dataset /home/data/maggie/result-newhome/project/stylegan2ada-svhn/20220715/00000/project-svhn-trainset --test_adv_dataset /home/data/maggie/result-newhome/attack/snow/densenet169-svhn/20220907/00000-eps-0.2-acc-60.00/attack-svhn-dataset/samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/SVHN/defense/representation-mixup-training/densenet169-svhn-rmt-linear-snow-20220905.log 2>&1

# densenet169 linear rmt beta(1,1) against elastic (eps=0.2)
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220905.py run --mode defense --defense_mode rmt --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode elastic --perceptualattack --attack_eps 0.2 --blackbox --exp_name densenet169-svhn --cla_model densenet169 --cla_network_pkl /home/data/maggie/result-newhome/defense/rmt/pgd/basemixup-betasampler/densenet169-svhn/blackbox/20220907/00002/rmt-svhn-dataset/rmt-trained-classifier-densenet169-on-svhn-epoch-0011-rmt-linear-cla3.pkl --gen_model stylegan2ada --gen_network_pkl /home/data/maggie/result-newhome/train/gen-train/stylegan2ada-svhn/20210718/00000/svhn-auto1-batch64-ada-bgc-noresume/network-snapshot-025000.pkl --dataset svhn --projected_dataset /home/data/maggie/result-newhome/project/stylegan2ada-svhn/20220715/00000/project-svhn-trainset --test_adv_dataset /home/data/maggie/result-newhome/attack/elastic/densenet169-svhn/20220908/00000-eps-0.2-acc-22.63/attack-svhn-dataset/samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/SVHN/defense/representation-mixup-training/densenet169-svhn-rmt-linear-elastic-20220905.log 2>&1

# densenet169 linear rmt beta(1,1) against jpeg (eps=0.2)
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220905.py run --mode defense --defense_mode rmt --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode jpeg --perceptualattack --attack_eps 0.2 --blackbox --exp_name densenet169-svhn --cla_model densenet169 --cla_network_pkl /home/data/maggie/result-newhome/defense/rmt/pgd/basemixup-betasampler/densenet169-svhn/blackbox/20220907/00002/rmt-svhn-dataset/rmt-trained-classifier-densenet169-on-svhn-epoch-0011-rmt-linear-cla3.pkl --gen_model stylegan2ada --gen_network_pkl /home/data/maggie/result-newhome/train/gen-train/stylegan2ada-svhn/20210718/00000/svhn-auto1-batch64-ada-bgc-noresume/network-snapshot-025000.pkl --dataset svhn --projected_dataset /home/data/maggie/result-newhome/project/stylegan2ada-svhn/20220715/00000/project-svhn-trainset --test_adv_dataset /home/data/maggie/result-newhome/attack/jpeg/densenet169-svhn/20220908/00000-eps-0.2-acc-0.93/attack-svhn-dataset/samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/SVHN/defense/representation-mixup-training/densenet169-svhn-rmt-linear-jpeg-20220905.log 2>&1