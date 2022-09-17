source ~/.bashrc
source /home/xieyi/anaconda3/bin/activate mmat

# CIFAR10 adv50000 cat cle50000

#--------rmt训练好的cla1-----------
# resnet34 linear rmt beta(1,1) against fog (eps=0.2)
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220905.py run --mode defense --defense_mode rmt --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode fog --perceptualattack --attack_eps 0.2 --blackbox --exp_name resnet34-svhn --cla_model resnet34 --cla_network_pkl /home/data/maggie/result-newhome/defense/rmt/pgd/basemixup-betasampler/resnet34-svhn/blackbox/20220904/00000/rmt-svhn-dataset/rmt-trained-classifier-resnet34-on-svhn-epoch-0011-rmt-linear-cla1.pkl --gen_model stylegan2ada --gen_network_pkl /home/data/maggie/result-newhome/train/gen-train/stylegan2ada-svhn/20210718/00000/svhn-auto1-batch64-ada-bgc-noresume/network-snapshot-025000.pkl --dataset svhn --projected_dataset /home/data/maggie/result-newhome/project/stylegan2ada-svhn/20220715/00000/project-svhn-trainset --test_adv_dataset /home/data/maggie/result-newhome/attack/fog/resnet34-svhn/20220907/00000-eps-0.2-acc-5.51/attack-svhn-dataset/samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/SVHN/defense/representation-mixup-training/resnet34-svhn-rmt-linear-fog-20220905.log 2>&1

# resnet34 linear rmt beta(1,1) against snow (eps=0.2)
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220905.py run --mode defense --defense_mode rmt --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode snow --perceptualattack --attack_eps 0.2 --blackbox --exp_name resnet34-svhn --cla_model resnet34 --cla_network_pkl /home/data/maggie/result-newhome/defense/rmt/pgd/basemixup-betasampler/resnet34-svhn/blackbox/20220904/00000/rmt-svhn-dataset/rmt-trained-classifier-resnet34-on-svhn-epoch-0011-rmt-linear-cla1.pkl --gen_model stylegan2ada --gen_network_pkl /home/data/maggie/result-newhome/train/gen-train/stylegan2ada-svhn/20210718/00000/svhn-auto1-batch64-ada-bgc-noresume/network-snapshot-025000.pkl --dataset svhn --projected_dataset /home/data/maggie/result-newhome/project/stylegan2ada-svhn/20220715/00000/project-svhn-trainset --test_adv_dataset /home/data/maggie/result-newhome/attack/snow/resnet34-svhn/20220907/00000-eps-0.2-acc-61.71/attack-svhn-dataset/samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/SVHN/defense/representation-mixup-training/resnet34-svhn-rmt-linear-snow-20220905.log 2>&1

# resnet34 linear rmt beta(1,1) against elastic (eps=0.2)
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220905.py run --mode defense --defense_mode rmt --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode elastic --perceptualattack --attack_eps 0.2 --blackbox --exp_name resnet34-svhn --cla_model resnet34 --cla_network_pkl /home/data/maggie/result-newhome/defense/rmt/pgd/basemixup-betasampler/resnet34-svhn/blackbox/20220904/00000/rmt-svhn-dataset/rmt-trained-classifier-resnet34-on-svhn-epoch-0011-rmt-linear-cla1.pkl --gen_model stylegan2ada --gen_network_pkl /home/data/maggie/result-newhome/train/gen-train/stylegan2ada-svhn/20210718/00000/svhn-auto1-batch64-ada-bgc-noresume/network-snapshot-025000.pkl --dataset svhn --projected_dataset /home/data/maggie/result-newhome/project/stylegan2ada-svhn/20220715/00000/project-svhn-trainset --test_adv_dataset /home/data/maggie/result-newhome/attack/elastic/resnet34-svhn/20220908/00000-eps-0.2-acc-24.96/attack-svhn-dataset/samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/SVHN/defense/representation-mixup-training/resnet34-svhn-rmt-linear-elastic-20220905.log 2>&1

# resnet34 linear rmt beta(1,1) against jpeg (eps=0.2)
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220905.py run --mode defense --defense_mode rmt --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode jpeg --perceptualattack --attack_eps 0.2 --blackbox --exp_name resnet34-svhn --cla_model resnet34 --cla_network_pkl /home/data/maggie/result-newhome/defense/rmt/pgd/basemixup-betasampler/resnet34-svhn/blackbox/20220904/00000/rmt-svhn-dataset/rmt-trained-classifier-resnet34-on-svhn-epoch-0011-rmt-linear-cla1.pkl --gen_model stylegan2ada --gen_network_pkl /home/data/maggie/result-newhome/train/gen-train/stylegan2ada-svhn/20210718/00000/svhn-auto1-batch64-ada-bgc-noresume/network-snapshot-025000.pkl --dataset svhn --projected_dataset /home/data/maggie/result-newhome/project/stylegan2ada-svhn/20220715/00000/project-svhn-trainset --test_adv_dataset /home/data/maggie/result-newhome/attack/jpeg/resnet34-svhn/20220908/00000-eps-0.2-acc-3.13/attack-svhn-dataset/samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/SVHN/defense/representation-mixup-training/resnet34-svhn-rmt-linear-jpeg-20220905.log 2>&1

#--------rmt训练好的cla2-----------
# resnet34 linear rmt beta(1,1) against fog (eps=0.2)
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220905.py run --mode defense --defense_mode rmt --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode fog --perceptualattack --attack_eps 0.2 --blackbox --exp_name resnet34-svhn --cla_model resnet34 --cla_network_pkl /home/data/maggie/result-newhome/defense/rmt/pgd/basemixup-betasampler/resnet34-svhn/blackbox/20220904/00001/rmt-svhn-dataset/rmt-trained-classifier-resnet34-on-svhn-epoch-0011-rmt-linear-cla2.pkl --gen_model stylegan2ada --gen_network_pkl /home/data/maggie/result-newhome/train/gen-train/stylegan2ada-svhn/20210718/00000/svhn-auto1-batch64-ada-bgc-noresume/network-snapshot-025000.pkl --dataset svhn --projected_dataset /home/data/maggie/result-newhome/project/stylegan2ada-svhn/20220715/00000/project-svhn-trainset --test_adv_dataset /home/data/maggie/result-newhome/attack/fog/resnet34-svhn/20220907/00000-eps-0.2-acc-5.51/attack-svhn-dataset/samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/SVHN/defense/representation-mixup-training/resnet34-svhn-rmt-linear-fog-20220905.log 2>&1

# resnet34 linear rmt beta(1,1) against snow (eps=0.2)
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220905.py run --mode defense --defense_mode rmt --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode snow --perceptualattack --attack_eps 0.2 --blackbox --exp_name resnet34-svhn --cla_model resnet34 --cla_network_pkl /home/data/maggie/result-newhome/defense/rmt/pgd/basemixup-betasampler/resnet34-svhn/blackbox/20220904/00001/rmt-svhn-dataset/rmt-trained-classifier-resnet34-on-svhn-epoch-0011-rmt-linear-cla2.pkl --gen_model stylegan2ada --gen_network_pkl /home/data/maggie/result-newhome/train/gen-train/stylegan2ada-svhn/20210718/00000/svhn-auto1-batch64-ada-bgc-noresume/network-snapshot-025000.pkl --dataset svhn --projected_dataset /home/data/maggie/result-newhome/project/stylegan2ada-svhn/20220715/00000/project-svhn-trainset --test_adv_dataset /home/data/maggie/result-newhome/attack/snow/resnet34-svhn/20220907/00000-eps-0.2-acc-61.71/attack-svhn-dataset/samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/SVHN/defense/representation-mixup-training/resnet34-svhn-rmt-linear-snow-20220905.log 2>&1

# resnet34 linear rmt beta(1,1) against elastic (eps=0.2)
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220905.py run --mode defense --defense_mode rmt --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode elastic --perceptualattack --attack_eps 0.2 --blackbox --exp_name resnet34-svhn --cla_model resnet34 --cla_network_pkl /home/data/maggie/result-newhome/defense/rmt/pgd/basemixup-betasampler/resnet34-svhn/blackbox/20220904/00001/rmt-svhn-dataset/rmt-trained-classifier-resnet34-on-svhn-epoch-0011-rmt-linear-cla2.pkl --gen_model stylegan2ada --gen_network_pkl /home/data/maggie/result-newhome/train/gen-train/stylegan2ada-svhn/20210718/00000/svhn-auto1-batch64-ada-bgc-noresume/network-snapshot-025000.pkl --dataset svhn --projected_dataset /home/data/maggie/result-newhome/project/stylegan2ada-svhn/20220715/00000/project-svhn-trainset --test_adv_dataset /home/data/maggie/result-newhome/attack/elastic/resnet34-svhn/20220908/00000-eps-0.2-acc-24.96/attack-svhn-dataset/samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/SVHN/defense/representation-mixup-training/resnet34-svhn-rmt-linear-elastic-20220905.log 2>&1

# resnet34 linear rmt beta(1,1) against jpeg (eps=0.2)
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220905.py run --mode defense --defense_mode rmt --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode jpeg --perceptualattack --attack_eps 0.2 --blackbox --exp_name resnet34-svhn --cla_model resnet34 --cla_network_pkl /home/data/maggie/result-newhome/defense/rmt/pgd/basemixup-betasampler/resnet34-svhn/blackbox/20220904/00001/rmt-svhn-dataset/rmt-trained-classifier-resnet34-on-svhn-epoch-0011-rmt-linear-cla2.pkl --gen_model stylegan2ada --gen_network_pkl /home/data/maggie/result-newhome/train/gen-train/stylegan2ada-svhn/20210718/00000/svhn-auto1-batch64-ada-bgc-noresume/network-snapshot-025000.pkl --dataset svhn --projected_dataset /home/data/maggie/result-newhome/project/stylegan2ada-svhn/20220715/00000/project-svhn-trainset --test_adv_dataset /home/data/maggie/result-newhome/attack/jpeg/resnet34-svhn/20220908/00000-eps-0.2-acc-3.13/attack-svhn-dataset/samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/SVHN/defense/representation-mixup-training/resnet34-svhn-rmt-linear-jpeg-20220905.log 2>&1

#--------rmt训练好的cla3-----------
# resnet34 linear rmt beta(1,1) against fog (eps=0.2)
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220905.py run --mode defense --defense_mode rmt --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode fog --perceptualattack --attack_eps 0.2 --blackbox --exp_name resnet34-svhn --cla_model resnet34 --cla_network_pkl /home/data/maggie/result-newhome/defense/rmt/pgd/basemixup-betasampler/resnet34-svhn/blackbox/20220905/00000/rmt-svhn-dataset/rmt-trained-classifier-resnet34-on-svhn-epoch-0011-rmt-linear-cla3.pkl --gen_model stylegan2ada --gen_network_pkl /home/data/maggie/result-newhome/train/gen-train/stylegan2ada-svhn/20210718/00000/svhn-auto1-batch64-ada-bgc-noresume/network-snapshot-025000.pkl --dataset svhn --projected_dataset /home/data/maggie/result-newhome/project/stylegan2ada-svhn/20220715/00000/project-svhn-trainset --test_adv_dataset /home/data/maggie/result-newhome/attack/fog/resnet34-svhn/20220907/00000-eps-0.2-acc-5.51/attack-svhn-dataset/samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/SVHN/defense/representation-mixup-training/resnet34-svhn-rmt-linear-fog-20220905.log 2>&1

# resnet34 linear rmt beta(1,1) against snow (eps=0.2)
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220905.py run --mode defense --defense_mode rmt --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode snow --perceptualattack --attack_eps 0.2 --blackbox --exp_name resnet34-svhn --cla_model resnet34 --cla_network_pkl /home/data/maggie/result-newhome/defense/rmt/pgd/basemixup-betasampler/resnet34-svhn/blackbox/20220905/00000/rmt-svhn-dataset/rmt-trained-classifier-resnet34-on-svhn-epoch-0011-rmt-linear-cla3.pkl --gen_model stylegan2ada --gen_network_pkl /home/data/maggie/result-newhome/train/gen-train/stylegan2ada-svhn/20210718/00000/svhn-auto1-batch64-ada-bgc-noresume/network-snapshot-025000.pkl --dataset svhn --projected_dataset /home/data/maggie/result-newhome/project/stylegan2ada-svhn/20220715/00000/project-svhn-trainset --test_adv_dataset /home/data/maggie/result-newhome/attack/snow/resnet34-svhn/20220907/00000-eps-0.2-acc-61.71/attack-svhn-dataset/samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/SVHN/defense/representation-mixup-training/resnet34-svhn-rmt-linear-snow-20220905.log 2>&1

# resnet34 linear rmt beta(1,1) against elastic (eps=0.2)
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220905.py run --mode defense --defense_mode rmt --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode elastic --perceptualattack --attack_eps 0.2 --blackbox --exp_name resnet34-svhn --cla_model resnet34 --cla_network_pkl /home/data/maggie/result-newhome/defense/rmt/pgd/basemixup-betasampler/resnet34-svhn/blackbox/20220905/00000/rmt-svhn-dataset/rmt-trained-classifier-resnet34-on-svhn-epoch-0011-rmt-linear-cla3.pkl --gen_model stylegan2ada --gen_network_pkl /home/data/maggie/result-newhome/train/gen-train/stylegan2ada-svhn/20210718/00000/svhn-auto1-batch64-ada-bgc-noresume/network-snapshot-025000.pkl --dataset svhn --projected_dataset /home/data/maggie/result-newhome/project/stylegan2ada-svhn/20220715/00000/project-svhn-trainset --test_adv_dataset /home/data/maggie/result-newhome/attack/elastic/resnet34-svhn/20220908/00000-eps-0.2-acc-24.96/attack-svhn-dataset/samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/SVHN/defense/representation-mixup-training/resnet34-svhn-rmt-linear-elastic-20220905.log 2>&1

# resnet34 linear rmt beta(1,1) against jpeg (eps=0.2)
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220905.py run --mode defense --defense_mode rmt --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode jpeg --perceptualattack --attack_eps 0.2 --blackbox --exp_name resnet34-svhn --cla_model resnet34 --cla_network_pkl /home/data/maggie/result-newhome/defense/rmt/pgd/basemixup-betasampler/resnet34-svhn/blackbox/20220905/00000/rmt-svhn-dataset/rmt-trained-classifier-resnet34-on-svhn-epoch-0011-rmt-linear-cla3.pkl --gen_model stylegan2ada --gen_network_pkl /home/data/maggie/result-newhome/train/gen-train/stylegan2ada-svhn/20210718/00000/svhn-auto1-batch64-ada-bgc-noresume/network-snapshot-025000.pkl --dataset svhn --projected_dataset /home/data/maggie/result-newhome/project/stylegan2ada-svhn/20220715/00000/project-svhn-trainset --test_adv_dataset /home/data/maggie/result-newhome/attack/jpeg/resnet34-svhn/20220908/00000-eps-0.2-acc-3.13/attack-svhn-dataset/samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/SVHN/defense/representation-mixup-training/resnet34-svhn-rmt-linear-jpeg-20220905.log 2>&1