source ~/.bashrc
source /home/xieyi/anaconda3/bin/activate mmat

# CIFAR10 adv50000 cat cle50000

#--------rmt训练好的cla1-----------
# resnet50 linear rmt beta(1,1) against fog (eps=0.2)
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220905.py run --mode defense --defense_mode rmt --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode fog --perceptualattack --attack_eps 0.2 --blackbox --exp_name resnet50-cifar10 --cla_model resnet50 --cla_network_pkl /home/data/maggie/result-newhome/defense/rmt/pgd/basemixup-betasampler/resnet50-cifar10/blackbox/20220905/00000/rmt-cifar10-dataset/rmt-trained-classifier-resnet50-on-cifar10-epoch-0006-rmt-linear-cla1.pkl --gen_model stylegan2ada --gen_network_pkl /home/data/maggie/result-newhome/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_dataset /home/data/maggie/result-newhome/project/stylegan2ada-cifar10/20220716/00000/project-cifar10-trainset --test_adv_dataset /home/data/maggie/result-newhome/attack/fog/resnet50-cifar10/20211020/00000-perattackacc-4.82/attack-cifar10-dataset/samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/CIFAR10/defense/representation-mixup-training/resnet50-cifar10-rmt-linear-fog-20220905.log 2>&1

# resnet50 linear rmt beta(1,1) against snow (eps=0.2)
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220905.py run --mode defense --defense_mode rmt --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode snow --perceptualattack --attack_eps 0.2 --blackbox --exp_name resnet50-cifar10 --cla_model resnet50 --cla_network_pkl /home/data/maggie/result-newhome/defense/rmt/pgd/basemixup-betasampler/resnet50-cifar10/blackbox/20220905/00000/rmt-cifar10-dataset/rmt-trained-classifier-resnet50-on-cifar10-epoch-0006-rmt-linear-cla1.pkl --gen_model stylegan2ada --gen_network_pkl /home/data/maggie/result-newhome/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_dataset /home/data/maggie/result-newhome/project/stylegan2ada-cifar10/20220716/00000/project-cifar10-trainset --test_adv_dataset /home/data/maggie/result-newhome/attack/snow/resnet50-cifar10/20211020/00000-perattackacc-54.58/attack-cifar10-dataset/samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/CIFAR10/defense/representation-mixup-training/resnet50-cifar10-rmt-linear-snow-20220905.log 2>&1

# resnet50 linear rmt beta(1,1) against elastic (eps=0.2)
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220905.py run --mode defense --defense_mode rmt --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode elastic --perceptualattack --attack_eps 0.2 --blackbox --exp_name resnet50-cifar10 --cla_model resnet50 --cla_network_pkl /home/data/maggie/result-newhome/defense/rmt/pgd/basemixup-betasampler/resnet50-cifar10/blackbox/20220905/00000/rmt-cifar10-dataset/rmt-trained-classifier-resnet50-on-cifar10-epoch-0006-rmt-linear-cla1.pkl --gen_model stylegan2ada --gen_network_pkl /home/data/maggie/result-newhome/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_dataset /home/data/maggie/result-newhome/project/stylegan2ada-cifar10/20220716/00000/project-cifar10-trainset --test_adv_dataset /home/data/maggie/result-newhome/attack/elastic/resnet50-cifar10/20211020/00000-perattackacc-5.47/attack-cifar10-dataset/samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/CIFAR10/defense/representation-mixup-training/resnet50-cifar10-rmt-linear-elastic-20220905.log 2>&1

# resnet50 linear rmt beta(1,1) against jpeg (eps=0.2)
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220905.py run --mode defense --defense_mode rmt --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode jpeg --perceptualattack --attack_eps 0.2 --blackbox --exp_name resnet50-cifar10 --cla_model resnet50 --cla_network_pkl /home/data/maggie/result-newhome/defense/rmt/pgd/basemixup-betasampler/resnet50-cifar10/blackbox/20220905/00000/rmt-cifar10-dataset/rmt-trained-classifier-resnet50-on-cifar10-epoch-0006-rmt-linear-cla1.pkl --gen_model stylegan2ada --gen_network_pkl /home/data/maggie/result-newhome/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_dataset /home/data/maggie/result-newhome/project/stylegan2ada-cifar10/20220716/00000/project-cifar10-trainset --test_adv_dataset /home/data/maggie/result-newhome/attack/jpeg/resnet50-cifar10/20211020/00000-perattackacc-0.72/attack-cifar10-dataset/samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/CIFAR10/defense/representation-mixup-training/resnet50-cifar10-rmt-linear-jpeg-20220905.log 2>&1

#--------rmt训练好的cla2-----------
# resnet50 linear rmt beta(1,1) against fog (eps=0.2)
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220905.py run --mode defense --defense_mode rmt --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode fog --perceptualattack --attack_eps 0.2 --blackbox --exp_name resnet50-cifar10 --cla_model resnet50 --cla_network_pkl /home/data/maggie/result-newhome/defense/rmt/pgd/basemixup-betasampler/resnet50-cifar10/blackbox/20220905/00000/rmt-cifar10-dataset/rmt-trained-classifier-resnet50-on-cifar10-epoch-0012-rmt-linear-cla2.pkl --gen_model stylegan2ada --gen_network_pkl /home/data/maggie/result-newhome/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_dataset /home/data/maggie/result-newhome/project/stylegan2ada-cifar10/20220716/00000/project-cifar10-trainset --test_adv_dataset /home/data/maggie/result-newhome/attack/fog/resnet50-cifar10/20211020/00000-perattackacc-4.82/attack-cifar10-dataset/samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/CIFAR10/defense/representation-mixup-training/resnet50-cifar10-rmt-linear-fog-20220905.log 2>&1

# resnet50 linear rmt beta(1,1) against snow (eps=0.2)
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220905.py run --mode defense --defense_mode rmt --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode snow --perceptualattack --attack_eps 0.2 --blackbox --exp_name resnet50-cifar10 --cla_model resnet50 --cla_network_pkl /home/data/maggie/result-newhome/defense/rmt/pgd/basemixup-betasampler/resnet50-cifar10/blackbox/20220905/00000/rmt-cifar10-dataset/rmt-trained-classifier-resnet50-on-cifar10-epoch-0012-rmt-linear-cla2.pkl --gen_model stylegan2ada --gen_network_pkl /home/data/maggie/result-newhome/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_dataset /home/data/maggie/result-newhome/project/stylegan2ada-cifar10/20220716/00000/project-cifar10-trainset --test_adv_dataset /home/data/maggie/result-newhome/attack/snow/resnet50-cifar10/20211020/00000-perattackacc-54.58/attack-cifar10-dataset/samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/CIFAR10/defense/representation-mixup-training/resnet50-cifar10-rmt-linear-snow-20220905.log 2>&1

# resnet50 linear rmt beta(1,1) against elastic (eps=0.2)
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220905.py run --mode defense --defense_mode rmt --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode elastic --perceptualattack --attack_eps 0.2 --blackbox --exp_name resnet50-cifar10 --cla_model resnet50 --cla_network_pkl /home/data/maggie/result-newhome/defense/rmt/pgd/basemixup-betasampler/resnet50-cifar10/blackbox/20220905/00000/rmt-cifar10-dataset/rmt-trained-classifier-resnet50-on-cifar10-epoch-0012-rmt-linear-cla2.pkl --gen_model stylegan2ada --gen_network_pkl /home/data/maggie/result-newhome/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_dataset /home/data/maggie/result-newhome/project/stylegan2ada-cifar10/20220716/00000/project-cifar10-trainset --test_adv_dataset /home/data/maggie/result-newhome/attack/elastic/resnet50-cifar10/20211020/00000-perattackacc-5.47/attack-cifar10-dataset/samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/CIFAR10/defense/representation-mixup-training/resnet50-cifar10-rmt-linear-elastic-20220905.log 2>&1

# resnet50 linear rmt beta(1,1) against jpeg (eps=0.2)
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220905.py run --mode defense --defense_mode rmt --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode jpeg --perceptualattack --attack_eps 0.2 --blackbox --exp_name resnet50-cifar10 --cla_model resnet50 --cla_network_pkl /home/data/maggie/result-newhome/defense/rmt/pgd/basemixup-betasampler/resnet50-cifar10/blackbox/20220905/00000/rmt-cifar10-dataset/rmt-trained-classifier-resnet50-on-cifar10-epoch-0012-rmt-linear-cla2.pkl --gen_model stylegan2ada --gen_network_pkl /home/data/maggie/result-newhome/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_dataset /home/data/maggie/result-newhome/project/stylegan2ada-cifar10/20220716/00000/project-cifar10-trainset --test_adv_dataset /home/data/maggie/result-newhome/attack/jpeg/resnet50-cifar10/20211020/00000-perattackacc-0.72/attack-cifar10-dataset/samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/CIFAR10/defense/representation-mixup-training/resnet50-cifar10-rmt-linear-jpeg-20220905.log 2>&1

#--------rmt训练好的cla3-----------
# resnet50 linear rmt beta(1,1) against fog (eps=0.2)
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220905.py run --mode defense --defense_mode rmt --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode fog --perceptualattack --attack_eps 0.2 --blackbox --exp_name resnet50-cifar10 --cla_model resnet50 --cla_network_pkl /home/data/maggie/result-newhome/defense/rmt/pgd/basemixup-betasampler/resnet50-cifar10/blackbox/20220905/00001/rmt-cifar10-dataset/rmt-trained-classifier-resnet50-on-cifar10-epoch-0012-rmt-linear-cla3.pkl --gen_model stylegan2ada --gen_network_pkl /home/data/maggie/result-newhome/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_dataset /home/data/maggie/result-newhome/project/stylegan2ada-cifar10/20220716/00000/project-cifar10-trainset --test_adv_dataset /home/data/maggie/result-newhome/attack/fog/resnet50-cifar10/20211020/00000-perattackacc-4.82/attack-cifar10-dataset/samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/CIFAR10/defense/representation-mixup-training/resnet50-cifar10-rmt-linear-fog-20220905.log 2>&1

# resnet50 linear rmt beta(1,1) against snow (eps=0.2)
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220905.py run --mode defense --defense_mode rmt --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode snow --perceptualattack --attack_eps 0.2 --blackbox --exp_name resnet50-cifar10 --cla_model resnet50 --cla_network_pkl /home/data/maggie/result-newhome/defense/rmt/pgd/basemixup-betasampler/resnet50-cifar10/blackbox/20220905/00001/rmt-cifar10-dataset/rmt-trained-classifier-resnet50-on-cifar10-epoch-0012-rmt-linear-cla3.pkl --gen_model stylegan2ada --gen_network_pkl /home/data/maggie/result-newhome/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_dataset /home/data/maggie/result-newhome/project/stylegan2ada-cifar10/20220716/00000/project-cifar10-trainset --test_adv_dataset /home/data/maggie/result-newhome/attack/snow/resnet50-cifar10/20211020/00000-perattackacc-54.58/attack-cifar10-dataset/samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/CIFAR10/defense/representation-mixup-training/resnet50-cifar10-rmt-linear-snow-20220905.log 2>&1

# resnet50 linear rmt beta(1,1) against elastic (eps=0.2)
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220905.py run --mode defense --defense_mode rmt --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode elastic --perceptualattack --attack_eps 0.2 --blackbox --exp_name resnet50-cifar10 --cla_model resnet50 --cla_network_pkl /home/data/maggie/result-newhome/defense/rmt/pgd/basemixup-betasampler/resnet50-cifar10/blackbox/20220905/00001/rmt-cifar10-dataset/rmt-trained-classifier-resnet50-on-cifar10-epoch-0012-rmt-linear-cla3.pkl --gen_model stylegan2ada --gen_network_pkl /home/data/maggie/result-newhome/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_dataset /home/data/maggie/result-newhome/project/stylegan2ada-cifar10/20220716/00000/project-cifar10-trainset --test_adv_dataset /home/data/maggie/result-newhome/attack/elastic/resnet50-cifar10/20211020/00000-perattackacc-5.47/attack-cifar10-dataset/samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/CIFAR10/defense/representation-mixup-training/resnet50-cifar10-rmt-linear-elastic-20220905.log 2>&1

# resnet50 linear rmt beta(1,1) against jpeg (eps=0.2)
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220905.py run --mode defense --defense_mode rmt --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode jpeg --perceptualattack --attack_eps 0.2 --blackbox --exp_name resnet50-cifar10 --cla_model resnet50 --cla_network_pkl /home/data/maggie/result-newhome/defense/rmt/pgd/basemixup-betasampler/resnet50-cifar10/blackbox/20220905/00001/rmt-cifar10-dataset/rmt-trained-classifier-resnet50-on-cifar10-epoch-0012-rmt-linear-cla3.pkl --gen_model stylegan2ada --gen_network_pkl /home/data/maggie/result-newhome/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_dataset /home/data/maggie/result-newhome/project/stylegan2ada-cifar10/20220716/00000/project-cifar10-trainset --test_adv_dataset /home/data/maggie/result-newhome/attack/jpeg/resnet50-cifar10/20211020/00000-perattackacc-0.72/attack-cifar10-dataset/samples/test --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/CIFAR10/defense/representation-mixup-training/resnet50-cifar10-rmt-linear-jpeg-20220905.log 2>&1
