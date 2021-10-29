source ~/.bashrc
source /home/xieyi/anaconda3/bin/activate mmat

# # preactresnet18 cifar10 rmt 20210927000  cle+mix lr =0.1 beta(2,2)
# CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20211003.py run --mode defense --defense_mode rmt --beta_alpha 2 --attack_mode fgsm --attack_eps 0.3 --whitebox --exp_name preactresnet18-cifar10 --cla_model preactresnet18 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/preactresnet18-cifar10/20210924/00001-teatacc-87.37/train-cifar10-dataset/standard-trained-classifier-preactresnet18-on-clean-cifar10-epoch-0011.pkl --gen_model stylegan2ada --gen_network_pkl /home/maggie/mmat/result/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_dataset /home/maggie/mmat/result/project/stylegan2ada-cifar10/20210913/00000/project-cifar10-trainset --adv_dataset /home/maggie/mmat/result/attack/fgsm/preactresnet18-cifar10/20210927/00000-attackacc-14.71/attack-cifar10-dataset/samples --batch_size 256 --epochs 40 --lr 0.1 >> /home/maggie/mmat/log/preactresnet18-cifar10-rmt/preactresnet18-cifar10-rmt-20211002.log 2>&1

# # preactresnet18 cifar10 rmt 20210927001  cle+mix lr =0.01 beta(2,2)
# CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20211003.py run --mode defense --defense_mode rmt --beta_alpha 2 --attack_mode fgsm --attack_eps 0.3 --whitebox --exp_name preactresnet18-cifar10 --cla_model preactresnet18 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/preactresnet18-cifar10/20210924/00001-teatacc-87.37/train-cifar10-dataset/standard-trained-classifier-preactresnet18-on-clean-cifar10-epoch-0011.pkl --gen_model stylegan2ada --gen_network_pkl /home/maggie/mmat/result/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_dataset /home/maggie/mmat/result/project/stylegan2ada-cifar10/20210913/00000/project-cifar10-trainset --adv_dataset /home/maggie/mmat/result/attack/fgsm/preactresnet18-cifar10/20210927/00000-attackacc-14.71/attack-cifar10-dataset/samples --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/preactresnet18-cifar10-rmt/preactresnet18-cifar10-rmt-20211002.log 2>&1

# # preactresnet18 cifar10 rmt 20210927002  cle+mix lr =0.001 beta(2,2)
# CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20211003.py run --mode defense --defense_mode rmt --beta_alpha 2 --attack_mode fgsm --attack_eps 0.3 --whitebox --exp_name preactresnet18-cifar10 --cla_model preactresnet18 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/preactresnet18-cifar10/20210924/00001-teatacc-87.37/train-cifar10-dataset/standard-trained-classifier-preactresnet18-on-clean-cifar10-epoch-0011.pkl --gen_model stylegan2ada --gen_network_pkl /home/maggie/mmat/result/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_dataset /home/maggie/mmat/result/project/stylegan2ada-cifar10/20210913/00000/project-cifar10-trainset --adv_dataset /home/maggie/mmat/result/attack/fgsm/preactresnet18-cifar10/20210927/00000-attackacc-14.71/attack-cifar10-dataset/samples --batch_size 256 --epochs 40 --lr 0.001 >> /home/maggie/mmat/log/preactresnet18-cifar10-rmt/preactresnet18-cifar10-rmt-20211002.log 2>&1


# # preactresnet18 cifar10 rmt 20210927003  cle+mix lr =0.1 beta(0.5, 0.5)
# CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20211003.py run --mode defense --defense_mode rmt --beta_alpha 0.5 --attack_mode fgsm --attack_eps 0.3 --whitebox --exp_name preactresnet18-cifar10 --cla_model preactresnet18 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/preactresnet18-cifar10/20210924/00001-teatacc-87.37/train-cifar10-dataset/standard-trained-classifier-preactresnet18-on-clean-cifar10-epoch-0011.pkl --gen_model stylegan2ada --gen_network_pkl /home/maggie/mmat/result/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_dataset /home/maggie/mmat/result/project/stylegan2ada-cifar10/20210913/00000/project-cifar10-trainset --adv_dataset /home/maggie/mmat/result/attack/fgsm/preactresnet18-cifar10/20210927/00000-attackacc-14.71/attack-cifar10-dataset/samples --batch_size 256 --epochs 40 --lr 0.1 >> /home/maggie/mmat/log/preactresnet18-cifar10-rmt/preactresnet18-cifar10-rmt-20211002.log 2>&1

# # preactresnet18 cifar10 rmt 20210927004  cle+mix lr =0.01 beta(0.5, 0.5)
# CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20211003.py run --mode defense --defense_mode rmt --beta_alpha 0.5 --attack_mode fgsm --attack_eps 0.3 --whitebox --exp_name preactresnet18-cifar10 --cla_model preactresnet18 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/preactresnet18-cifar10/20210924/00001-teatacc-87.37/train-cifar10-dataset/standard-trained-classifier-preactresnet18-on-clean-cifar10-epoch-0011.pkl --gen_model stylegan2ada --gen_network_pkl /home/maggie/mmat/result/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_dataset /home/maggie/mmat/result/project/stylegan2ada-cifar10/20210913/00000/project-cifar10-trainset --adv_dataset /home/maggie/mmat/result/attack/fgsm/preactresnet18-cifar10/20210927/00000-attackacc-14.71/attack-cifar10-dataset/samples --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/preactresnet18-cifar10-rmt/preactresnet18-cifar10-rmt-20211002.log 2>&1

# # preactresnet18 cifar10 rmt 20210927005  cle+mix lr =0.001 beta(0.5, 0.5)
# CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20211003.py run --mode defense --defense_mode rmt --beta_alpha 0.5 --attack_mode fgsm --attack_eps 0.3 --whitebox --exp_name preactresnet18-cifar10 --cla_model preactresnet18 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/preactresnet18-cifar10/20210924/00001-teatacc-87.37/train-cifar10-dataset/standard-trained-classifier-preactresnet18-on-clean-cifar10-epoch-0011.pkl --gen_model stylegan2ada --gen_network_pkl /home/maggie/mmat/result/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_dataset /home/maggie/mmat/result/project/stylegan2ada-cifar10/20210913/00000/project-cifar10-trainset --adv_dataset /home/maggie/mmat/result/attack/fgsm/preactresnet18-cifar10/20210927/00000-attackacc-14.71/attack-cifar10-dataset/samples --batch_size 256 --epochs 40 --lr 0.001 >> /home/maggie/mmat/log/preactresnet18-cifar10-rmt/preactresnet18-cifar10-rmt-20211002.log 2>&1

# # preactresnet18 cifar10 rmt 20210927005  cle+mix lr =0.001 beta(1, 1) dual convex mixup
# CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20211020.py run --mode defense --defense_mode rmt --beta_alpha 1 --attack_mode fgsm --attack_eps 0.3 --whitebox --exp_name preactresnet18-cifar10 --cla_model preactresnet18 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/preactresnet18-cifar10/20210924/00001-teatacc-87.37/train-cifar10-dataset/standard-trained-classifier-preactresnet18-on-clean-cifar10-epoch-0011.pkl --gen_model stylegan2ada --gen_network_pkl /home/maggie/mmat/result/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_dataset /home/maggie/mmat/result/project/stylegan2ada-cifar10/20210913/00000/project-cifar10-trainset --adv_dataset /home/maggie/mmat/result/attack/fgsm/preactresnet18-cifar10/20210927/00000-attackacc-14.71/attack-cifar10-dataset/samples --batch_size 256 --epochs 40 --lr 0.001 >> /home/maggie/mmat/log/preactresnet18-cifar10-rmt/preactresnet18-cifar10-rmt-20211002.log 2>&1


# #------------------------------
# # preactresnet18 cifar10 rmt 20210927005  cle+mix lr =0.001 dual mask mixup 已完成 20211027001
# CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20211026.py run --mode defense --defense_mode rmt --mix_w_num 2 --mix_mode maskmixup --sample_mode bernoullisampler --attack_mode fgsm --attack_eps 0.3 --whitebox --exp_name preactresnet18-cifar10 --cla_model preactresnet18 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/preactresnet18-cifar10/20210924/00001-teatacc-87.37/train-cifar10-dataset/standard-trained-classifier-preactresnet18-on-clean-cifar10-epoch-0011.pkl --gen_model stylegan2ada --gen_network_pkl /home/maggie/mmat/result/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_dataset /home/maggie/mmat/result/project/stylegan2ada-cifar10/20210913/00000/project-cifar10-trainset --adv_dataset /home/maggie/mmat/result/attack/fgsm/preactresnet18-cifar10/20210927/00000-attackacc-14.71/attack-cifar10-dataset/samples --batch_size 256 --epochs 40 --lr 0.001 >> /home/maggie/mmat/log/preactresnet18-cifar10-rmt/preactresnet18-cifar10-rmt-20211002.log 2>&1

# # preactresnet18 cifar10 rmt 20210927005  cle+mix lr =0.001 dirichlet(10, 10, 10) ternary convex mixup 已完成 20211027/000
# CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20211026.py run --mode defense --defense_mode rmt --mix_w_num 3 --mix_mode basemixup --sample_mode dirichletsampler --dirichlet_gama 10 --attack_mode fgsm --attack_eps 0.3 --whitebox --exp_name preactresnet18-cifar10 --cla_model preactresnet18 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/preactresnet18-cifar10/20210924/00001-teatacc-87.37/train-cifar10-dataset/standard-trained-classifier-preactresnet18-on-clean-cifar10-epoch-0011.pkl --gen_model stylegan2ada --gen_network_pkl /home/maggie/mmat/result/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_dataset /home/maggie/mmat/result/project/stylegan2ada-cifar10/20210913/00000/project-cifar10-trainset --adv_dataset /home/maggie/mmat/result/attack/fgsm/preactresnet18-cifar10/20210927/00000-attackacc-14.71/attack-cifar10-dataset/samples --batch_size 256 --epochs 40 --lr 0.001 >> /home/maggie/mmat/log/preactresnet18-cifar10-rmt/preactresnet18-cifar10-rmt-20211002.log 2>&1

# # preactresnet18 cifar10 rmt 20210927005  cle+mix lr =0.001 ternary mask mixup  bug
# CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20211026.py run --mode defense --defense_mode rmt --mix_w_num 3 --mix_mode maskmixup --sample_mode bernoullisampler3 --attack_mode fgsm --attack_eps 0.3 --whitebox --exp_name preactresnet18-cifar10 --cla_model preactresnet18 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/preactresnet18-cifar10/20210924/00001-teatacc-87.37/train-cifar10-dataset/standard-trained-classifier-preactresnet18-on-clean-cifar10-epoch-0011.pkl --gen_model stylegan2ada --gen_network_pkl /home/maggie/mmat/result/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_dataset /home/maggie/mmat/result/project/stylegan2ada-cifar10/20210913/00000/project-cifar10-trainset --adv_dataset /home/maggie/mmat/result/attack/fgsm/preactresnet18-cifar10/20210927/00000-attackacc-14.71/attack-cifar10-dataset/samples --batch_size 256 --epochs 40 --lr 0.001 >> /home/maggie/mmat/log/preactresnet18-cifar10-rmt/preactresnet18-cifar10-rmt-20211002.log 2>&1

# #---------------------gamma------------
# # preactresnet18 cifar10 rmt 20210927005  cle+mix lr =0.001 dirichlet(5, 5, 5) ternary convex mixup #20211027/00001 在进行
# CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20211026.py run --mode defense --defense_mode rmt --mix_w_num 3 --mix_mode basemixup --sample_mode dirichletsampler --dirichlet_gama 5 --attack_mode fgsm --attack_eps 0.3 --whitebox --exp_name preactresnet18-cifar10 --cla_model preactresnet18 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/preactresnet18-cifar10/20210924/00001-teatacc-87.37/train-cifar10-dataset/standard-trained-classifier-preactresnet18-on-clean-cifar10-epoch-0011.pkl --gen_model stylegan2ada --gen_network_pkl /home/maggie/mmat/result/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_dataset /home/maggie/mmat/result/project/stylegan2ada-cifar10/20210913/00000/project-cifar10-trainset --adv_dataset /home/maggie/mmat/result/attack/fgsm/preactresnet18-cifar10/20210927/00000-attackacc-14.71/attack-cifar10-dataset/samples --batch_size 256 --epochs 40 --lr 0.001 >> /home/maggie/mmat/log/preactresnet18-cifar10-rmt/preactresnet18-cifar10-rmt-20211002.log 2>&1

# # preactresnet18 cifar10 rmt 20210927005  cle+mix lr =0.001 ternary mask mixup  已经debug
# CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20211026.py run --mode defense --defense_mode rmt --mix_w_num 3 --mix_mode maskmixup --sample_mode bernoullisampler3 --attack_mode fgsm --attack_eps 0.3 --whitebox --exp_name preactresnet18-cifar10 --cla_model preactresnet18 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/preactresnet18-cifar10/20210924/00001-teatacc-87.37/train-cifar10-dataset/standard-trained-classifier-preactresnet18-on-clean-cifar10-epoch-0011.pkl --gen_model stylegan2ada --gen_network_pkl /home/maggie/mmat/result/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_dataset /home/maggie/mmat/result/project/stylegan2ada-cifar10/20210913/00000/project-cifar10-trainset --adv_dataset /home/maggie/mmat/result/attack/fgsm/preactresnet18-cifar10/20210927/00000-attackacc-14.71/attack-cifar10-dataset/samples --batch_size 256 --epochs 40 --lr 0.001 >> /home/maggie/mmat/log/preactresnet18-cifar10-rmt/preactresnet18-cifar10-rmt-20211002.log 2>&1

# # preactresnet18 cifar10 rmt 20210927005  cle+mix lr =0.001 dirichlet(1, 1, 1) ternary convex mixup
# CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20211026.py run --mode defense --defense_mode rmt --mix_w_num 3 --mix_mode basemixup --sample_mode dirichletsampler --dirichlet_gama 1 --attack_mode fgsm --attack_eps 0.3 --whitebox --exp_name preactresnet18-cifar10 --cla_model preactresnet18 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/preactresnet18-cifar10/20210924/00001-teatacc-87.37/train-cifar10-dataset/standard-trained-classifier-preactresnet18-on-clean-cifar10-epoch-0011.pkl --gen_model stylegan2ada --gen_network_pkl /home/maggie/mmat/result/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_dataset /home/maggie/mmat/result/project/stylegan2ada-cifar10/20210913/00000/project-cifar10-trainset --adv_dataset /home/maggie/mmat/result/attack/fgsm/preactresnet18-cifar10/20210927/00000-attackacc-14.71/attack-cifar10-dataset/samples --batch_size 256 --epochs 40 --lr 0.001 >> /home/maggie/mmat/log/preactresnet18-cifar10-rmt/preactresnet18-cifar10-rmt-20211002.log 2>&1
