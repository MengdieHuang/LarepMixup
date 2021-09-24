source ~/.bashrc
source /home/xieyi/anaconda3/bin/activate mmat

# # resnet18 cifar10 rmt 20210923000  cle+mix lr =0.1 beta(2,2)
# CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20210922.py run --mode defense --defense_mode rmt --beta_alpha 2 --attack_mode fgsm --attack_eps 0.3 --whitebox True --exp_name resnet18-cifar10 --cla_model resnet18 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/resnet18-cifar10/20210908/00017-testacc-0.7779/train-cifar10-dataset/standard-trained-classifier-resnet18-on-clean-cifar10-finished.pkl --gen_model stylegan2ada --gen_network_pkl /home/maggie/mmat/result/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_dataset /home/maggie/mmat/result/project/stylegan2ada-cifar10/20210913/00000/project-cifar10-trainset --adv_dataset /home/maggie/mmat/result/attack/fgsm/resnet18-cifar10/20210917/00000-attackacc-0.038200/attack-cifar10-dataset/samples --batch_size 256 --epochs 40 --lr 0.1 >> /home/maggie/mmat/log/resnet18-cifar10-rmt/resnet18-cifar10-rmt-20210923.log 2>&1

# # resnet18 cifar10 rmt 20210923000  cle+mix lr =0.01 beta(2,2)
# CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20210922.py run --mode defense --defense_mode rmt --beta_alpha 2 --attack_mode fgsm --attack_eps 0.3 --whitebox True --exp_name resnet18-cifar10 --cla_model resnet18 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/resnet18-cifar10/20210908/00017-testacc-0.7779/train-cifar10-dataset/standard-trained-classifier-resnet18-on-clean-cifar10-finished.pkl --gen_model stylegan2ada --gen_network_pkl /home/maggie/mmat/result/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_dataset /home/maggie/mmat/result/project/stylegan2ada-cifar10/20210913/00000/project-cifar10-trainset --adv_dataset /home/maggie/mmat/result/attack/fgsm/resnet18-cifar10/20210917/00000-attackacc-0.038200/attack-cifar10-dataset/samples --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/resnet18-cifar10-rmt/resnet18-cifar10-rmt-20210923.log 2>&1

# # resnet18 cifar10 rmt 20210923000  cle+mix lr =0.001 beta(2,2)
# CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20210922.py run --mode defense --defense_mode rmt --beta_alpha 2 --attack_mode fgsm --attack_eps 0.3 --whitebox True --exp_name resnet18-cifar10 --cla_model resnet18 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/resnet18-cifar10/20210908/00017-testacc-0.7779/train-cifar10-dataset/standard-trained-classifier-resnet18-on-clean-cifar10-finished.pkl --gen_model stylegan2ada --gen_network_pkl /home/maggie/mmat/result/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_dataset /home/maggie/mmat/result/project/stylegan2ada-cifar10/20210913/00000/project-cifar10-trainset --adv_dataset /home/maggie/mmat/result/attack/fgsm/resnet18-cifar10/20210917/00000-attackacc-0.038200/attack-cifar10-dataset/samples --batch_size 256 --epochs 40 --lr 0.001 >> /home/maggie/mmat/log/resnet18-cifar10-rmt/resnet18-cifar10-rmt-20210923.log 2>&1



# # resnet18 cifar10 rmt 20210923000  cle+mix lr =0.1 beta(0.5,0.5)
# CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20210922.py run --mode defense --defense_mode rmt --beta_alpha 0.5 --attack_mode fgsm --attack_eps 0.3 --whitebox True --exp_name resnet18-cifar10 --cla_model resnet18 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/resnet18-cifar10/20210908/00017-testacc-0.7779/train-cifar10-dataset/standard-trained-classifier-resnet18-on-clean-cifar10-finished.pkl --gen_model stylegan2ada --gen_network_pkl /home/maggie/mmat/result/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_dataset /home/maggie/mmat/result/project/stylegan2ada-cifar10/20210913/00000/project-cifar10-trainset --adv_dataset /home/maggie/mmat/result/attack/fgsm/resnet18-cifar10/20210917/00000-attackacc-0.038200/attack-cifar10-dataset/samples --batch_size 256 --epochs 40 --lr 0.1 >> /home/maggie/mmat/log/resnet18-cifar10-rmt/resnet18-cifar10-rmt-20210923.log 2>&1

# # resnet18 cifar10 rmt 20210923000  cle+mix lr =0.01 beta(0.5,0.5)
# CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20210922.py run --mode defense --defense_mode rmt --beta_alpha 0.5 --attack_mode fgsm --attack_eps 0.3 --whitebox True --exp_name resnet18-cifar10 --cla_model resnet18 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/resnet18-cifar10/20210908/00017-testacc-0.7779/train-cifar10-dataset/standard-trained-classifier-resnet18-on-clean-cifar10-finished.pkl --gen_model stylegan2ada --gen_network_pkl /home/maggie/mmat/result/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_dataset /home/maggie/mmat/result/project/stylegan2ada-cifar10/20210913/00000/project-cifar10-trainset --adv_dataset /home/maggie/mmat/result/attack/fgsm/resnet18-cifar10/20210917/00000-attackacc-0.038200/attack-cifar10-dataset/samples --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/resnet18-cifar10-rmt/resnet18-cifar10-rmt-20210923.log 2>&1

# # resnet18 cifar10 rmt 20210923000  cle+mix lr =0.001 beta(0.5,0.5)
# CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20210922.py run --mode defense --defense_mode rmt --beta_alpha 0.5 --attack_mode fgsm --attack_eps 0.3 --whitebox True --exp_name resnet18-cifar10 --cla_model resnet18 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/resnet18-cifar10/20210908/00017-testacc-0.7779/train-cifar10-dataset/standard-trained-classifier-resnet18-on-clean-cifar10-finished.pkl --gen_model stylegan2ada --gen_network_pkl /home/maggie/mmat/result/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_dataset /home/maggie/mmat/result/project/stylegan2ada-cifar10/20210913/00000/project-cifar10-trainset --adv_dataset /home/maggie/mmat/result/attack/fgsm/resnet18-cifar10/20210917/00000-attackacc-0.038200/attack-cifar10-dataset/samples --batch_size 256 --epochs 40 --lr 0.001 >> /home/maggie/mmat/log/resnet18-cifar10-rmt/resnet18-cifar10-rmt-20210923.log 2>&1



# # resnet34 cifar10 rmt 20210923000  cle+mix lr =0.1 beta(2,2)
# CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20210922.py run --mode defense --defense_mode rmt --beta_alpha 2 --attack_mode fgsm --attack_eps 0.3 --whitebox True --exp_name resnet34-cifar10 --cla_model resnet34 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/resnet34-cifar10/20210907/00030-testacc-0.7739/train-cifar10-dataset/standard-trained-classifier-resnet34-on-clean-cifar10-finished.pkl --gen_model stylegan2ada --gen_network_pkl /home/maggie/mmat/result/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_dataset /home/maggie/mmat/result/project/stylegan2ada-cifar10/20210913/00000/project-cifar10-trainset --adv_dataset /home/maggie/mmat/result/attack/fgsm/resnet34-cifar10/20210917/00000-attackacc-0.03310/attack-cifar10-dataset/samples --batch_size 256 --epochs 40 --lr 0.1 >> /home/maggie/mmat/log/resnet34-cifar10-rmt/resnet34-cifar10-rmt-20210923.log 2>&1

# # resnet34 cifar10 rmt 20210923000  cle+mix lr =0.01 beta(2,2)
# CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20210922.py run --mode defense --defense_mode rmt --beta_alpha 2 --attack_mode fgsm --attack_eps 0.3 --whitebox True --exp_name resnet34-cifar10 --cla_model resnet34 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/resnet34-cifar10/20210907/00030-testacc-0.7739/train-cifar10-dataset/standard-trained-classifier-resnet34-on-clean-cifar10-finished.pkl --gen_model stylegan2ada --gen_network_pkl /home/maggie/mmat/result/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_dataset /home/maggie/mmat/result/project/stylegan2ada-cifar10/20210913/00000/project-cifar10-trainset --adv_dataset /home/maggie/mmat/result/attack/fgsm/resnet34-cifar10/20210917/00000-attackacc-0.03310/attack-cifar10-dataset/samples --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/resnet34-cifar10-rmt/resnet34-cifar10-rmt-20210923.log 2>&1

# # resnet34 cifar10 rmt 20210923000  cle+mix lr =0.001 beta(2,2)
# CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20210922.py run --mode defense --defense_mode rmt --beta_alpha 2 --attack_mode fgsm --attack_eps 0.3 --whitebox True --exp_name resnet34-cifar10 --cla_model resnet34 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/resnet34-cifar10/20210907/00030-testacc-0.7739/train-cifar10-dataset/standard-trained-classifier-resnet34-on-clean-cifar10-finished.pkl --gen_model stylegan2ada --gen_network_pkl /home/maggie/mmat/result/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_dataset /home/maggie/mmat/result/project/stylegan2ada-cifar10/20210913/00000/project-cifar10-trainset --adv_dataset /home/maggie/mmat/result/attack/fgsm/resnet34-cifar10/20210917/00000-attackacc-0.03310/attack-cifar10-dataset/samples --batch_size 256 --epochs 40 --lr 0.001 >> /home/maggie/mmat/log/resnet34-cifar10-rmt/resnet34-cifar10-rmt-20210923.log 2>&1

# # resnet50 cifar10 rmt 20210923000  cle+mix lr =0.1 beta(2,2)
# CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20210922.py run --mode defense --defense_mode rmt --beta_alpha 2 --attack_mode fgsm --attack_eps 0.3 --whitebox True --exp_name resnet50-cifar10 --cla_model resnet50 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/resnet50-cifar10/20210908/00002-attackacc-0.7624/train-cifar10-dataset/standard-trained-classifier-resnet50-on-clean-cifar10-finished.pkl --gen_model stylegan2ada --gen_network_pkl /home/maggie/mmat/result/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_dataset /home/maggie/mmat/result/project/stylegan2ada-cifar10/20210913/00000/project-cifar10-trainset --adv_dataset /home/maggie/mmat/result/attack/fgsm/resnet50-cifar10/20210917/00000-attack-0.052800/attack-cifar10-dataset/samples --batch_size 256 --epochs 40 --lr 0.1 >> /home/maggie/mmat/log/resnet50-cifar10-rmt/resnet50-cifar10-rmt-20210922.log 2>&1

# # resnet50 cifar10 rmt 20210923000  cle+mix lr =0.01 beta(2,2)
# CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20210922.py run --mode defense --defense_mode rmt --beta_alpha 2 --attack_mode fgsm --attack_eps 0.3 --whitebox True --exp_name resnet50-cifar10 --cla_model resnet50 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/resnet50-cifar10/20210908/00002-attackacc-0.7624/train-cifar10-dataset/standard-trained-classifier-resnet50-on-clean-cifar10-finished.pkl --gen_model stylegan2ada --gen_network_pkl /home/maggie/mmat/result/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_dataset /home/maggie/mmat/result/project/stylegan2ada-cifar10/20210913/00000/project-cifar10-trainset --adv_dataset /home/maggie/mmat/result/attack/fgsm/resnet50-cifar10/20210917/00000-attack-0.052800/attack-cifar10-dataset/samples --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/resnet50-cifar10-rmt/resnet50-cifar10-rmt-20210922.log 2>&1

# # resnet50 cifar10 rmt 20210923000  cle+mix lr =0.001 beta(2,2)
# CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20210922.py run --mode defense --defense_mode rmt --beta_alpha 2 --attack_mode fgsm --attack_eps 0.3 --whitebox True --exp_name resnet50-cifar10 --cla_model resnet50 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/resnet50-cifar10/20210908/00002-attackacc-0.7624/train-cifar10-dataset/standard-trained-classifier-resnet50-on-clean-cifar10-finished.pkl --gen_model stylegan2ada --gen_network_pkl /home/maggie/mmat/result/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_dataset /home/maggie/mmat/result/project/stylegan2ada-cifar10/20210913/00000/project-cifar10-trainset --adv_dataset /home/maggie/mmat/result/attack/fgsm/resnet50-cifar10/20210917/00000-attack-0.052800/attack-cifar10-dataset/samples --batch_size 256 --epochs 40 --lr 0.001 >> /home/maggie/mmat/log/resnet50-cifar10-rmt/resnet50-cifar10-rmt-20210922.log 2>&1

# # alexnet cifar10 rmt 20210924000  cle+mix lr =0.1 beta(2,2)
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20210922.py run --mode defense --defense_mode rmt --beta_alpha 2 --attack_mode fgsm --attack_eps 0.3 --whitebox True --exp_name alexnet-cifar10 --cla_model alexnet --cla_network_pkl /home/maggie/mmat/result/train/cla-train/alexnet-cifar10/20210908/00015-testacc-0.8355/train-cifar10-dataset/standard-trained-classifier-alexnet-on-clean-cifar10-finished.pkl --gen_model stylegan2ada --gen_network_pkl /home/maggie/mmat/result/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_dataset /home/maggie/mmat/result/project/stylegan2ada-cifar10/20210913/00000/project-cifar10-trainset --adv_dataset /home/maggie/mmat/result/attack/fgsm/alexnet-cifar10/20210917/00000-attackacc-0.070800/attack-cifar10-dataset/samples --batch_size 256 --epochs 50 --lr 0.1 >> /home/maggie/mmat/log/alexnet-cifar10-rmt/alexnet-cifar10-rmt-20210924.log 2>&1

# # alexnet cifar10 rmt 20210924000  cle+mix lr =0.01 beta(2,2)
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20210922.py run --mode defense --defense_mode rmt --beta_alpha 2 --attack_mode fgsm --attack_eps 0.3 --whitebox True --exp_name alexnet-cifar10 --cla_model alexnet --cla_network_pkl /home/maggie/mmat/result/train/cla-train/alexnet-cifar10/20210908/00015-testacc-0.8355/train-cifar10-dataset/standard-trained-classifier-alexnet-on-clean-cifar10-finished.pkl --gen_model stylegan2ada --gen_network_pkl /home/maggie/mmat/result/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_dataset /home/maggie/mmat/result/project/stylegan2ada-cifar10/20210913/00000/project-cifar10-trainset --adv_dataset /home/maggie/mmat/result/attack/fgsm/alexnet-cifar10/20210917/00000-attackacc-0.070800/attack-cifar10-dataset/samples --batch_size 256 --epochs 50 --lr 0.01 >> /home/maggie/mmat/log/alexnet-cifar10-rmt/alexnet-cifar10-rmt-20210924.log 2>&1

# # alexnet cifar10 rmt 20210924000  cle+mix lr =0.001 beta(2,2)
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20210922.py run --mode defense --defense_mode rmt --beta_alpha 2 --attack_mode fgsm --attack_eps 0.3 --whitebox True --exp_name alexnet-cifar10 --cla_model alexnet --cla_network_pkl /home/maggie/mmat/result/train/cla-train/alexnet-cifar10/20210908/00015-testacc-0.8355/train-cifar10-dataset/standard-trained-classifier-alexnet-on-clean-cifar10-finished.pkl --gen_model stylegan2ada --gen_network_pkl /home/maggie/mmat/result/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_dataset /home/maggie/mmat/result/project/stylegan2ada-cifar10/20210913/00000/project-cifar10-trainset --adv_dataset /home/maggie/mmat/result/attack/fgsm/alexnet-cifar10/20210917/00000-attackacc-0.070800/attack-cifar10-dataset/samples --batch_size 256 --epochs 50 --lr 0.001 >> /home/maggie/mmat/log/alexnet-cifar10-rmt/alexnet-cifar10-rmt-20210924.log 2>&1

# # alexnet cifar10 rmt 20210924000  cle+mix lr =0.1 beta(0.5,0.5)
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20210922.py run --mode defense --defense_mode rmt --beta_alpha 0.5 --attack_mode fgsm --attack_eps 0.3 --whitebox True --exp_name alexnet-cifar10 --cla_model alexnet --cla_network_pkl /home/maggie/mmat/result/train/cla-train/alexnet-cifar10/20210908/00015-testacc-0.8355/train-cifar10-dataset/standard-trained-classifier-alexnet-on-clean-cifar10-finished.pkl --gen_model stylegan2ada --gen_network_pkl /home/maggie/mmat/result/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_dataset /home/maggie/mmat/result/project/stylegan2ada-cifar10/20210913/00000/project-cifar10-trainset --adv_dataset /home/maggie/mmat/result/attack/fgsm/alexnet-cifar10/20210917/00000-attackacc-0.070800/attack-cifar10-dataset/samples --batch_size 256 --epochs 50 --lr 0.1 >> /home/maggie/mmat/log/alexnet-cifar10-rmt/alexnet-cifar10-rmt-20210924.log 2>&1

# # alexnet cifar10 rmt 20210924000  cle+mix lr =0.01 beta(0.5,0.5)
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20210922.py run --mode defense --defense_mode rmt --beta_alpha 0.5 --attack_mode fgsm --attack_eps 0.3 --whitebox True --exp_name alexnet-cifar10 --cla_model alexnet --cla_network_pkl /home/maggie/mmat/result/train/cla-train/alexnet-cifar10/20210908/00015-testacc-0.8355/train-cifar10-dataset/standard-trained-classifier-alexnet-on-clean-cifar10-finished.pkl --gen_model stylegan2ada --gen_network_pkl /home/maggie/mmat/result/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_dataset /home/maggie/mmat/result/project/stylegan2ada-cifar10/20210913/00000/project-cifar10-trainset --adv_dataset /home/maggie/mmat/result/attack/fgsm/alexnet-cifar10/20210917/00000-attackacc-0.070800/attack-cifar10-dataset/samples --batch_size 256 --epochs 50 --lr 0.01 >> /home/maggie/mmat/log/alexnet-cifar10-rmt/alexnet-cifar10-rmt-20210924.log 2>&1

# # alexnet cifar10 rmt 20210924000  cle+mix lr =0.001 beta(0.5,0.5)
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20210922.py run --mode defense --defense_mode rmt --beta_alpha 0.5 --attack_mode fgsm --attack_eps 0.3 --whitebox True --exp_name alexnet-cifar10 --cla_model alexnet --cla_network_pkl /home/maggie/mmat/result/train/cla-train/alexnet-cifar10/20210908/00015-testacc-0.8355/train-cifar10-dataset/standard-trained-classifier-alexnet-on-clean-cifar10-finished.pkl --gen_model stylegan2ada --gen_network_pkl /home/maggie/mmat/result/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_dataset /home/maggie/mmat/result/project/stylegan2ada-cifar10/20210913/00000/project-cifar10-trainset --adv_dataset /home/maggie/mmat/result/attack/fgsm/alexnet-cifar10/20210917/00000-attackacc-0.070800/attack-cifar10-dataset/samples --batch_size 256 --epochs 50 --lr 0.001 >> /home/maggie/mmat/log/alexnet-cifar10-rmt/alexnet-cifar10-rmt-20210924.log 2>&1