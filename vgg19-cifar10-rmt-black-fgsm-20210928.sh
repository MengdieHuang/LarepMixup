source ~/.bashrc
source /home/xieyi/anaconda3/bin/activate mmat

#-----------------------------black alexnet fgsm--------------------------------

# # vgg19 cifar10 rmt 20210928000  cle+mix lr =0.1 beta(2,2) 

# # vgg19 cifar10 rmt 20210928000  cle+mix lr =0.01 beta(2,2) 
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20210928.py run --mode defense --defense_mode rmt --beta_alpha 2 --attack_mode fgsm --attack_eps 0.3 --blackbox --exp_name vgg19-cifar10 --cla_model vgg19 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/vgg19-cifar10/20210908/00002-testacc-0.8705/train-cifar10-dataset/standard-trained-classifier-vgg19-on-clean-cifar10-epoch-0013.pkl --gen_model stylegan2ada --gen_network_pkl /home/maggie/mmat/result/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_dataset /home/maggie/mmat/result/project/stylegan2ada-cifar10/20210913/00000/project-cifar10-trainset --adv_dataset /home/maggie/mmat/result/attack/fgsm/alexnet-cifar10/20210917/00000-attackacc-0.070800/attack-cifar10-dataset/samples --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/vgg19-cifar10-rmt/vgg19-cifar10-rmt-20210928.log 2>&1

# # vgg19 cifar10 rmt 20210928000  cle+mix lr =0.001 beta(2,2)
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20210928.py run --mode defense --defense_mode rmt --beta_alpha 2 --attack_mode fgsm --attack_eps 0.3 --blackbox --exp_name vgg19-cifar10 --cla_model vgg19 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/vgg19-cifar10/20210908/00002-testacc-0.8705/train-cifar10-dataset/standard-trained-classifier-vgg19-on-clean-cifar10-epoch-0013.pkl --gen_model stylegan2ada --gen_network_pkl /home/maggie/mmat/result/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_dataset /home/maggie/mmat/result/project/stylegan2ada-cifar10/20210913/00000/project-cifar10-trainset --adv_dataset /home/maggie/mmat/result/attack/fgsm/alexnet-cifar10/20210917/00000-attackacc-0.070800/attack-cifar10-dataset/samples --batch_size 256 --epochs 40 --lr 0.001 >> /home/maggie/mmat/log/vgg19-cifar10-rmt/vgg19-cifar10-rmt-20210928.log 2>&1

# # vgg19 cifar10 rmt 20210928000  cle+mix lr =0.1 beta(0.5,0.5) 

# # vgg19 cifar10 rmt 20210928000  cle+mix lr =0.01 beta(0.5,0.5)
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20210928.py run --mode defense --defense_mode rmt --beta_alpha 0.5 --attack_mode fgsm --attack_eps 0.3 --blackbox --exp_name vgg19-cifar10 --cla_model vgg19 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/vgg19-cifar10/20210908/00002-testacc-0.8705/train-cifar10-dataset/standard-trained-classifier-vgg19-on-clean-cifar10-epoch-0013.pkl --gen_model stylegan2ada --gen_network_pkl /home/maggie/mmat/result/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_dataset /home/maggie/mmat/result/project/stylegan2ada-cifar10/20210913/00000/project-cifar10-trainset --adv_dataset /home/maggie/mmat/result/attack/fgsm/alexnet-cifar10/20210917/00000-attackacc-0.070800/attack-cifar10-dataset/samples --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/vgg19-cifar10-rmt/vgg19-cifar10-rmt-20210928.log 2>&1

# # vgg19 cifar10 rmt 20210928000  cle+mix lr =0.001 beta(0.5,0.5) 
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20210928.py run --mode defense --defense_mode rmt --beta_alpha 0.5 --attack_mode fgsm --attack_eps 0.3 --blackbox --exp_name vgg19-cifar10 --cla_model vgg19 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/vgg19-cifar10/20210908/00002-testacc-0.8705/train-cifar10-dataset/standard-trained-classifier-vgg19-on-clean-cifar10-epoch-0013.pkl --gen_model stylegan2ada --gen_network_pkl /home/maggie/mmat/result/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_dataset /home/maggie/mmat/result/project/stylegan2ada-cifar10/20210913/00000/project-cifar10-trainset --adv_dataset /home/maggie/mmat/result/attack/fgsm/alexnet-cifar10/20210917/00000-attackacc-0.070800/attack-cifar10-dataset/samples --batch_size 256 --epochs 40 --lr 0.001 >> /home/maggie/mmat/log/vgg19-cifar10-rmt/vgg19-cifar10-rmt-20210928.log 2>&1

#-----------------------------black resnet18 fgsm--------------------------------
# # vgg19 cifar10 rmt 20210928000  cle+mix lr =0.1 beta(2,2) 

# # vgg19 cifar10 rmt 20210928000  cle+mix lr =0.01 beta(2,2)
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20210928.py run --mode defense --defense_mode rmt --beta_alpha 2 --attack_mode fgsm --attack_eps 0.3 --blackbox --exp_name vgg19-cifar10 --cla_model vgg19 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/vgg19-cifar10/20210908/00002-testacc-0.8705/train-cifar10-dataset/standard-trained-classifier-vgg19-on-clean-cifar10-epoch-0013.pkl --gen_model stylegan2ada --gen_network_pkl /home/maggie/mmat/result/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_dataset /home/maggie/mmat/result/project/stylegan2ada-cifar10/20210913/00000/project-cifar10-trainset --adv_dataset /home/maggie/mmat/result/attack/fgsm/resnet18-cifar10/20210917/00000-attackacc-0.038200/attack-cifar10-dataset/samples --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/vgg19-cifar10-rmt/vgg19-cifar10-rmt-20210928.log 2>&1

# # vgg19 cifar10 rmt 20210928000  cle+mix lr =0.001 beta(2,2) 
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20210928.py run --mode defense --defense_mode rmt --beta_alpha 2 --attack_mode fgsm --attack_eps 0.3 --blackbox --exp_name vgg19-cifar10 --cla_model vgg19 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/vgg19-cifar10/20210908/00002-testacc-0.8705/train-cifar10-dataset/standard-trained-classifier-vgg19-on-clean-cifar10-epoch-0013.pkl --gen_model stylegan2ada --gen_network_pkl /home/maggie/mmat/result/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_dataset /home/maggie/mmat/result/project/stylegan2ada-cifar10/20210913/00000/project-cifar10-trainset --adv_dataset /home/maggie/mmat/result/attack/fgsm/resnet18-cifar10/20210917/00000-attackacc-0.038200/attack-cifar10-dataset/samples --batch_size 256 --epochs 40 --lr 0.001 >> /home/maggie/mmat/log/vgg19-cifar10-rmt/vgg19-cifar10-rmt-20210928.log 2>&1

# # vgg19 cifar10 rmt 20210928000  cle+mix lr =0.1 beta(0.5,0.5)

# # vgg19 cifar10 rmt 20210928000  cle+mix lr =0.01 beta(0.5,0.5) 
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20210928.py run --mode defense --defense_mode rmt --beta_alpha 0.5 --attack_mode fgsm --attack_eps 0.3 --blackbox --exp_name vgg19-cifar10 --cla_model vgg19 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/vgg19-cifar10/20210908/00002-testacc-0.8705/train-cifar10-dataset/standard-trained-classifier-vgg19-on-clean-cifar10-epoch-0013.pkl --gen_model stylegan2ada --gen_network_pkl /home/maggie/mmat/result/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_dataset /home/maggie/mmat/result/project/stylegan2ada-cifar10/20210913/00000/project-cifar10-trainset --adv_dataset /home/maggie/mmat/result/attack/fgsm/resnet18-cifar10/20210917/00000-attackacc-0.038200/attack-cifar10-dataset/samples --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/vgg19-cifar10-rmt/vgg19-cifar10-rmt-20210928.log 2>&1

# # vgg19 cifar10 rmt 20210928000  cle+mix lr =0.001 beta(0.5,0.5) 
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20210928.py run --mode defense --defense_mode rmt --beta_alpha 0.5 --attack_mode fgsm --attack_eps 0.3 --blackbox --exp_name vgg19-cifar10 --cla_model vgg19 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/vgg19-cifar10/20210908/00002-testacc-0.8705/train-cifar10-dataset/standard-trained-classifier-vgg19-on-clean-cifar10-epoch-0013.pkl --gen_model stylegan2ada --gen_network_pkl /home/maggie/mmat/result/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_dataset /home/maggie/mmat/result/project/stylegan2ada-cifar10/20210913/00000/project-cifar10-trainset --adv_dataset /home/maggie/mmat/result/attack/fgsm/resnet18-cifar10/20210917/00000-attackacc-0.038200/attack-cifar10-dataset/samples --batch_size 256 --epochs 40 --lr 0.001 >> /home/maggie/mmat/log/vgg19-cifar10-rmt/vgg19-cifar10-rmt-20210928.log 2>&1

#-----------------------------black resnet34 fgsm--------------------------------
# # vgg19 cifar10 rmt 20210928000  cle+mix lr =0.1 beta(2,2)

# # vgg19 cifar10 rmt 20210928000  cle+mix lr =0.01 beta(2,2)
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20210928.py run --mode defense --defense_mode rmt --beta_alpha 2 --attack_mode fgsm --attack_eps 0.3 --blackbox --exp_name vgg19-cifar10 --cla_model vgg19 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/vgg19-cifar10/20210908/00002-testacc-0.8705/train-cifar10-dataset/standard-trained-classifier-vgg19-on-clean-cifar10-epoch-0013.pkl --gen_model stylegan2ada --gen_network_pkl /home/maggie/mmat/result/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_dataset /home/maggie/mmat/result/project/stylegan2ada-cifar10/20210913/00000/project-cifar10-trainset --adv_dataset /home/maggie/mmat/result/attack/fgsm/resnet34-cifar10/20210917/00000-attackacc-0.03310/attack-cifar10-dataset/samples --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/vgg19-cifar10-rmt/vgg19-cifar10-rmt-20210928.log 2>&1

# # vgg19 cifar10 rmt 20210928000  cle+mix lr =0.001 beta(2,2)
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20210928.py run --mode defense --defense_mode rmt --beta_alpha 2 --attack_mode fgsm --attack_eps 0.3 --blackbox --exp_name vgg19-cifar10 --cla_model vgg19 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/vgg19-cifar10/20210908/00002-testacc-0.8705/train-cifar10-dataset/standard-trained-classifier-vgg19-on-clean-cifar10-epoch-0013.pkl --gen_model stylegan2ada --gen_network_pkl /home/maggie/mmat/result/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_dataset /home/maggie/mmat/result/project/stylegan2ada-cifar10/20210913/00000/project-cifar10-trainset --adv_dataset /home/maggie/mmat/result/attack/fgsm/resnet34-cifar10/20210917/00000-attackacc-0.03310/attack-cifar10-dataset/samples --batch_size 256 --epochs 40 --lr 0.001 >> /home/maggie/mmat/log/vgg19-cifar10-rmt/vgg19-cifar10-rmt-20210928.log 2>&1

# # vgg19 cifar10 rmt 20210928000  cle+mix lr =0.1 beta(0.5,0.5)

# # vgg19 cifar10 rmt 20210928000  cle+mix lr =0.01 beta(0.5,0.5) 
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20210928.py run --mode defense --defense_mode rmt --beta_alpha 0.5 --attack_mode fgsm --attack_eps 0.3 --blackbox --exp_name vgg19-cifar10 --cla_model vgg19 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/vgg19-cifar10/20210908/00002-testacc-0.8705/train-cifar10-dataset/standard-trained-classifier-vgg19-on-clean-cifar10-epoch-0013.pkl --gen_model stylegan2ada --gen_network_pkl /home/maggie/mmat/result/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_dataset /home/maggie/mmat/result/project/stylegan2ada-cifar10/20210913/00000/project-cifar10-trainset --adv_dataset /home/maggie/mmat/result/attack/fgsm/resnet34-cifar10/20210917/00000-attackacc-0.03310/attack-cifar10-dataset/samples --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/vgg19-cifar10-rmt/vgg19-cifar10-rmt-20210928.log 2>&1

# # vgg19 cifar10 rmt 20210928000  cle+mix lr =0.001 beta(0.5,0.5)
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20210928.py run --mode defense --defense_mode rmt --beta_alpha 0.5 --attack_mode fgsm --attack_eps 0.3 --blackbox --exp_name vgg19-cifar10 --cla_model vgg19 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/vgg19-cifar10/20210908/00002-testacc-0.8705/train-cifar10-dataset/standard-trained-classifier-vgg19-on-clean-cifar10-epoch-0013.pkl --gen_model stylegan2ada --gen_network_pkl /home/maggie/mmat/result/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_dataset /home/maggie/mmat/result/project/stylegan2ada-cifar10/20210913/00000/project-cifar10-trainset --adv_dataset /home/maggie/mmat/result/attack/fgsm/resnet34-cifar10/20210917/00000-attackacc-0.03310/attack-cifar10-dataset/samples --batch_size 256 --epochs 40 --lr 0.001 >> /home/maggie/mmat/log/vgg19-cifar10-rmt/vgg19-cifar10-rmt-20210928.log 2>&1

#-----------------------------black resnet50 fgsm--------------------------------
# # vgg19 cifar10 rmt 20210928000  cle+mix lr =0.1 beta(2,2)

# # vgg19 cifar10 rmt 20210928000  cle+mix lr =0.01 beta(2,2)
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20210928.py run --mode defense --defense_mode rmt --beta_alpha 2 --attack_mode fgsm --attack_eps 0.3 --blackbox --exp_name vgg19-cifar10 --cla_model vgg19 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/vgg19-cifar10/20210908/00002-testacc-0.8705/train-cifar10-dataset/standard-trained-classifier-vgg19-on-clean-cifar10-epoch-0013.pkl --gen_model stylegan2ada --gen_network_pkl /home/maggie/mmat/result/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_dataset /home/maggie/mmat/result/project/stylegan2ada-cifar10/20210913/00000/project-cifar10-trainset --adv_dataset /home/maggie/mmat/result/attack/fgsm/resnet50-cifar10/20210917/00000-attack-0.052800/attack-cifar10-dataset/samples --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/vgg19-cifar10-rmt/vgg19-cifar10-rmt-20210928.log 2>&1

# # vgg19 cifar10 rmt 20210928000  cle+mix lr =0.001 beta(2,2)
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20210928.py run --mode defense --defense_mode rmt --beta_alpha 2 --attack_mode fgsm --attack_eps 0.3 --blackbox --exp_name vgg19-cifar10 --cla_model vgg19 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/vgg19-cifar10/20210908/00002-testacc-0.8705/train-cifar10-dataset/standard-trained-classifier-vgg19-on-clean-cifar10-epoch-0013.pkl --gen_model stylegan2ada --gen_network_pkl /home/maggie/mmat/result/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_dataset /home/maggie/mmat/result/project/stylegan2ada-cifar10/20210913/00000/project-cifar10-trainset --adv_dataset /home/maggie/mmat/result/attack/fgsm/resnet50-cifar10/20210917/00000-attack-0.052800/attack-cifar10-dataset/samples --batch_size 256 --epochs 40 --lr 0.001 >> /home/maggie/mmat/log/vgg19-cifar10-rmt/vgg19-cifar10-rmt-20210928.log 2>&1

# # vgg19 cifar10 rmt 20210928000  cle+mix lr =0.1 beta(0.5,0.5)

# # vgg19 cifar10 rmt 20210928000  cle+mix lr =0.01 beta(0.5,0.5) 
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20210928.py run --mode defense --defense_mode rmt --beta_alpha 0.5 --attack_mode fgsm --attack_eps 0.3 --blackbox --exp_name vgg19-cifar10 --cla_model vgg19 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/vgg19-cifar10/20210908/00002-testacc-0.8705/train-cifar10-dataset/standard-trained-classifier-vgg19-on-clean-cifar10-epoch-0013.pkl --gen_model stylegan2ada --gen_network_pkl /home/maggie/mmat/result/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_dataset /home/maggie/mmat/result/project/stylegan2ada-cifar10/20210913/00000/project-cifar10-trainset --adv_dataset /home/maggie/mmat/result/attack/fgsm/resnet50-cifar10/20210917/00000-attack-0.052800/attack-cifar10-dataset/samples --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/vgg19-cifar10-rmt/vgg19-cifar10-rmt-20210928.log 2>&1

# # vgg19 cifar10 rmt 20210928000  cle+mix lr =0.001 beta(0.5,0.5)
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20210928.py run --mode defense --defense_mode rmt --beta_alpha 0.5 --attack_mode fgsm --attack_eps 0.3 --blackbox --exp_name vgg19-cifar10 --cla_model vgg19 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/vgg19-cifar10/20210908/00002-testacc-0.8705/train-cifar10-dataset/standard-trained-classifier-vgg19-on-clean-cifar10-epoch-0013.pkl --gen_model stylegan2ada --gen_network_pkl /home/maggie/mmat/result/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_dataset /home/maggie/mmat/result/project/stylegan2ada-cifar10/20210913/00000/project-cifar10-trainset --adv_dataset /home/maggie/mmat/result/attack/fgsm/resnet50-cifar10/20210917/00000-attack-0.052800/attack-cifar10-dataset/samples --batch_size 256 --epochs 40 --lr 0.001 >> /home/maggie/mmat/log/vgg19-cifar10-rmt/vgg19-cifar10-rmt-20210928.log 2>&1


#-----------------------------black densenet169 fgsm--------------------------------
# # vgg19 cifar10 rmt 20210928000  cle+mix lr =0.1 beta(2,2)

# # vgg19 cifar10 rmt 20210928000  cle+mix lr =0.01 beta(2,2)
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20210928.py run --mode defense --defense_mode rmt --beta_alpha 2 --attack_mode fgsm --attack_eps 0.3 --blackbox --exp_name vgg19-cifar10 --cla_model vgg19 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/vgg19-cifar10/20210908/00002-testacc-0.8705/train-cifar10-dataset/standard-trained-classifier-vgg19-on-clean-cifar10-epoch-0013.pkl --gen_model stylegan2ada --gen_network_pkl /home/maggie/mmat/result/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_dataset /home/maggie/mmat/result/project/stylegan2ada-cifar10/20210913/00000/project-cifar10-trainset --adv_dataset /home/maggie/mmat/result/attack/fgsm/densenet169-cifar10/20210917/00000-attackacc-0.04470/attack-cifar10-dataset/samples --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/vgg19-cifar10-rmt/vgg19-cifar10-rmt-20210928.log 2>&1

# # vgg19 cifar10 rmt 20210928000  cle+mix lr =0.001 beta(2,2)
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20210928.py run --mode defense --defense_mode rmt --beta_alpha 2 --attack_mode fgsm --attack_eps 0.3 --blackbox --exp_name vgg19-cifar10 --cla_model vgg19 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/vgg19-cifar10/20210908/00002-testacc-0.8705/train-cifar10-dataset/standard-trained-classifier-vgg19-on-clean-cifar10-epoch-0013.pkl --gen_model stylegan2ada --gen_network_pkl /home/maggie/mmat/result/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_dataset /home/maggie/mmat/result/project/stylegan2ada-cifar10/20210913/00000/project-cifar10-trainset --adv_dataset /home/maggie/mmat/result/attack/fgsm/densenet169-cifar10/20210917/00000-attackacc-0.04470/attack-cifar10-dataset/samples --batch_size 256 --epochs 40 --lr 0.001 >> /home/maggie/mmat/log/vgg19-cifar10-rmt/vgg19-cifar10-rmt-20210928.log 2>&1

# # vgg19 cifar10 rmt 20210928000  cle+mix lr =0.1 beta(0.5,0.5)

# # vgg19 cifar10 rmt 20210928000  cle+mix lr =0.01 beta(0.5,0.5) 
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20210928.py run --mode defense --defense_mode rmt --beta_alpha 0.5 --attack_mode fgsm --attack_eps 0.3 --blackbox --exp_name vgg19-cifar10 --cla_model vgg19 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/vgg19-cifar10/20210908/00002-testacc-0.8705/train-cifar10-dataset/standard-trained-classifier-vgg19-on-clean-cifar10-epoch-0013.pkl --gen_model stylegan2ada --gen_network_pkl /home/maggie/mmat/result/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_dataset /home/maggie/mmat/result/project/stylegan2ada-cifar10/20210913/00000/project-cifar10-trainset --adv_dataset /home/maggie/mmat/result/attack/fgsm/densenet169-cifar10/20210917/00000-attackacc-0.04470/attack-cifar10-dataset/samples --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/vgg19-cifar10-rmt/vgg19-cifar10-rmt-20210928.log 2>&1

# # vgg19 cifar10 rmt 20210928000  cle+mix lr =0.001 beta(0.5,0.5)
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20210928.py run --mode defense --defense_mode rmt --beta_alpha 0.5 --attack_mode fgsm --attack_eps 0.3 --blackbox --exp_name vgg19-cifar10 --cla_model vgg19 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/vgg19-cifar10/20210908/00002-testacc-0.8705/train-cifar10-dataset/standard-trained-classifier-vgg19-on-clean-cifar10-epoch-0013.pkl --gen_model stylegan2ada --gen_network_pkl /home/maggie/mmat/result/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_dataset /home/maggie/mmat/result/project/stylegan2ada-cifar10/20210913/00000/project-cifar10-trainset --adv_dataset /home/maggie/mmat/result/attack/fgsm/densenet169-cifar10/20210917/00000-attackacc-0.04470/attack-cifar10-dataset/samples --batch_size 256 --epochs 40 --lr 0.001 >> /home/maggie/mmat/log/vgg19-cifar10-rmt/vgg19-cifar10-rmt-20210928.log 2>&1


#-----------------------------black googlenet fgsm--------------------------------
# # vgg19 cifar10 rmt 20210928000  cle+mix lr =0.1 beta(2,2)

# # vgg19 cifar10 rmt 20210928000  cle+mix lr =0.01 beta(2,2)
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20210928.py run --mode defense --defense_mode rmt --beta_alpha 2 --attack_mode fgsm --attack_eps 0.3 --blackbox --exp_name vgg19-cifar10 --cla_model vgg19 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/vgg19-cifar10/20210908/00002-testacc-0.8705/train-cifar10-dataset/standard-trained-classifier-vgg19-on-clean-cifar10-epoch-0013.pkl --gen_model stylegan2ada --gen_network_pkl /home/maggie/mmat/result/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_dataset /home/maggie/mmat/result/project/stylegan2ada-cifar10/20210913/00000/project-cifar10-trainset --adv_dataset /home/maggie/mmat/result/attack/fgsm/googlenet-cifar10/20210917/00000-attackacc-0.06640/attack-cifar10-dataset/samples --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/vgg19-cifar10-rmt/vgg19-cifar10-rmt-20210928.log 2>&1

# # vgg19 cifar10 rmt 20210928000  cle+mix lr =0.001 beta(2,2)
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20210928.py run --mode defense --defense_mode rmt --beta_alpha 2 --attack_mode fgsm --attack_eps 0.3 --blackbox --exp_name vgg19-cifar10 --cla_model vgg19 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/vgg19-cifar10/20210908/00002-testacc-0.8705/train-cifar10-dataset/standard-trained-classifier-vgg19-on-clean-cifar10-epoch-0013.pkl --gen_model stylegan2ada --gen_network_pkl /home/maggie/mmat/result/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_dataset /home/maggie/mmat/result/project/stylegan2ada-cifar10/20210913/00000/project-cifar10-trainset --adv_dataset /home/maggie/mmat/result/attack/fgsm/googlenet-cifar10/20210917/00000-attackacc-0.06640/attack-cifar10-dataset/samples --batch_size 256 --epochs 40 --lr 0.001 >> /home/maggie/mmat/log/vgg19-cifar10-rmt/vgg19-cifar10-rmt-20210928.log 2>&1

# # vgg19 cifar10 rmt 20210928000  cle+mix lr =0.1 beta(0.5,0.5)

# # vgg19 cifar10 rmt 20210928000  cle+mix lr =0.01 beta(0.5,0.5) 
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20210928.py run --mode defense --defense_mode rmt --beta_alpha 0.5 --attack_mode fgsm --attack_eps 0.3 --blackbox --exp_name vgg19-cifar10 --cla_model vgg19 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/vgg19-cifar10/20210908/00002-testacc-0.8705/train-cifar10-dataset/standard-trained-classifier-vgg19-on-clean-cifar10-epoch-0013.pkl --gen_model stylegan2ada --gen_network_pkl /home/maggie/mmat/result/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_dataset /home/maggie/mmat/result/project/stylegan2ada-cifar10/20210913/00000/project-cifar10-trainset --adv_dataset /home/maggie/mmat/result/attack/fgsm/googlenet-cifar10/20210917/00000-attackacc-0.06640/attack-cifar10-dataset/samples --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/vgg19-cifar10-rmt/vgg19-cifar10-rmt-20210928.log 2>&1

# # vgg19 cifar10 rmt 20210928000  cle+mix lr =0.001 beta(0.5,0.5)
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20210928.py run --mode defense --defense_mode rmt --beta_alpha 0.5 --attack_mode fgsm --attack_eps 0.3 --blackbox --exp_name vgg19-cifar10 --cla_model vgg19 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/vgg19-cifar10/20210908/00002-testacc-0.8705/train-cifar10-dataset/standard-trained-classifier-vgg19-on-clean-cifar10-epoch-0013.pkl --gen_model stylegan2ada --gen_network_pkl /home/maggie/mmat/result/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_dataset /home/maggie/mmat/result/project/stylegan2ada-cifar10/20210913/00000/project-cifar10-trainset --adv_dataset /home/maggie/mmat/result/attack/fgsm/googlenet-cifar10/20210917/00000-attackacc-0.06640/attack-cifar10-dataset/samples --batch_size 256 --epochs 40 --lr 0.001 >> /home/maggie/mmat/log/vgg19-cifar10-rmt/vgg19-cifar10-rmt-20210928.log 2>&1
