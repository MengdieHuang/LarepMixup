source ~/.bashrc
source /home/xieyi/anaconda3/bin/activate mmat

# #   -----------------------------black alexnet om-fgsm--------------------------------

# # resnet18 cifar10 rmt 20210928000  cle+mix lr =0.01 beta(2,2)
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20211010.py run --mode defense --defense_mode rmt --beta_alpha 2 --attack_mode om-fgsm --attack_eps 0.3 --blackbox --exp_name resnet18-cifar10 --cla_model resnet18 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/resnet18-cifar10/20210908/00017-testacc-0.7779/train-cifar10-dataset/standard-trained-classifier-resnet18-on-clean-cifar10-finished.pkl --gen_model stylegan2ada --gen_network_pkl /home/maggie/mmat/result/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_dataset /home/maggie/mmat/result/project/stylegan2ada-cifar10/20210913/00000/project-cifar10-trainset --adv_dataset /home/maggie/mmat/result/attack/fgsm/alexnet-cifar10/20211010/00003-epsilon-0.3/attack-cifar10-dataset/latent-attack-samples --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/resnet18-cifar10-rmt/resnet18-cifar10-rmt-20211010.log 2>&1

# # resnet18 cifar10 rmt 20210928000  cle+mix lr =0.001 beta(2,2)
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20211010.py run --mode defense --defense_mode rmt --beta_alpha 2 --attack_mode om-fgsm --attack_eps 0.3 --blackbox --exp_name resnet18-cifar10 --cla_model resnet18 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/resnet18-cifar10/20210908/00017-testacc-0.7779/train-cifar10-dataset/standard-trained-classifier-resnet18-on-clean-cifar10-finished.pkl --gen_model stylegan2ada --gen_network_pkl /home/maggie/mmat/result/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_dataset /home/maggie/mmat/result/project/stylegan2ada-cifar10/20210913/00000/project-cifar10-trainset --adv_dataset /home/maggie/mmat/result/attack/fgsm/alexnet-cifar10/20211010/00003-epsilon-0.3/attack-cifar10-dataset/latent-attack-samples --batch_size 256 --epochs 40 --lr 0.001 >> /home/maggie/mmat/log/resnet18-cifar10-rmt/resnet18-cifar10-rmt-20211010.log 2>&1

# # resnet18 cifar10 rmt 20210928000  cle+mix lr =0.01 beta(0.5,0.5) 
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20211010.py run --mode defense --defense_mode rmt --beta_alpha 0.5 --attack_mode om-fgsm --attack_eps 0.3 --blackbox --exp_name resnet18-cifar10 --cla_model resnet18 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/resnet18-cifar10/20210908/00017-testacc-0.7779/train-cifar10-dataset/standard-trained-classifier-resnet18-on-clean-cifar10-finished.pkl --gen_model stylegan2ada --gen_network_pkl /home/maggie/mmat/result/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_dataset /home/maggie/mmat/result/project/stylegan2ada-cifar10/20210913/00000/project-cifar10-trainset --adv_dataset /home/maggie/mmat/result/attack/fgsm/alexnet-cifar10/20211010/00003-epsilon-0.3/attack-cifar10-dataset/latent-attack-samples --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/resnet18-cifar10-rmt/resnet18-cifar10-rmt-20211010.log 2>&1

# # resnet18 cifar10 rmt 20210928000  cle+mix lr =0.001 beta(0.5,0.5)
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20211010.py run --mode defense --defense_mode rmt --beta_alpha 0.5 --attack_mode om-fgsm --attack_eps 0.3 --blackbox --exp_name resnet18-cifar10 --cla_model resnet18 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/resnet18-cifar10/20210908/00017-testacc-0.7779/train-cifar10-dataset/standard-trained-classifier-resnet18-on-clean-cifar10-finished.pkl --gen_model stylegan2ada --gen_network_pkl /home/maggie/mmat/result/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_dataset /home/maggie/mmat/result/project/stylegan2ada-cifar10/20210913/00000/project-cifar10-trainset --adv_dataset /home/maggie/mmat/result/attack/fgsm/alexnet-cifar10/20211010/00003-epsilon-0.3/attack-cifar10-dataset/latent-attack-samples --batch_size 256 --epochs 40 --lr 0.001 >> /home/maggie/mmat/log/resnet18-cifar10-rmt/resnet18-cifar10-rmt-20211010.log 2>&1



# #   -----------------------------black resnet18 om-fgsm--------------------------------

# # resnet18 cifar10 rmt 20210928000  cle+mix lr =0.01 beta(2,2)   
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20211010.py run --mode defense --defense_mode rmt --beta_alpha 2 --attack_mode om-fgsm --attack_eps 0.3 --blackbox --exp_name resnet18-cifar10 --cla_model resnet18 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/resnet18-cifar10/20210908/00017-testacc-0.7779/train-cifar10-dataset/standard-trained-classifier-resnet18-on-clean-cifar10-finished.pkl --gen_model stylegan2ada --gen_network_pkl /home/maggie/mmat/result/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_dataset /home/maggie/mmat/result/project/stylegan2ada-cifar10/20210913/00000/project-cifar10-trainset --adv_dataset /home/maggie/mmat/result/attack/fgsm/resnet18-cifar10/20211010/00004-epsilon-0.3/attack-cifar10-dataset/latent-attack-samples --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/resnet18-cifar10-rmt/resnet18-cifar10-rmt-20211010.log 2>&1


# # resnet18 cifar10 rmt 20210928000  cle+mix lr =0.001 beta(2,2)   
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20211010.py run --mode defense --defense_mode rmt --beta_alpha 2 --attack_mode om-fgsm --attack_eps 0.3 --blackbox --exp_name resnet18-cifar10 --cla_model resnet18 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/resnet18-cifar10/20210908/00017-testacc-0.7779/train-cifar10-dataset/standard-trained-classifier-resnet18-on-clean-cifar10-finished.pkl --gen_model stylegan2ada --gen_network_pkl /home/maggie/mmat/result/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_dataset /home/maggie/mmat/result/project/stylegan2ada-cifar10/20210913/00000/project-cifar10-trainset --adv_dataset /home/maggie/mmat/result/attack/fgsm/resnet18-cifar10/20211010/00004-epsilon-0.3/attack-cifar10-dataset/latent-attack-samples --batch_size 256 --epochs 40 --lr 0.001 >> /home/maggie/mmat/log/resnet18-cifar10-rmt/resnet18-cifar10-rmt-20211010.log 2>&1

# # resnet18 cifar10 rmt 20210928000  cle+mix lr =0.01 beta(0.5,0.5)   
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20211010.py run --mode defense --defense_mode rmt --beta_alpha 0.5 --attack_mode om-fgsm --attack_eps 0.3 --blackbox --exp_name resnet18-cifar10 --cla_model resnet18 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/resnet18-cifar10/20210908/00017-testacc-0.7779/train-cifar10-dataset/standard-trained-classifier-resnet18-on-clean-cifar10-finished.pkl --gen_model stylegan2ada --gen_network_pkl /home/maggie/mmat/result/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_dataset /home/maggie/mmat/result/project/stylegan2ada-cifar10/20210913/00000/project-cifar10-trainset --adv_dataset /home/maggie/mmat/result/attack/fgsm/resnet18-cifar10/20211010/00004-epsilon-0.3/attack-cifar10-dataset/latent-attack-samples --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/resnet18-cifar10-rmt/resnet18-cifar10-rmt-20211010.log 2>&1

# # resnet18 cifar10 rmt 20210928000  cle+mix lr =0.001 beta(0.5,0.5)   
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20211010.py run --mode defense --defense_mode rmt --beta_alpha 0.5 --attack_mode om-fgsm --attack_eps 0.3 --blackbox --exp_name resnet18-cifar10 --cla_model resnet18 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/resnet18-cifar10/20210908/00017-testacc-0.7779/train-cifar10-dataset/standard-trained-classifier-resnet18-on-clean-cifar10-finished.pkl --gen_model stylegan2ada --gen_network_pkl /home/maggie/mmat/result/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_dataset /home/maggie/mmat/result/project/stylegan2ada-cifar10/20210913/00000/project-cifar10-trainset --adv_dataset /home/maggie/mmat/result/attack/fgsm/resnet18-cifar10/20211010/00004-epsilon-0.3/attack-cifar10-dataset/latent-attack-samples --batch_size 256 --epochs 40 --lr 0.001 >> /home/maggie/mmat/log/resnet18-cifar10-rmt/resnet18-cifar10-rmt-20211010.log 2>&1



# #   -----------------------------black resnet34 om-fgsm--------------------------------

# # resnet18 cifar10 rmt 20210928000  cle+mix lr =0.01 beta(2,2)
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20211010.py run --mode defense --defense_mode rmt --beta_alpha 2 --attack_mode om-fgsm --attack_eps 0.3 --blackbox --exp_name resnet18-cifar10 --cla_model resnet18 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/resnet18-cifar10/20210908/00017-testacc-0.7779/train-cifar10-dataset/standard-trained-classifier-resnet18-on-clean-cifar10-finished.pkl --gen_model stylegan2ada --gen_network_pkl /home/maggie/mmat/result/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_dataset /home/maggie/mmat/result/project/stylegan2ada-cifar10/20210913/00000/project-cifar10-trainset --adv_dataset /home/maggie/mmat/result/attack/fgsm/resnet34-cifar10/20211010/00004-epsilon-0.3/attack-cifar10-dataset/latent-attack-samples --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/resnet18-cifar10-rmt/resnet18-cifar10-rmt-20211010.log 2>&1


# # resnet18 cifar10 rmt 20210928000  cle+mix lr =0.001 beta(2,2) 
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20211010.py run --mode defense --defense_mode rmt --beta_alpha 2 --attack_mode om-fgsm --attack_eps 0.3 --blackbox --exp_name resnet18-cifar10 --cla_model resnet18 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/resnet18-cifar10/20210908/00017-testacc-0.7779/train-cifar10-dataset/standard-trained-classifier-resnet18-on-clean-cifar10-finished.pkl --gen_model stylegan2ada --gen_network_pkl /home/maggie/mmat/result/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_dataset /home/maggie/mmat/result/project/stylegan2ada-cifar10/20210913/00000/project-cifar10-trainset --adv_dataset /home/maggie/mmat/result/attack/fgsm/resnet34-cifar10/20211010/00004-epsilon-0.3/attack-cifar10-dataset/latent-attack-samples --batch_size 256 --epochs 40 --lr 0.001 >> /home/maggie/mmat/log/resnet18-cifar10-rmt/resnet18-cifar10-rmt-20211010.log 2>&1

# # resnet18 cifar10 rmt 20210928000  cle+mix lr =0.01 beta(0.5,0.5)
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20211010.py run --mode defense --defense_mode rmt --beta_alpha 0.5 --attack_mode om-fgsm --attack_eps 0.3 --blackbox --exp_name resnet18-cifar10 --cla_model resnet18 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/resnet18-cifar10/20210908/00017-testacc-0.7779/train-cifar10-dataset/standard-trained-classifier-resnet18-on-clean-cifar10-finished.pkl --gen_model stylegan2ada --gen_network_pkl /home/maggie/mmat/result/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_dataset /home/maggie/mmat/result/project/stylegan2ada-cifar10/20210913/00000/project-cifar10-trainset --adv_dataset /home/maggie/mmat/result/attack/fgsm/resnet34-cifar10/20211010/00004-epsilon-0.3/attack-cifar10-dataset/latent-attack-samples --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/resnet18-cifar10-rmt/resnet18-cifar10-rmt-20211010.log 2>&1

# # resnet18 cifar10 rmt 20210928000  cle+mix lr =0.001 beta(0.5,0.5)
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20211010.py run --mode defense --defense_mode rmt --beta_alpha 0.5 --attack_mode om-fgsm --attack_eps 0.3 --blackbox --exp_name resnet18-cifar10 --cla_model resnet18 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/resnet18-cifar10/20210908/00017-testacc-0.7779/train-cifar10-dataset/standard-trained-classifier-resnet18-on-clean-cifar10-finished.pkl --gen_model stylegan2ada --gen_network_pkl /home/maggie/mmat/result/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_dataset /home/maggie/mmat/result/project/stylegan2ada-cifar10/20210913/00000/project-cifar10-trainset --adv_dataset /home/maggie/mmat/result/attack/fgsm/resnet34-cifar10/20211010/00004-epsilon-0.3/attack-cifar10-dataset/latent-attack-samples --batch_size 256 --epochs 40 --lr 0.001 >> /home/maggie/mmat/log/resnet18-cifar10-rmt/resnet18-cifar10-rmt-20211010.log 2>&1


# #   -----------------------------black resnet50 om-fgsm--------------------------------

# # resnet18 cifar10 rmt 20210928000  cle+mix lr =0.01 beta(2,2)
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20211010.py run --mode defense --defense_mode rmt --beta_alpha 2 --attack_mode om-fgsm --attack_eps 0.3 --blackbox --exp_name resnet18-cifar10 --cla_model resnet18 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/resnet18-cifar10/20210908/00017-testacc-0.7779/train-cifar10-dataset/standard-trained-classifier-resnet18-on-clean-cifar10-finished.pkl --gen_model stylegan2ada --gen_network_pkl /home/maggie/mmat/result/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_dataset /home/maggie/mmat/result/project/stylegan2ada-cifar10/20210913/00000/project-cifar10-trainset --adv_dataset /home/maggie/mmat/result/attack/fgsm/resnet50-cifar10/20211010/00004-epsilon-0.3/attack-cifar10-dataset/latent-attack-samples --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/resnet18-cifar10-rmt/resnet18-cifar10-rmt-20211010.log 2>&1


# # resnet18 cifar10 rmt 20210928000  cle+mix lr =0.001 beta(2,2)
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20211010.py run --mode defense --defense_mode rmt --beta_alpha 2 --attack_mode om-fgsm --attack_eps 0.3 --blackbox --exp_name resnet18-cifar10 --cla_model resnet18 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/resnet18-cifar10/20210908/00017-testacc-0.7779/train-cifar10-dataset/standard-trained-classifier-resnet18-on-clean-cifar10-finished.pkl --gen_model stylegan2ada --gen_network_pkl /home/maggie/mmat/result/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_dataset /home/maggie/mmat/result/project/stylegan2ada-cifar10/20210913/00000/project-cifar10-trainset --adv_dataset /home/maggie/mmat/result/attack/fgsm/resnet50-cifar10/20211010/00004-epsilon-0.3/attack-cifar10-dataset/latent-attack-samples --batch_size 256 --epochs 40 --lr 0.001 >> /home/maggie/mmat/log/resnet18-cifar10-rmt/resnet18-cifar10-rmt-20211010.log 2>&1

# # resnet18 cifar10 rmt 20210928000  cle+mix lr =0.01 beta(0.5,0.5) 
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20211010.py run --mode defense --defense_mode rmt --beta_alpha 0.5 --attack_mode om-fgsm --attack_eps 0.3 --blackbox --exp_name resnet18-cifar10 --cla_model resnet18 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/resnet18-cifar10/20210908/00017-testacc-0.7779/train-cifar10-dataset/standard-trained-classifier-resnet18-on-clean-cifar10-finished.pkl --gen_model stylegan2ada --gen_network_pkl /home/maggie/mmat/result/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_dataset /home/maggie/mmat/result/project/stylegan2ada-cifar10/20210913/00000/project-cifar10-trainset --adv_dataset /home/maggie/mmat/result/attack/fgsm/resnet50-cifar10/20211010/00004-epsilon-0.3/attack-cifar10-dataset/latent-attack-samples --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/resnet18-cifar10-rmt/resnet18-cifar10-rmt-20211010.log 2>&1

# # resnet18 cifar10 rmt 20210928000  cle+mix lr =0.001 beta(0.5,0.5)
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20211010.py run --mode defense --defense_mode rmt --beta_alpha 0.5 --attack_mode om-fgsm --attack_eps 0.3 --blackbox --exp_name resnet18-cifar10 --cla_model resnet18 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/resnet18-cifar10/20210908/00017-testacc-0.7779/train-cifar10-dataset/standard-trained-classifier-resnet18-on-clean-cifar10-finished.pkl --gen_model stylegan2ada --gen_network_pkl /home/maggie/mmat/result/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_dataset /home/maggie/mmat/result/project/stylegan2ada-cifar10/20210913/00000/project-cifar10-trainset --adv_dataset /home/maggie/mmat/result/attack/fgsm/resnet50-cifar10/20211010/00004-epsilon-0.3/attack-cifar10-dataset/latent-attack-samples --batch_size 256 --epochs 40 --lr 0.001 >> /home/maggie/mmat/log/resnet18-cifar10-rmt/resnet18-cifar10-rmt-20211010.log 2>&1


# #   -----------------------------black vgg19 om-fgsm--------------------------------

# # resnet18 cifar10 rmt 20210928000  cle+mix lr =0.01 beta(2,2)
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20211010.py run --mode defense --defense_mode rmt --beta_alpha 2 --attack_mode om-fgsm --attack_eps 0.3 --blackbox --exp_name resnet18-cifar10 --cla_model resnet18 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/resnet18-cifar10/20210908/00017-testacc-0.7779/train-cifar10-dataset/standard-trained-classifier-resnet18-on-clean-cifar10-finished.pkl --gen_model stylegan2ada --gen_network_pkl /home/maggie/mmat/result/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_dataset /home/maggie/mmat/result/project/stylegan2ada-cifar10/20210913/00000/project-cifar10-trainset --adv_dataset /home/maggie/mmat/result/attack/fgsm/vgg19-cifar10/20211010/00004-epsilon-0.3/attack-cifar10-dataset/latent-attack-samples --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/resnet18-cifar10-rmt/resnet18-cifar10-rmt-20211010.log 2>&1


# # resnet18 cifar10 rmt 20210928000  cle+mix lr =0.001 beta(2,2)
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20211010.py run --mode defense --defense_mode rmt --beta_alpha 2 --attack_mode om-fgsm --attack_eps 0.3 --blackbox --exp_name resnet18-cifar10 --cla_model resnet18 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/resnet18-cifar10/20210908/00017-testacc-0.7779/train-cifar10-dataset/standard-trained-classifier-resnet18-on-clean-cifar10-finished.pkl --gen_model stylegan2ada --gen_network_pkl /home/maggie/mmat/result/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_dataset /home/maggie/mmat/result/project/stylegan2ada-cifar10/20210913/00000/project-cifar10-trainset --adv_dataset /home/maggie/mmat/result/attack/fgsm/vgg19-cifar10/20211010/00004-epsilon-0.3/attack-cifar10-dataset/latent-attack-samples --batch_size 256 --epochs 40 --lr 0.001 >> /home/maggie/mmat/log/resnet18-cifar10-rmt/resnet18-cifar10-rmt-20211010.log 2>&1

# # resnet18 cifar10 rmt 20210928000  cle+mix lr =0.01 beta(0.5,0.5) 
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20211010.py run --mode defense --defense_mode rmt --beta_alpha 0.5 --attack_mode om-fgsm --attack_eps 0.3 --blackbox --exp_name resnet18-cifar10 --cla_model resnet18 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/resnet18-cifar10/20210908/00017-testacc-0.7779/train-cifar10-dataset/standard-trained-classifier-resnet18-on-clean-cifar10-finished.pkl --gen_model stylegan2ada --gen_network_pkl /home/maggie/mmat/result/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_dataset /home/maggie/mmat/result/project/stylegan2ada-cifar10/20210913/00000/project-cifar10-trainset --adv_dataset /home/maggie/mmat/result/attack/fgsm/vgg19-cifar10/20211010/00004-epsilon-0.3/attack-cifar10-dataset/latent-attack-samples --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/resnet18-cifar10-rmt/resnet18-cifar10-rmt-20211010.log 2>&1

# # resnet18 cifar10 rmt 20210928000  cle+mix lr =0.001 beta(0.5,0.5)
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20211010.py run --mode defense --defense_mode rmt --beta_alpha 0.5 --attack_mode om-fgsm --attack_eps 0.3 --blackbox --exp_name resnet18-cifar10 --cla_model resnet18 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/resnet18-cifar10/20210908/00017-testacc-0.7779/train-cifar10-dataset/standard-trained-classifier-resnet18-on-clean-cifar10-finished.pkl --gen_model stylegan2ada --gen_network_pkl /home/maggie/mmat/result/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_dataset /home/maggie/mmat/result/project/stylegan2ada-cifar10/20210913/00000/project-cifar10-trainset --adv_dataset /home/maggie/mmat/result/attack/fgsm/vgg19-cifar10/20211010/00004-epsilon-0.3/attack-cifar10-dataset/latent-attack-samples --batch_size 256 --epochs 40 --lr 0.001 >> /home/maggie/mmat/log/resnet18-cifar10-rmt/resnet18-cifar10-rmt-20211010.log 2>&1

# #   -----------------------------black densenet169 om-fgsm--------------------------------

# # resnet18 cifar10 rmt 20210928000  cle+mix lr =0.01 beta(2,2)
# CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20211010.py run --mode defense --defense_mode rmt --beta_alpha 2 --attack_mode om-fgsm --attack_eps 0.3 --blackbox --exp_name resnet18-cifar10 --cla_model resnet18 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/resnet18-cifar10/20210908/00017-testacc-0.7779/train-cifar10-dataset/standard-trained-classifier-resnet18-on-clean-cifar10-finished.pkl --gen_model stylegan2ada --gen_network_pkl /home/maggie/mmat/result/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_dataset /home/maggie/mmat/result/project/stylegan2ada-cifar10/20210913/00000/project-cifar10-trainset --adv_dataset /home/maggie/mmat/result/attack/fgsm/densenet169-cifar10/20211010/00004-epsilon-0.3/attack-cifar10-dataset/latent-attack-samples --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/resnet18-cifar10-rmt/resnet18-cifar10-rmt-20211010.log 2>&1


# # resnet18 cifar10 rmt 20210928000  cle+mix lr =0.001 beta(2,2)
# CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20211010.py run --mode defense --defense_mode rmt --beta_alpha 2 --attack_mode om-fgsm --attack_eps 0.3 --blackbox --exp_name resnet18-cifar10 --cla_model resnet18 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/resnet18-cifar10/20210908/00017-testacc-0.7779/train-cifar10-dataset/standard-trained-classifier-resnet18-on-clean-cifar10-finished.pkl --gen_model stylegan2ada --gen_network_pkl /home/maggie/mmat/result/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_dataset /home/maggie/mmat/result/project/stylegan2ada-cifar10/20210913/00000/project-cifar10-trainset --adv_dataset /home/maggie/mmat/result/attack/fgsm/densenet169-cifar10/20211010/00004-epsilon-0.3/attack-cifar10-dataset/latent-attack-samples --batch_size 256 --epochs 40 --lr 0.001 >> /home/maggie/mmat/log/resnet18-cifar10-rmt/resnet18-cifar10-rmt-20211010.log 2>&1


# # resnet18 cifar10 rmt 20210928000  cle+mix lr =0.01 beta(0.5,0.5) 
# CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20211010.py run --mode defense --defense_mode rmt --beta_alpha 0.5 --attack_mode om-fgsm --attack_eps 0.3 --blackbox --exp_name resnet18-cifar10 --cla_model resnet18 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/resnet18-cifar10/20210908/00017-testacc-0.7779/train-cifar10-dataset/standard-trained-classifier-resnet18-on-clean-cifar10-finished.pkl --gen_model stylegan2ada --gen_network_pkl /home/maggie/mmat/result/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_dataset /home/maggie/mmat/result/project/stylegan2ada-cifar10/20210913/00000/project-cifar10-trainset --adv_dataset /home/maggie/mmat/result/attack/fgsm/densenet169-cifar10/20211010/00004-epsilon-0.3/attack-cifar10-dataset/latent-attack-samples --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/resnet18-cifar10-rmt/resnet18-cifar10-rmt-20211010.log 2>&1

# # resnet18 cifar10 rmt 20210928000  cle+mix lr =0.001 beta(0.5,0.5)
# CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20211010.py run --mode defense --defense_mode rmt --beta_alpha 0.5 --attack_mode om-fgsm --attack_eps 0.3 --blackbox --exp_name resnet18-cifar10 --cla_model resnet18 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/resnet18-cifar10/20210908/00017-testacc-0.7779/train-cifar10-dataset/standard-trained-classifier-resnet18-on-clean-cifar10-finished.pkl --gen_model stylegan2ada --gen_network_pkl /home/maggie/mmat/result/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_dataset /home/maggie/mmat/result/project/stylegan2ada-cifar10/20210913/00000/project-cifar10-trainset --adv_dataset /home/maggie/mmat/result/attack/fgsm/densenet169-cifar10/20211010/00004-epsilon-0.3/attack-cifar10-dataset/latent-attack-samples --batch_size 256 --epochs 40 --lr 0.001 >> /home/maggie/mmat/log/resnet18-cifar10-rmt/resnet18-cifar10-rmt-20211010.log 2>&1

# #   -----------------------------black googlenet om-fgsm--------------------------------

# # resnet18 cifar10 rmt 20210928000  cle+mix lr =0.01 beta(2,2)
# CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20211010.py run --mode defense --defense_mode rmt --beta_alpha 2 --attack_mode om-fgsm --attack_eps 0.3 --blackbox --exp_name resnet18-cifar10 --cla_model resnet18 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/resnet18-cifar10/20210908/00017-testacc-0.7779/train-cifar10-dataset/standard-trained-classifier-resnet18-on-clean-cifar10-finished.pkl --gen_model stylegan2ada --gen_network_pkl /home/maggie/mmat/result/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_dataset /home/maggie/mmat/result/project/stylegan2ada-cifar10/20210913/00000/project-cifar10-trainset --adv_dataset /home/maggie/mmat/result/attack/fgsm/googlenet-cifar10/20211010/00004-epsilon-0.3/attack-cifar10-dataset/latent-attack-samples --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/resnet18-cifar10-rmt/resnet18-cifar10-rmt-20211010.log 2>&1


# # resnet18 cifar10 rmt 20210928000  cle+mix lr =0.001 beta(2,2)
# CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20211010.py run --mode defense --defense_mode rmt --beta_alpha 2 --attack_mode om-fgsm --attack_eps 0.3 --blackbox --exp_name resnet18-cifar10 --cla_model resnet18 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/resnet18-cifar10/20210908/00017-testacc-0.7779/train-cifar10-dataset/standard-trained-classifier-resnet18-on-clean-cifar10-finished.pkl --gen_model stylegan2ada --gen_network_pkl /home/maggie/mmat/result/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_dataset /home/maggie/mmat/result/project/stylegan2ada-cifar10/20210913/00000/project-cifar10-trainset --adv_dataset /home/maggie/mmat/result/attack/fgsm/googlenet-cifar10/20211010/00004-epsilon-0.3/attack-cifar10-dataset/latent-attack-samples --batch_size 256 --epochs 40 --lr 0.001 >> /home/maggie/mmat/log/resnet18-cifar10-rmt/resnet18-cifar10-rmt-20211010.log 2>&1


# # resnet18 cifar10 rmt 20210928000  cle+mix lr =0.01 beta(0.5,0.5) 
# CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20211010.py run --mode defense --defense_mode rmt --beta_alpha 0.5 --attack_mode om-fgsm --attack_eps 0.3 --blackbox --exp_name resnet18-cifar10 --cla_model resnet18 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/resnet18-cifar10/20210908/00017-testacc-0.7779/train-cifar10-dataset/standard-trained-classifier-resnet18-on-clean-cifar10-finished.pkl --gen_model stylegan2ada --gen_network_pkl /home/maggie/mmat/result/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_dataset /home/maggie/mmat/result/project/stylegan2ada-cifar10/20210913/00000/project-cifar10-trainset --adv_dataset /home/maggie/mmat/result/attack/fgsm/googlenet-cifar10/20211010/00004-epsilon-0.3/attack-cifar10-dataset/latent-attack-samples --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/resnet18-cifar10-rmt/resnet18-cifar10-rmt-20211010.log 2>&1

# resnet18 cifar10 rmt 20210928000  cle+mix lr =0.001 beta(0.5,0.5)
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20211010.py run --mode defense --defense_mode rmt --beta_alpha 0.5 --attack_mode om-fgsm --attack_eps 0.3 --blackbox --exp_name resnet18-cifar10 --cla_model resnet18 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/resnet18-cifar10/20210908/00017-testacc-0.7779/train-cifar10-dataset/standard-trained-classifier-resnet18-on-clean-cifar10-finished.pkl --gen_model stylegan2ada --gen_network_pkl /home/maggie/mmat/result/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_dataset /home/maggie/mmat/result/project/stylegan2ada-cifar10/20210913/00000/project-cifar10-trainset --adv_dataset /home/maggie/mmat/result/attack/fgsm/googlenet-cifar10/20211010/00004-epsilon-0.3/attack-cifar10-dataset/latent-attack-samples --batch_size 256 --epochs 40 --lr 0.001 >> /home/maggie/mmat/log/resnet18-cifar10-rmt/resnet18-cifar10-rmt-20211010.log 2>&1

