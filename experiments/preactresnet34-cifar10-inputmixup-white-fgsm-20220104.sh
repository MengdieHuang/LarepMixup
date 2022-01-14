source ~/.bashrc
source /home/xieyi/anaconda3/bin/activate mmat

#-----------------------FGSM-------------------------------
# preactresnet34 defense fgsm inputmixup beta(0.5,0.5)
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220104.py run --mode defense --defense_mode inputmixup --beta_alpha 0.5 --attack_mode fgsm --attack_eps 0.3 --whitebox --exp_name preactresnet34-cifar10 --cla_model preactresnet34 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/preactresnet34-cifar10/20210927/00002-testacc-83.57/train-cifar10-dataset/standard-trained-classifier-preactresnet34-on-clean-cifar10-epoch-0021.pkl --gen_model stylegan2ada --gen_network_pkl /home/maggie/mmat/result/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_dataset /home/maggie/mmat/result/project/stylegan2ada-cifar10/20210913/00000/project-cifar10-trainset --adv_dataset /home/maggie/mmat/result/attack/fgsm/preactresnet34-cifar10/20210927/00000-attackacc-18.04/attack-cifar10-dataset/samples --batch_size 256 --epochs 40 --lr 0.001 >> /home/maggie/mmat/log/preactresnet34-cifar10-inputmixup/preactresnet34-cifar10-inputmixup-20220104.log 2>&1

# preactresnet34 defense fgsm inputmixup beta(1,1)
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220104.py run --mode defense --defense_mode inputmixup --beta_alpha 1 --attack_mode fgsm --attack_eps 0.3 --whitebox --exp_name preactresnet34-cifar10 --cla_model preactresnet34 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/preactresnet34-cifar10/20210927/00002-testacc-83.57/train-cifar10-dataset/standard-trained-classifier-preactresnet34-on-clean-cifar10-epoch-0021.pkl --gen_model stylegan2ada --gen_network_pkl /home/maggie/mmat/result/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_dataset /home/maggie/mmat/result/project/stylegan2ada-cifar10/20210913/00000/project-cifar10-trainset --adv_dataset /home/maggie/mmat/result/attack/fgsm/preactresnet34-cifar10/20210927/00000-attackacc-18.04/attack-cifar10-dataset/samples --batch_size 256 --epochs 40 --lr 0.001 >> /home/maggie/mmat/log/preactresnet34-cifar10-inputmixup/preactresnet34-cifar10-inputmixup-20220104.log 2>&1

# preactresnet34 defense fgsm inputmixup beta(2,2)
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220104.py run --mode defense --defense_mode inputmixup --beta_alpha 2 --attack_mode fgsm --attack_eps 0.3 --whitebox --exp_name preactresnet34-cifar10 --cla_model preactresnet34 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/preactresnet34-cifar10/20210927/00002-testacc-83.57/train-cifar10-dataset/standard-trained-classifier-preactresnet34-on-clean-cifar10-epoch-0021.pkl --gen_model stylegan2ada --gen_network_pkl /home/maggie/mmat/result/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_dataset /home/maggie/mmat/result/project/stylegan2ada-cifar10/20210913/00000/project-cifar10-trainset --adv_dataset /home/maggie/mmat/result/attack/fgsm/preactresnet34-cifar10/20210927/00000-attackacc-18.04/attack-cifar10-dataset/samples --batch_size 256 --epochs 40 --lr 0.001 >> /home/maggie/mmat/log/preactresnet34-cifar10-inputmixup/preactresnet34-cifar10-inputmixup-20220104.log 2>&1

#-----------------------FGSM-2------------------------------
# preactresnet34 defense fgsm inputmixup beta(0.5,0.5)
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220104.py run --mode defense --defense_mode inputmixup --beta_alpha 0.5 --attack_mode fgsm --attack_eps 0.3 --whitebox --exp_name preactresnet34-cifar10 --cla_model preactresnet34 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/preactresnet34-cifar10/20210927/00002-testacc-83.57/train-cifar10-dataset/standard-trained-classifier-preactresnet34-on-clean-cifar10-epoch-0021.pkl --gen_model stylegan2ada --gen_network_pkl /home/maggie/mmat/result/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_dataset /home/maggie/mmat/result/project/stylegan2ada-cifar10/20210913/00000/project-cifar10-trainset --adv_dataset /home/maggie/mmat/result/attack/fgsm/preactresnet34-cifar10/20210927/00000-attackacc-18.04/attack-cifar10-dataset/samples --batch_size 256 --epochs 40 --lr 0.001 >> /home/maggie/mmat/log/preactresnet34-cifar10-inputmixup/preactresnet34-cifar10-inputmixup-20220104.log 2>&1

# preactresnet34 defense fgsm inputmixup beta(1,1)
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220104.py run --mode defense --defense_mode inputmixup --beta_alpha 1 --attack_mode fgsm --attack_eps 0.3 --whitebox --exp_name preactresnet34-cifar10 --cla_model preactresnet34 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/preactresnet34-cifar10/20210927/00002-testacc-83.57/train-cifar10-dataset/standard-trained-classifier-preactresnet34-on-clean-cifar10-epoch-0021.pkl --gen_model stylegan2ada --gen_network_pkl /home/maggie/mmat/result/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_dataset /home/maggie/mmat/result/project/stylegan2ada-cifar10/20210913/00000/project-cifar10-trainset --adv_dataset /home/maggie/mmat/result/attack/fgsm/preactresnet34-cifar10/20210927/00000-attackacc-18.04/attack-cifar10-dataset/samples --batch_size 256 --epochs 40 --lr 0.001 >> /home/maggie/mmat/log/preactresnet34-cifar10-inputmixup/preactresnet34-cifar10-inputmixup-20220104.log 2>&1

# preactresnet34 defense fgsm inputmixup beta(2,2)
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220104.py run --mode defense --defense_mode inputmixup --beta_alpha 2 --attack_mode fgsm --attack_eps 0.3 --whitebox --exp_name preactresnet34-cifar10 --cla_model preactresnet34 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/preactresnet34-cifar10/20210927/00002-testacc-83.57/train-cifar10-dataset/standard-trained-classifier-preactresnet34-on-clean-cifar10-epoch-0021.pkl --gen_model stylegan2ada --gen_network_pkl /home/maggie/mmat/result/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_dataset /home/maggie/mmat/result/project/stylegan2ada-cifar10/20210913/00000/project-cifar10-trainset --adv_dataset /home/maggie/mmat/result/attack/fgsm/preactresnet34-cifar10/20210927/00000-attackacc-18.04/attack-cifar10-dataset/samples --batch_size 256 --epochs 40 --lr 0.001 >> /home/maggie/mmat/log/preactresnet34-cifar10-inputmixup/preactresnet34-cifar10-inputmixup-20220104.log 2>&1

#-------------------------------------------------------------------
#-------------------OM FGSM-----------------------------------
# preactresnet34 defense omfgsm inputmixup beta(0.5,0.5)
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220104.py run --mode defense --defense_mode inputmixup --beta_alpha 0.5 --attack_mode om-fgsm --attack_eps 0.3 --blackbox --exp_name preactresnet34-cifar10 --cla_model preactresnet34 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/preactresnet34-cifar10/20210927/00002-testacc-83.57/train-cifar10-dataset/standard-trained-classifier-preactresnet34-on-clean-cifar10-epoch-0021.pkl --gen_model stylegan2ada --gen_network_pkl /home/maggie/mmat/result/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_dataset /home/maggie/mmat/result/project/stylegan2ada-cifar10/20210913/00000/project-cifar10-trainset --adv_dataset /home/maggie/mmat/result/attack/fgsm/preactresnet34-cifar10/20211010/00004-epsilon-0.3/attack-cifar10-dataset/latent-attack-samples --batch_size 256 --epochs 40 --lr 0.001 >> /home/maggie/mmat/log/preactresnet34-cifar10-inputmixup/preactresnet34-cifar10-inputmixup-omfgsm-20220104.log 2>&1

# preactresnet34 defense omfgsm inputmixup beta(1.0,1.0)
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220104.py run --mode defense --defense_mode inputmixup --beta_alpha 1 --attack_mode om-fgsm --attack_eps 0.3 --blackbox --exp_name preactresnet34-cifar10 --cla_model preactresnet34 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/preactresnet34-cifar10/20210927/00002-testacc-83.57/train-cifar10-dataset/standard-trained-classifier-preactresnet34-on-clean-cifar10-epoch-0021.pkl --gen_model stylegan2ada --gen_network_pkl /home/maggie/mmat/result/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_dataset /home/maggie/mmat/result/project/stylegan2ada-cifar10/20210913/00000/project-cifar10-trainset --adv_dataset /home/maggie/mmat/result/attack/fgsm/preactresnet34-cifar10/20211010/00004-epsilon-0.3/attack-cifar10-dataset/latent-attack-samples --batch_size 256 --epochs 40 --lr 0.001 >> /home/maggie/mmat/log/preactresnet34-cifar10-inputmixup/preactresnet34-cifar10-inputmixup-omfgsm-20220104.log 2>&1

# preactresnet34 defense omfgsm inputmixup beta(2.0,2.0)
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220104.py run --mode defense --defense_mode inputmixup --beta_alpha 2 --attack_mode om-fgsm --attack_eps 0.3 --blackbox --exp_name preactresnet34-cifar10 --cla_model preactresnet34 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/preactresnet34-cifar10/20210927/00002-testacc-83.57/train-cifar10-dataset/standard-trained-classifier-preactresnet34-on-clean-cifar10-epoch-0021.pkl --gen_model stylegan2ada --gen_network_pkl /home/maggie/mmat/result/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_dataset /home/maggie/mmat/result/project/stylegan2ada-cifar10/20210913/00000/project-cifar10-trainset --adv_dataset /home/maggie/mmat/result/attack/fgsm/preactresnet34-cifar10/20211010/00004-epsilon-0.3/attack-cifar10-dataset/latent-attack-samples --batch_size 256 --epochs 40 --lr 0.001 >> /home/maggie/mmat/log/preactresnet34-cifar10-inputmixup/preactresnet34-cifar10-inputmixup-omfgsm-20220104.log 2>&1

#-------------------OM FGSM--2---------------------------------
# preactresnet34 defense omfgsm inputmixup beta(0.5,0.5)
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220104.py run --mode defense --defense_mode inputmixup --beta_alpha 0.5 --attack_mode om-fgsm --attack_eps 0.3 --blackbox --exp_name preactresnet34-cifar10 --cla_model preactresnet34 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/preactresnet34-cifar10/20210927/00002-testacc-83.57/train-cifar10-dataset/standard-trained-classifier-preactresnet34-on-clean-cifar10-epoch-0021.pkl --gen_model stylegan2ada --gen_network_pkl /home/maggie/mmat/result/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_dataset /home/maggie/mmat/result/project/stylegan2ada-cifar10/20210913/00000/project-cifar10-trainset --adv_dataset /home/maggie/mmat/result/attack/fgsm/preactresnet34-cifar10/20211010/00004-epsilon-0.3/attack-cifar10-dataset/latent-attack-samples --batch_size 256 --epochs 40 --lr 0.001 >> /home/maggie/mmat/log/preactresnet34-cifar10-inputmixup/preactresnet34-cifar10-inputmixup-omfgsm-20220104.log 2>&1

# preactresnet34 defense omfgsm inputmixup beta(1.0,1.0)
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220104.py run --mode defense --defense_mode inputmixup --beta_alpha 1 --attack_mode om-fgsm --attack_eps 0.3 --blackbox --exp_name preactresnet34-cifar10 --cla_model preactresnet34 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/preactresnet34-cifar10/20210927/00002-testacc-83.57/train-cifar10-dataset/standard-trained-classifier-preactresnet34-on-clean-cifar10-epoch-0021.pkl --gen_model stylegan2ada --gen_network_pkl /home/maggie/mmat/result/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_dataset /home/maggie/mmat/result/project/stylegan2ada-cifar10/20210913/00000/project-cifar10-trainset --adv_dataset /home/maggie/mmat/result/attack/fgsm/preactresnet34-cifar10/20211010/00004-epsilon-0.3/attack-cifar10-dataset/latent-attack-samples --batch_size 256 --epochs 40 --lr 0.001 >> /home/maggie/mmat/log/preactresnet34-cifar10-inputmixup/preactresnet34-cifar10-inputmixup-omfgsm-20220104.log 2>&1

# preactresnet34 defense omfgsm inputmixup beta(2.0,2.0)
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220104.py run --mode defense --defense_mode inputmixup --beta_alpha 2 --attack_mode om-fgsm --attack_eps 0.3 --blackbox --exp_name preactresnet34-cifar10 --cla_model preactresnet34 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/preactresnet34-cifar10/20210927/00002-testacc-83.57/train-cifar10-dataset/standard-trained-classifier-preactresnet34-on-clean-cifar10-epoch-0021.pkl --gen_model stylegan2ada --gen_network_pkl /home/maggie/mmat/result/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_dataset /home/maggie/mmat/result/project/stylegan2ada-cifar10/20210913/00000/project-cifar10-trainset --adv_dataset /home/maggie/mmat/result/attack/fgsm/preactresnet34-cifar10/20211010/00004-epsilon-0.3/attack-cifar10-dataset/latent-attack-samples --batch_size 256 --epochs 40 --lr 0.001 >> /home/maggie/mmat/log/preactresnet34-cifar10-inputmixup/preactresnet34-cifar10-inputmixup-omfgsm-20220104.log 2>&1