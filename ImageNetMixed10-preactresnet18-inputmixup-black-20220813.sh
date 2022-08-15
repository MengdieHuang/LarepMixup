source ~/.bashrc
source /root/miniconda3/bin/activate mmat

# adv50000 cat cle50000
# PreActResNet18 inputmixup beta(1,1) against FGSM (eps=0.05)
python -u tasklauncher-20220815.py run --mode defense --defense_mode inputmixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode fgsm --attack_eps 0.02 --blackbox --exp_name preactresnet18-imagenetmixed10 --cla_model preactresnet18 --cla_network_pkl /root/autodl-tmp/maggie/result/train/cla-train/preactresnet18-imagenetmixed10/standard-trained-classifier-preactresnet18-on-clean-imagenetmixed10-epoch-0023-acc-90.47.pkl --dataset imagenetmixed10 --test_adv_dataset /root/autodl-tmp/maggie/result/attack/fgsm/preactresnet18-imagenetmixed10/20220813/00000-testacc-13.93/attack-imagenetmixed10-dataset/samples/test --batch_size 32 --epochs 40 --lr 0.01 >> /root/autodl-nas/maggie/mmat/log/ImagenetMixed10/defense/input-mixup-training/preactresnet18-imagenetmixed10-inputmixup-fgsm-20220815.log 2>&1




# #-------------以上要存储at训练好的模型-----------
# # #   使用PGD对抗训练后的 cla1
# # PreActResNet18 inputmixup beta(1,1) against pgd (eps=0.05)
# python -u tasklauncher-20220815.py run --mode defense --defense_mode inputmixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode pgd --attack_eps 0.02 --blackbox --exp_name preactresnet18-imagenetmixed10 --cla_model preactresnet18 --cla_network_pkl XXX-cla1-XXX --dataset imagenetmixed10 --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/preactresnet18-imagenetmixed10/20220624/00001-pgd-eps-0.05-acc-25.71/attack-cifar10-dataset/samples/test --batch_size 32 --epochs 40 --lr 0.01 >> /root/autodl-nas/maggie/mmat/log/ImagenetMixed10/defense/input-mixup-training/preactresnet18-imagenetmixed10-inputmixup-pgd-20220815.log 2>&1

# # PreActResNet18 inputmixup beta(1,1) against autoattack (eps=0.05)
# python -u tasklauncher-20220815.py run --mode defense --defense_mode inputmixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode autoattack --attack_eps 0.02 --blackbox --exp_name preactresnet18-imagenetmixed10 --cla_model preactresnet18 --cla_network_pkl XXX-cla1-XXX --dataset imagenetmixed10 --test_adv_dataset /home/data/maggie/result-newhome/attack/autoattack/preactresnet18-imagenetmixed10/20220712/00000-autoattack-eps-0.05-acc-5.27/attack-cifar10-dataset/samples/test --batch_size 32 --epochs 40 --lr 0.01 >> /root/autodl-nas/maggie/mmat/log/ImagenetMixed10/defense/input-mixup-training/preactresnet18-imagenetmixed10-inputmixup-autoattack-20220815.log 2>&1

# # PreActResNet18 inputmixup beta(1,1) against deepfool (eps=0.02)
# python -u tasklauncher-20220815.py run --mode defense --defense_mode inputmixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode deepfool --attack_eps 0.02 --blackbox --exp_name preactresnet18-imagenetmixed10 --cla_model preactresnet18 --cla_network_pkl XXX-cla1-XXX --dataset imagenetmixed10 --test_adv_dataset /home/data/maggie/result-newhome/attack/deepfool/preactresnet18-imagenetmixed10/20220707/00000-deepfool-eps-0.02-acc-12.27/attack-cifar10-dataset/samples/test --batch_size 32 --epochs 40 --lr 0.01 >> /root/autodl-nas/maggie/mmat/log/ImagenetMixed10/defense/input-mixup-training/preactresnet18-imagenetmixed10-inputmixup-deepfool-20220815.log 2>&1

# # PreActResNet18 inputmixup beta(1,1) against cw (eps=0.02)
# python -u tasklauncher-20220815.py run --mode defense --defense_mode inputmixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode cw --attack_eps 0.02 --blackbox --exp_name preactresnet18-imagenetmixed10 --cla_model preactresnet18 --cla_network_pkl XXX-cla1-XXX --dataset imagenetmixed10 --test_adv_dataset /home/data/maggie/result-newhome/attack/cw/preactresnet18-imagenetmixed10/20220704/00000-cw-confidence-0.0-acc-1.89/attack-cifar10-dataset/samples/test --batch_size 32 --epochs 40 --lr 0.01 >> /root/autodl-nas/maggie/mmat/log/ImagenetMixed10/defense/input-mixup-training/preactresnet18-imagenetmixed10-inputmixup-cw-20220815.log 2>&1

# # PreActResNet18 inputmixup beta(1,1) against om-fgsm (eps=0.05)
# python -u tasklauncher-20220815.py run --mode defense --defense_mode inputmixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode om-fgsm --attack_eps 0.02 --blackbox --exp_name preactresnet18-imagenetmixed10 --cla_model preactresnet18 --cla_network_pkl XXX-cla1-XXX --dataset imagenetmixed10 --test_adv_dataset /home/data/maggie/result-newhome/attack/fgsm/preactresnet18-imagenetmixed10/20220702/00001-omfgsm-eps-0.05-acc-49.23/attack-cifar10-dataset/latent-attack-samples/test --batch_size 32 --epochs 40 --lr 0.01 >> /root/autodl-nas/maggie/mmat/log/ImagenetMixed10/defense/input-mixup-training/preactresnet18-imagenetmixed10-inputmixup-om-fgsm-20220815.log 2>&1

# # PreActResNet18 inputmixup beta(1,1) against om-pgd (eps=0.05)
# python -u tasklauncher-20220815.py run --mode defense --defense_mode inputmixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode om-pgd --attack_eps 0.02 --blackbox --exp_name preactresnet18-imagenetmixed10 --cla_model preactresnet18 --cla_network_pkl XXX-cla1-XXX --dataset imagenetmixed10 --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/preactresnet18-imagenetmixed10/20220703/00001-ompgd-eps-0.05-acc-17.05/attack-cifar10-dataset/latent-attack-samples/test --batch_size 32 --epochs 40 --lr 0.01 >> /root/autodl-nas/maggie/mmat/log/ImagenetMixed10/defense/input-mixup-training/preactresnet18-imagenetmixed10-inputmixup-om-pgd-20220815.log 2>&1
