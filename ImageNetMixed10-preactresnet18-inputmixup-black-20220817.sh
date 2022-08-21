source ~/.bashrc
source /root/miniconda3/bin/activate mmat

# adv50000 cat cle50000
# # PreActResNet18 inputmixup beta(1,1) against FGSM (eps=0.02)
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220818.py run --save_path /root/autodl-tmp/maggie/result --mode defense --defense_mode inputmixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode fgsm --attack_eps 0.02 --blackbox --exp_name preactresnet18-imagenetmixed10 --cla_model preactresnet18 --cla_network_pkl /root/autodl-tmp/maggie/result/train/cla-train/preactresnet18-imagenetmixed10/standard-trained-classifier-preactresnet18-on-clean-imagenetmixed10-epoch-0023-acc-90.47.pkl --dataset imagenetmixed10 --test_adv_dataset /root/autodl-tmp/maggie/result/attack/fgsm/preactresnet18-imagenetmixed10/20220813/00000-testacc-13.93/attack-imagenetmixed10-dataset/samples/test --batch_size 32 --epochs 40 --lr 0.01 >> /root/autodl-nas/maggie/mmat/log/ImagenetMixed10/defense/input-mixup-training/preactresnet18-imagenetmixed10-inputmixup-fgsm-20220818.log 2>&1

# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220818.py run --save_path /root/autodl-tmp/maggie/result --mode defense --defense_mode inputmixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode fgsm --attack_eps 0.02 --blackbox --exp_name preactresnet18-imagenetmixed10 --cla_model preactresnet18 --cla_network_pkl /root/autodl-tmp/maggie/result/train/cla-train/preactresnet18-imagenetmixed10/standard-trained-classifier-preactresnet18-on-clean-imagenetmixed10-epoch-0023-acc-90.47.pkl --dataset imagenetmixed10 --test_adv_dataset /root/autodl-tmp/maggie/result/attack/fgsm/preactresnet18-imagenetmixed10/20220813/00000-testacc-13.93/attack-imagenetmixed10-dataset/samples/test --batch_size 32 --epochs 40 --lr 0.001 >> /root/autodl-nas/maggie/mmat/log/ImagenetMixed10/defense/input-mixup-training/preactresnet18-imagenetmixed10-inputmixup-fgsm-20220818.log 2>&1

# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220818.py run --save_path /root/autodl-tmp/maggie/result --mode defense --defense_mode inputmixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode fgsm --attack_eps 0.02 --blackbox --exp_name preactresnet18-imagenetmixed10 --cla_model preactresnet18 --cla_network_pkl /root/autodl-tmp/maggie/result/train/cla-train/preactresnet18-imagenetmixed10/standard-trained-classifier-preactresnet18-on-clean-imagenetmixed10-epoch-0023-acc-90.47.pkl --dataset imagenetmixed10 --test_adv_dataset /root/autodl-tmp/maggie/result/attack/fgsm/preactresnet18-imagenetmixed10/20220813/00000-testacc-13.93/attack-imagenetmixed10-dataset/samples/test --batch_size 32 --epochs 40 --lr 0.01 >> /root/autodl-nas/maggie/mmat/log/ImagenetMixed10/defense/input-mixup-training/preactresnet18-imagenetmixed10-inputmixup-fgsm-20220818.log 2>&1

# python -u tasklauncher-20220818.py run --save_path /root/autodl-tmp/maggie/result --mode defense --defense_mode inputmixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode fgsm --attack_eps 0.02 --blackbox --exp_name preactresnet18-imagenetmixed10 --cla_model preactresnet18 --cla_network_pkl /root/autodl-tmp/maggie/result/train/cla-train/preactresnet18-imagenetmixed10/standard-trained-classifier-preactresnet18-on-clean-imagenetmixed10-epoch-0023-acc-90.47.pkl --dataset imagenetmixed10 --test_adv_dataset /root/autodl-tmp/maggie/result/attack/fgsm/preactresnet18-imagenetmixed10/20220813/00000-testacc-13.93/attack-imagenetmixed10-dataset/samples/test --batch_size 32 --epochs 40 --lr 0.01 >> /root/autodl-nas/maggie/mmat/log/ImagenetMixed10/defense/input-mixup-training/preactresnet18-imagenetmixed10-inputmixup-fgsm-20220818.log 2>&1

# python -u tasklauncher-20220818.py run --save_path /root/autodl-tmp/maggie/result --mode defense --defense_mode inputmixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode fgsm --attack_eps 0.02 --blackbox --exp_name preactresnet18-imagenetmixed10 --cla_model preactresnet18 --cla_network_pkl /root/autodl-tmp/maggie/result/train/cla-train/preactresnet18-imagenetmixed10/standard-trained-classifier-preactresnet18-on-clean-imagenetmixed10-epoch-0023-acc-90.47.pkl --dataset imagenetmixed10 --test_adv_dataset /root/autodl-tmp/maggie/result/attack/fgsm/preactresnet18-imagenetmixed10/20220813/00000-testacc-13.93/attack-imagenetmixed10-dataset/samples/test --batch_size 32 --epochs 40 --lr 0.01 >> /root/autodl-nas/maggie/mmat/log/ImagenetMixed10/defense/input-mixup-training/preactresnet18-imagenetmixed10-inputmixup-fgsm-20220818.log 2>&1

# python -u tasklauncher-20220818.py run --save_path /root/autodl-tmp/maggie/result --mode defense --defense_mode inputmixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode fgsm --attack_eps 0.02 --blackbox --exp_name preactresnet18-imagenetmixed10 --cla_model preactresnet18 --cla_network_pkl /root/autodl-tmp/maggie/result/train/cla-train/preactresnet18-imagenetmixed10/standard-trained-classifier-preactresnet18-on-clean-imagenetmixed10-epoch-0023-acc-90.47.pkl --dataset imagenetmixed10 --test_adv_dataset /root/autodl-tmp/maggie/result/attack/fgsm/preactresnet18-imagenetmixed10/20220813/00000-testacc-13.93/attack-imagenetmixed10-dataset/samples/test --batch_size 32 --epochs 40 --lr 0.01 >> /root/autodl-nas/maggie/mmat/log/ImagenetMixed10/defense/input-mixup-training/preactresnet18-imagenetmixed10-inputmixup-fgsm-20220818.log 2>&1

# #-------------以上要存储at训练好的模型-----------
# # #   使用PGD对抗训练后的 cla1
# # PreActResNet18 inputmixup beta(1,1) against pgd (eps=0.02)
# python -u tasklauncher-20220818.py run --save_path /root/autodl-tmp/maggie/result --mode defense --defense_mode inputmixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode pgd --attack_eps 0.02 --blackbox --exp_name preactresnet18-imagenetmixed10 --cla_model preactresnet18 --cla_network_pkl XXX-cla1-XXX --dataset imagenetmixed10 --test_adv_dataset /root/autodl-tmp/maggie/result/attack/pgd/preactresnet18-imagenetmixed10/20220812/00000-testacc-2.03/attack-imagenetmixed10-dataset/samples/test --batch_size 32 --epochs 40 --lr 0.01 >> /root/autodl-nas/maggie/mmat/log/ImagenetMixed10/defense/input-mixup-training/preactresnet18-imagenetmixed10-inputmixup-pgd-20220818.log 2>&1

# # PreActResNet18 inputmixup beta(1,1) against autoattack (eps=0.02)
# python -u tasklauncher-20220818.py run --save_path /root/autodl-tmp/maggie/result --mode defense --defense_mode inputmixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode autoattack --attack_eps 0.02 --blackbox --exp_name preactresnet18-imagenetmixed10 --cla_model preactresnet18 --cla_network_pkl XXX-cla1-XXX --dataset imagenetmixed10 --test_adv_dataset /root/autodl-tmp/maggie/result/attack/autoattack/preactresnet18-imagenetmixed10/20220813/00000-testacc-0.00/attack-imagenetmixed10-dataset/samples/test --batch_size 32 --epochs 40 --lr 0.01 >> /root/autodl-nas/maggie/mmat/log/ImagenetMixed10/defense/input-mixup-training/preactresnet18-imagenetmixed10-inputmixup-autoattack-20220818.log 2>&1

# # PreActResNet18 inputmixup beta(1,1) against deepfool (eps=0.02)
# python -u tasklauncher-20220818.py run --save_path /root/autodl-tmp/maggie/result --mode defense --defense_mode inputmixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode deepfool --attack_eps 0.02 --blackbox --exp_name preactresnet18-imagenetmixed10 --cla_model preactresnet18 --cla_network_pkl XXX-cla1-XXX --dataset imagenetmixed10 --test_adv_dataset /root/autodl-tmp/maggie/result/attack/deepfool/preactresnet18-imagenetmixed10/20220813/00000-testacc-8.87/attack-imagenetmixed10-dataset/samples/test --batch_size 32 --epochs 40 --lr 0.01 >> /root/autodl-nas/maggie/mmat/log/ImagenetMixed10/defense/input-mixup-training/preactresnet18-imagenetmixed10-inputmixup-deepfool-20220818.log 2>&1

# # PreActResNet18 inputmixup beta(1,1) against cw (eps=0.02)
# python -u tasklauncher-20220818.py run --save_path /root/autodl-tmp/maggie/result --mode defense --defense_mode inputmixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode cw --attack_eps 0.02 --blackbox --exp_name preactresnet18-imagenetmixed10 --cla_model preactresnet18 --cla_network_pkl XXX-cla1-XXX --dataset imagenetmixed10 --test_adv_dataset /root/autodl-tmp/maggie/result/attack/cw/preactresnet18-imagenetmixed10/20220813/00000-testacc-0.10/attack-imagenetmixed10-dataset/samples/test --batch_size 32 --epochs 40 --lr 0.01 >> /root/autodl-nas/maggie/mmat/log/ImagenetMixed10/defense/input-mixup-training/preactresnet18-imagenetmixed10-inputmixup-cw-20220818.log 2>&1

# # PreActResNet18 inputmixup beta(1,1) against om-fgsm (eps=0.02)
# python -u tasklauncher-20220818.py run --save_path /root/autodl-tmp/maggie/result --mode defense --defense_mode inputmixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode om-fgsm --attack_eps 0.02 --blackbox --exp_name preactresnet18-imagenetmixed10 --cla_model preactresnet18 --cla_network_pkl XXX-cla1-XXX --dataset imagenetmixed10 --test_adv_dataset /root/autodl-tmp/maggie/result/attack/fgsm/preactresnet18-imagenetmixed10/20220813/00003-testacc-26.90/attack-imagenetmixed10-dataset/latent-attack-samples/test --batch_size 32 --epochs 40 --lr 0.01 >> /root/autodl-nas/maggie/mmat/log/ImagenetMixed10/defense/input-mixup-training/preactresnet18-imagenetmixed10-inputmixup-om-fgsm-20220818.log 2>&1

# # PreActResNet18 inputmixup beta(1,1) against om-pgd (eps=0.02)
# python -u tasklauncher-20220818.py run --save_path /root/autodl-tmp/maggie/result --mode defense --defense_mode inputmixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode om-pgd --attack_eps 0.02 --blackbox --exp_name preactresnet18-imagenetmixed10 --cla_model preactresnet18 --cla_network_pkl XXX-cla1-XXX --dataset imagenetmixed10 --test_adv_dataset /root/autodl-tmp/maggie/result/attack/pgd/preactresnet18-imagenetmixed10/20220813/00003-testacc-20.43/attack-imagenetmixed10-dataset/latent-attack-samples/test --batch_size 32 --epochs 40 --lr 0.01 >> /root/autodl-nas/maggie/mmat/log/ImagenetMixed10/defense/input-mixup-training/preactresnet18-imagenetmixed10-inputmixup-om-pgd-20220818.log 2>&1

# # #   使用PGD对抗训练后的 cla2
# # PreActResNet18 inputmixup beta(1,1) against pgd (eps=0.02)
# python -u tasklauncher-20220818.py run --save_path /root/autodl-tmp/maggie/result --mode defense --defense_mode inputmixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode pgd --attack_eps 0.02 --blackbox --exp_name preactresnet18-imagenetmixed10 --cla_model preactresnet18 --cla_network_pkl XXX-cla2-XXX --dataset imagenetmixed10 --test_adv_dataset /root/autodl-tmp/maggie/result/attack/pgd/preactresnet18-imagenetmixed10/20220812/00000-testacc-2.03/attack-imagenetmixed10-dataset/samples/test --batch_size 32 --epochs 40 --lr 0.01 >> /root/autodl-nas/maggie/mmat/log/ImagenetMixed10/defense/input-mixup-training/preactresnet18-imagenetmixed10-inputmixup-pgd-20220818.log 2>&1

# # PreActResNet18 inputmixup beta(1,1) against autoattack (eps=0.02)
# python -u tasklauncher-20220818.py run --save_path /root/autodl-tmp/maggie/result --mode defense --defense_mode inputmixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode autoattack --attack_eps 0.02 --blackbox --exp_name preactresnet18-imagenetmixed10 --cla_model preactresnet18 --cla_network_pkl XXX-cla2-XXX --dataset imagenetmixed10 --test_adv_dataset /root/autodl-tmp/maggie/result/attack/autoattack/preactresnet18-imagenetmixed10/20220813/00000-testacc-0.00/attack-imagenetmixed10-dataset/samples/test --batch_size 32 --epochs 40 --lr 0.01 >> /root/autodl-nas/maggie/mmat/log/ImagenetMixed10/defense/input-mixup-training/preactresnet18-imagenetmixed10-inputmixup-autoattack-20220818.log 2>&1

# # PreActResNet18 inputmixup beta(1,1) against deepfool (eps=0.02)
# python -u tasklauncher-20220818.py run --save_path /root/autodl-tmp/maggie/result --mode defense --defense_mode inputmixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode deepfool --attack_eps 0.02 --blackbox --exp_name preactresnet18-imagenetmixed10 --cla_model preactresnet18 --cla_network_pkl XXX-cla2-XXX --dataset imagenetmixed10 --test_adv_dataset /root/autodl-tmp/maggie/result/attack/deepfool/preactresnet18-imagenetmixed10/20220813/00000-testacc-8.87/attack-imagenetmixed10-dataset/samples/test --batch_size 32 --epochs 40 --lr 0.01 >> /root/autodl-nas/maggie/mmat/log/ImagenetMixed10/defense/input-mixup-training/preactresnet18-imagenetmixed10-inputmixup-deepfool-20220818.log 2>&1

# # PreActResNet18 inputmixup beta(1,1) against cw (eps=0.02)
# python -u tasklauncher-20220818.py run --save_path /root/autodl-tmp/maggie/result --mode defense --defense_mode inputmixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode cw --attack_eps 0.02 --blackbox --exp_name preactresnet18-imagenetmixed10 --cla_model preactresnet18 --cla_network_pkl XXX-cla2-XXX --dataset imagenetmixed10 --test_adv_dataset /root/autodl-tmp/maggie/result/attack/cw/preactresnet18-imagenetmixed10/20220813/00000-testacc-0.10/attack-imagenetmixed10-dataset/samples/test --batch_size 32 --epochs 40 --lr 0.01 >> /root/autodl-nas/maggie/mmat/log/ImagenetMixed10/defense/input-mixup-training/preactresnet18-imagenetmixed10-inputmixup-cw-20220818.log 2>&1

# # PreActResNet18 inputmixup beta(1,1) against om-fgsm (eps=0.02)
# python -u tasklauncher-20220818.py run --save_path /root/autodl-tmp/maggie/result --mode defense --defense_mode inputmixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode om-fgsm --attack_eps 0.02 --blackbox --exp_name preactresnet18-imagenetmixed10 --cla_model preactresnet18 --cla_network_pkl XXX-cla2-XXX --dataset imagenetmixed10 --test_adv_dataset /root/autodl-tmp/maggie/result/attack/fgsm/preactresnet18-imagenetmixed10/20220813/00003-testacc-26.90/attack-imagenetmixed10-dataset/latent-attack-samples/test --batch_size 32 --epochs 40 --lr 0.01 >> /root/autodl-nas/maggie/mmat/log/ImagenetMixed10/defense/input-mixup-training/preactresnet18-imagenetmixed10-inputmixup-om-fgsm-20220818.log 2>&1

# # PreActResNet18 inputmixup beta(1,1) against om-pgd (eps=0.02)
# python -u tasklauncher-20220818.py run --save_path /root/autodl-tmp/maggie/result --mode defense --defense_mode inputmixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode om-pgd --attack_eps 0.02 --blackbox --exp_name preactresnet18-imagenetmixed10 --cla_model preactresnet18 --cla_network_pkl XXX-cla2-XXX --dataset imagenetmixed10 --test_adv_dataset /root/autodl-tmp/maggie/result/attack/pgd/preactresnet18-imagenetmixed10/20220813/00003-testacc-20.43/attack-imagenetmixed10-dataset/latent-attack-samples/test --batch_size 32 --epochs 40 --lr 0.01 >> /root/autodl-nas/maggie/mmat/log/ImagenetMixed10/defense/input-mixup-training/preactresnet18-imagenetmixed10-inputmixup-om-pgd-20220818.log 2>&1

# # #   使用PGD对抗训练后的 cla3
# # PreActResNet18 inputmixup beta(1,1) against pgd (eps=0.02)
# python -u tasklauncher-20220818.py run --save_path /root/autodl-tmp/maggie/result --mode defense --defense_mode inputmixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode pgd --attack_eps 0.02 --blackbox --exp_name preactresnet18-imagenetmixed10 --cla_model preactresnet18 --cla_network_pkl XXX-cla3-XXX --dataset imagenetmixed10 --test_adv_dataset /root/autodl-tmp/maggie/result/attack/pgd/preactresnet18-imagenetmixed10/20220812/00000-testacc-2.03/attack-imagenetmixed10-dataset/samples/test --batch_size 32 --epochs 40 --lr 0.01 >> /root/autodl-nas/maggie/mmat/log/ImagenetMixed10/defense/input-mixup-training/preactresnet18-imagenetmixed10-inputmixup-pgd-20220818.log 2>&1

# # PreActResNet18 inputmixup beta(1,1) against autoattack (eps=0.02)
# python -u tasklauncher-20220818.py run --save_path /root/autodl-tmp/maggie/result --mode defense --defense_mode inputmixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode autoattack --attack_eps 0.02 --blackbox --exp_name preactresnet18-imagenetmixed10 --cla_model preactresnet18 --cla_network_pkl XXX-cla3-XXX --dataset imagenetmixed10 --test_adv_dataset /root/autodl-tmp/maggie/result/attack/autoattack/preactresnet18-imagenetmixed10/20220813/00000-testacc-0.00/attack-imagenetmixed10-dataset/samples/test --batch_size 32 --epochs 40 --lr 0.01 >> /root/autodl-nas/maggie/mmat/log/ImagenetMixed10/defense/input-mixup-training/preactresnet18-imagenetmixed10-inputmixup-autoattack-20220818.log 2>&1

# # PreActResNet18 inputmixup beta(1,1) against deepfool (eps=0.02)
# python -u tasklauncher-20220818.py run --save_path /root/autodl-tmp/maggie/result --mode defense --defense_mode inputmixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode deepfool --attack_eps 0.02 --blackbox --exp_name preactresnet18-imagenetmixed10 --cla_model preactresnet18 --cla_network_pkl XXX-cla3-XXX --dataset imagenetmixed10 --test_adv_dataset /root/autodl-tmp/maggie/result/attack/deepfool/preactresnet18-imagenetmixed10/20220813/00000-testacc-8.87/attack-imagenetmixed10-dataset/samples/test --batch_size 32 --epochs 40 --lr 0.01 >> /root/autodl-nas/maggie/mmat/log/ImagenetMixed10/defense/input-mixup-training/preactresnet18-imagenetmixed10-inputmixup-deepfool-20220818.log 2>&1

# # PreActResNet18 inputmixup beta(1,1) against cw (eps=0.02)
# python -u tasklauncher-20220818.py run --save_path /root/autodl-tmp/maggie/result --mode defense --defense_mode inputmixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode cw --attack_eps 0.02 --blackbox --exp_name preactresnet18-imagenetmixed10 --cla_model preactresnet18 --cla_network_pkl XXX-cla3-XXX --dataset imagenetmixed10 --test_adv_dataset /root/autodl-tmp/maggie/result/attack/cw/preactresnet18-imagenetmixed10/20220813/00000-testacc-0.10/attack-imagenetmixed10-dataset/samples/test --batch_size 32 --epochs 40 --lr 0.01 >> /root/autodl-nas/maggie/mmat/log/ImagenetMixed10/defense/input-mixup-training/preactresnet18-imagenetmixed10-inputmixup-cw-20220818.log 2>&1

# # PreActResNet18 inputmixup beta(1,1) against om-fgsm (eps=0.02)
# python -u tasklauncher-20220818.py run --save_path /root/autodl-tmp/maggie/result --mode defense --defense_mode inputmixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode om-fgsm --attack_eps 0.02 --blackbox --exp_name preactresnet18-imagenetmixed10 --cla_model preactresnet18 --cla_network_pkl XXX-cla3-XXX --dataset imagenetmixed10 --test_adv_dataset /root/autodl-tmp/maggie/result/attack/fgsm/preactresnet18-imagenetmixed10/20220813/00003-testacc-26.90/attack-imagenetmixed10-dataset/latent-attack-samples/test --batch_size 32 --epochs 40 --lr 0.01 >> /root/autodl-nas/maggie/mmat/log/ImagenetMixed10/defense/input-mixup-training/preactresnet18-imagenetmixed10-inputmixup-om-fgsm-20220818.log 2>&1

# # PreActResNet18 inputmixup beta(1,1) against om-pgd (eps=0.02)
# python -u tasklauncher-20220818.py run --save_path /root/autodl-tmp/maggie/result --mode defense --defense_mode inputmixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode om-pgd --attack_eps 0.02 --blackbox --exp_name preactresnet18-imagenetmixed10 --cla_model preactresnet18 --cla_network_pkl XXX-cla3-XXX --dataset imagenetmixed10 --test_adv_dataset /root/autodl-tmp/maggie/result/attack/pgd/preactresnet18-imagenetmixed10/20220813/00003-testacc-20.43/attack-imagenetmixed10-dataset/latent-attack-samples/test --batch_size 32 --epochs 40 --lr 0.01 >> /root/autodl-nas/maggie/mmat/log/ImagenetMixed10/defense/input-mixup-training/preactresnet18-imagenetmixed10-inputmixup-om-pgd-20220818.log 2>&1


# # #   使用PGD对抗训练后的 cla4
# # PreActResNet18 inputmixup beta(1,1) against pgd (eps=0.02)
# python -u tasklauncher-20220818.py run --save_path /root/autodl-tmp/maggie/result --mode defense --defense_mode inputmixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode pgd --attack_eps 0.02 --blackbox --exp_name preactresnet18-imagenetmixed10 --cla_model preactresnet18 --cla_network_pkl XXX-cla4-XXX --dataset imagenetmixed10 --test_adv_dataset /root/autodl-tmp/maggie/result/attack/pgd/preactresnet18-imagenetmixed10/20220812/00000-testacc-2.03/attack-imagenetmixed10-dataset/samples/test --batch_size 32 --epochs 40 --lr 0.01 >> /root/autodl-nas/maggie/mmat/log/ImagenetMixed10/defense/input-mixup-training/preactresnet18-imagenetmixed10-inputmixup-pgd-20220818.log 2>&1

# # PreActResNet18 inputmixup beta(1,1) against autoattack (eps=0.02)
# python -u tasklauncher-20220818.py run --save_path /root/autodl-tmp/maggie/result --mode defense --defense_mode inputmixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode autoattack --attack_eps 0.02 --blackbox --exp_name preactresnet18-imagenetmixed10 --cla_model preactresnet18 --cla_network_pkl XXX-cla4-XXX --dataset imagenetmixed10 --test_adv_dataset /root/autodl-tmp/maggie/result/attack/autoattack/preactresnet18-imagenetmixed10/20220813/00000-testacc-0.00/attack-imagenetmixed10-dataset/samples/test --batch_size 32 --epochs 40 --lr 0.01 >> /root/autodl-nas/maggie/mmat/log/ImagenetMixed10/defense/input-mixup-training/preactresnet18-imagenetmixed10-inputmixup-autoattack-20220818.log 2>&1

# # PreActResNet18 inputmixup beta(1,1) against deepfool (eps=0.02)
# python -u tasklauncher-20220818.py run --save_path /root/autodl-tmp/maggie/result --mode defense --defense_mode inputmixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode deepfool --attack_eps 0.02 --blackbox --exp_name preactresnet18-imagenetmixed10 --cla_model preactresnet18 --cla_network_pkl XXX-cla4-XXX --dataset imagenetmixed10 --test_adv_dataset /root/autodl-tmp/maggie/result/attack/deepfool/preactresnet18-imagenetmixed10/20220813/00000-testacc-8.87/attack-imagenetmixed10-dataset/samples/test --batch_size 32 --epochs 40 --lr 0.01 >> /root/autodl-nas/maggie/mmat/log/ImagenetMixed10/defense/input-mixup-training/preactresnet18-imagenetmixed10-inputmixup-deepfool-20220818.log 2>&1

# # PreActResNet18 inputmixup beta(1,1) against cw (eps=0.02)
# python -u tasklauncher-20220818.py run --save_path /root/autodl-tmp/maggie/result --mode defense --defense_mode inputmixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode cw --attack_eps 0.02 --blackbox --exp_name preactresnet18-imagenetmixed10 --cla_model preactresnet18 --cla_network_pkl XXX-cla4-XXX --dataset imagenetmixed10 --test_adv_dataset /root/autodl-tmp/maggie/result/attack/cw/preactresnet18-imagenetmixed10/20220813/00000-testacc-0.10/attack-imagenetmixed10-dataset/samples/test --batch_size 32 --epochs 40 --lr 0.01 >> /root/autodl-nas/maggie/mmat/log/ImagenetMixed10/defense/input-mixup-training/preactresnet18-imagenetmixed10-inputmixup-cw-20220818.log 2>&1

# # PreActResNet18 inputmixup beta(1,1) against om-fgsm (eps=0.02)
# python -u tasklauncher-20220818.py run --save_path /root/autodl-tmp/maggie/result --mode defense --defense_mode inputmixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode om-fgsm --attack_eps 0.02 --blackbox --exp_name preactresnet18-imagenetmixed10 --cla_model preactresnet18 --cla_network_pkl XXX-cla4-XXX --dataset imagenetmixed10 --test_adv_dataset /root/autodl-tmp/maggie/result/attack/fgsm/preactresnet18-imagenetmixed10/20220813/00003-testacc-26.90/attack-imagenetmixed10-dataset/latent-attack-samples/test --batch_size 32 --epochs 40 --lr 0.01 >> /root/autodl-nas/maggie/mmat/log/ImagenetMixed10/defense/input-mixup-training/preactresnet18-imagenetmixed10-inputmixup-om-fgsm-20220818.log 2>&1

# # PreActResNet18 inputmixup beta(1,1) against om-pgd (eps=0.02)
# python -u tasklauncher-20220818.py run --save_path /root/autodl-tmp/maggie/result --mode defense --defense_mode inputmixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode om-pgd --attack_eps 0.02 --blackbox --exp_name preactresnet18-imagenetmixed10 --cla_model preactresnet18 --cla_network_pkl XXX-cla4-XXX --dataset imagenetmixed10 --test_adv_dataset /root/autodl-tmp/maggie/result/attack/pgd/preactresnet18-imagenetmixed10/20220813/00003-testacc-20.43/attack-imagenetmixed10-dataset/latent-attack-samples/test --batch_size 32 --epochs 40 --lr 0.01 >> /root/autodl-nas/maggie/mmat/log/ImagenetMixed10/defense/input-mixup-training/preactresnet18-imagenetmixed10-inputmixup-om-pgd-20220818.log 2>&1


# # #   使用PGD对抗训练后的 cla5
# # PreActResNet18 inputmixup beta(1,1) against pgd (eps=0.02)
# python -u tasklauncher-20220818.py run --save_path /root/autodl-tmp/maggie/result --mode defense --defense_mode inputmixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode pgd --attack_eps 0.02 --blackbox --exp_name preactresnet18-imagenetmixed10 --cla_model preactresnet18 --cla_network_pkl XXX-cla5-XXX --dataset imagenetmixed10 --test_adv_dataset /root/autodl-tmp/maggie/result/attack/pgd/preactresnet18-imagenetmixed10/20220812/00000-testacc-2.03/attack-imagenetmixed10-dataset/samples/test --batch_size 32 --epochs 40 --lr 0.01 >> /root/autodl-nas/maggie/mmat/log/ImagenetMixed10/defense/input-mixup-training/preactresnet18-imagenetmixed10-inputmixup-pgd-20220818.log 2>&1

# # PreActResNet18 inputmixup beta(1,1) against autoattack (eps=0.02)
# python -u tasklauncher-20220818.py run --save_path /root/autodl-tmp/maggie/result --mode defense --defense_mode inputmixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode autoattack --attack_eps 0.02 --blackbox --exp_name preactresnet18-imagenetmixed10 --cla_model preactresnet18 --cla_network_pkl XXX-cla5-XXX --dataset imagenetmixed10 --test_adv_dataset /root/autodl-tmp/maggie/result/attack/autoattack/preactresnet18-imagenetmixed10/20220813/00000-testacc-0.00/attack-imagenetmixed10-dataset/samples/test --batch_size 32 --epochs 40 --lr 0.01 >> /root/autodl-nas/maggie/mmat/log/ImagenetMixed10/defense/input-mixup-training/preactresnet18-imagenetmixed10-inputmixup-autoattack-20220818.log 2>&1

# # PreActResNet18 inputmixup beta(1,1) against deepfool (eps=0.02)
# python -u tasklauncher-20220818.py run --save_path /root/autodl-tmp/maggie/result --mode defense --defense_mode inputmixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode deepfool --attack_eps 0.02 --blackbox --exp_name preactresnet18-imagenetmixed10 --cla_model preactresnet18 --cla_network_pkl XXX-cla5-XXX --dataset imagenetmixed10 --test_adv_dataset /root/autodl-tmp/maggie/result/attack/deepfool/preactresnet18-imagenetmixed10/20220813/00000-testacc-8.87/attack-imagenetmixed10-dataset/samples/test --batch_size 32 --epochs 40 --lr 0.01 >> /root/autodl-nas/maggie/mmat/log/ImagenetMixed10/defense/input-mixup-training/preactresnet18-imagenetmixed10-inputmixup-deepfool-20220818.log 2>&1

# # PreActResNet18 inputmixup beta(1,1) against cw (eps=0.02)
# python -u tasklauncher-20220818.py run --save_path /root/autodl-tmp/maggie/result --mode defense --defense_mode inputmixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode cw --attack_eps 0.02 --blackbox --exp_name preactresnet18-imagenetmixed10 --cla_model preactresnet18 --cla_network_pkl XXX-cla5-XXX --dataset imagenetmixed10 --test_adv_dataset /root/autodl-tmp/maggie/result/attack/cw/preactresnet18-imagenetmixed10/20220813/00000-testacc-0.10/attack-imagenetmixed10-dataset/samples/test --batch_size 32 --epochs 40 --lr 0.01 >> /root/autodl-nas/maggie/mmat/log/ImagenetMixed10/defense/input-mixup-training/preactresnet18-imagenetmixed10-inputmixup-cw-20220818.log 2>&1

# # PreActResNet18 inputmixup beta(1,1) against om-fgsm (eps=0.02)
# python -u tasklauncher-20220818.py run --save_path /root/autodl-tmp/maggie/result --mode defense --defense_mode inputmixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode om-fgsm --attack_eps 0.02 --blackbox --exp_name preactresnet18-imagenetmixed10 --cla_model preactresnet18 --cla_network_pkl XXX-cla5-XXX --dataset imagenetmixed10 --test_adv_dataset /root/autodl-tmp/maggie/result/attack/fgsm/preactresnet18-imagenetmixed10/20220813/00003-testacc-26.90/attack-imagenetmixed10-dataset/latent-attack-samples/test --batch_size 32 --epochs 40 --lr 0.01 >> /root/autodl-nas/maggie/mmat/log/ImagenetMixed10/defense/input-mixup-training/preactresnet18-imagenetmixed10-inputmixup-om-fgsm-20220818.log 2>&1

# # PreActResNet18 inputmixup beta(1,1) against om-pgd (eps=0.02)
# python -u tasklauncher-20220818.py run --save_path /root/autodl-tmp/maggie/result --mode defense --defense_mode inputmixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode om-pgd --attack_eps 0.02 --blackbox --exp_name preactresnet18-imagenetmixed10 --cla_model preactresnet18 --cla_network_pkl XXX-cla5-XXX --dataset imagenetmixed10 --test_adv_dataset /root/autodl-tmp/maggie/result/attack/pgd/preactresnet18-imagenetmixed10/20220813/00003-testacc-20.43/attack-imagenetmixed10-dataset/latent-attack-samples/test --batch_size 32 --epochs 40 --lr 0.01 >> /root/autodl-nas/maggie/mmat/log/ImagenetMixed10/defense/input-mixup-training/preactresnet18-imagenetmixed10-inputmixup-om-pgd-20220818.log 2>&1


# # #   使用PGD对抗训练后的 cla6
# # PreActResNet18 inputmixup beta(1,1) against pgd (eps=0.02)
# python -u tasklauncher-20220818.py run --save_path /root/autodl-tmp/maggie/result --mode defense --defense_mode inputmixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode pgd --attack_eps 0.02 --blackbox --exp_name preactresnet18-imagenetmixed10 --cla_model preactresnet18 --cla_network_pkl XXX-cla6-XXX --dataset imagenetmixed10 --test_adv_dataset /root/autodl-tmp/maggie/result/attack/pgd/preactresnet18-imagenetmixed10/20220812/00000-testacc-2.03/attack-imagenetmixed10-dataset/samples/test --batch_size 32 --epochs 40 --lr 0.01 >> /root/autodl-nas/maggie/mmat/log/ImagenetMixed10/defense/input-mixup-training/preactresnet18-imagenetmixed10-inputmixup-pgd-20220818.log 2>&1

# # PreActResNet18 inputmixup beta(1,1) against autoattack (eps=0.02)
# python -u tasklauncher-20220818.py run --save_path /root/autodl-tmp/maggie/result --mode defense --defense_mode inputmixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode autoattack --attack_eps 0.02 --blackbox --exp_name preactresnet18-imagenetmixed10 --cla_model preactresnet18 --cla_network_pkl XXX-cla6-XXX --dataset imagenetmixed10 --test_adv_dataset /root/autodl-tmp/maggie/result/attack/autoattack/preactresnet18-imagenetmixed10/20220813/00000-testacc-0.00/attack-imagenetmixed10-dataset/samples/test --batch_size 32 --epochs 40 --lr 0.01 >> /root/autodl-nas/maggie/mmat/log/ImagenetMixed10/defense/input-mixup-training/preactresnet18-imagenetmixed10-inputmixup-autoattack-20220818.log 2>&1

# # PreActResNet18 inputmixup beta(1,1) against deepfool (eps=0.02)
# python -u tasklauncher-20220818.py run --save_path /root/autodl-tmp/maggie/result --mode defense --defense_mode inputmixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode deepfool --attack_eps 0.02 --blackbox --exp_name preactresnet18-imagenetmixed10 --cla_model preactresnet18 --cla_network_pkl XXX-cla6-XXX --dataset imagenetmixed10 --test_adv_dataset /root/autodl-tmp/maggie/result/attack/deepfool/preactresnet18-imagenetmixed10/20220813/00000-testacc-8.87/attack-imagenetmixed10-dataset/samples/test --batch_size 32 --epochs 40 --lr 0.01 >> /root/autodl-nas/maggie/mmat/log/ImagenetMixed10/defense/input-mixup-training/preactresnet18-imagenetmixed10-inputmixup-deepfool-20220818.log 2>&1

# # PreActResNet18 inputmixup beta(1,1) against cw (eps=0.02)
# python -u tasklauncher-20220818.py run --save_path /root/autodl-tmp/maggie/result --mode defense --defense_mode inputmixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode cw --attack_eps 0.02 --blackbox --exp_name preactresnet18-imagenetmixed10 --cla_model preactresnet18 --cla_network_pkl XXX-cla6-XXX --dataset imagenetmixed10 --test_adv_dataset /root/autodl-tmp/maggie/result/attack/cw/preactresnet18-imagenetmixed10/20220813/00000-testacc-0.10/attack-imagenetmixed10-dataset/samples/test --batch_size 32 --epochs 40 --lr 0.01 >> /root/autodl-nas/maggie/mmat/log/ImagenetMixed10/defense/input-mixup-training/preactresnet18-imagenetmixed10-inputmixup-cw-20220818.log 2>&1

# # PreActResNet18 inputmixup beta(1,1) against om-fgsm (eps=0.02)
# python -u tasklauncher-20220818.py run --save_path /root/autodl-tmp/maggie/result --mode defense --defense_mode inputmixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode om-fgsm --attack_eps 0.02 --blackbox --exp_name preactresnet18-imagenetmixed10 --cla_model preactresnet18 --cla_network_pkl XXX-cla6-XXX --dataset imagenetmixed10 --test_adv_dataset /root/autodl-tmp/maggie/result/attack/fgsm/preactresnet18-imagenetmixed10/20220813/00003-testacc-26.90/attack-imagenetmixed10-dataset/latent-attack-samples/test --batch_size 32 --epochs 40 --lr 0.01 >> /root/autodl-nas/maggie/mmat/log/ImagenetMixed10/defense/input-mixup-training/preactresnet18-imagenetmixed10-inputmixup-om-fgsm-20220818.log 2>&1

# # PreActResNet18 inputmixup beta(1,1) against om-pgd (eps=0.02)
# python -u tasklauncher-20220818.py run --save_path /root/autodl-tmp/maggie/result --mode defense --defense_mode inputmixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode om-pgd --attack_eps 0.02 --blackbox --exp_name preactresnet18-imagenetmixed10 --cla_model preactresnet18 --cla_network_pkl XXX-cla6-XXX --dataset imagenetmixed10 --test_adv_dataset /root/autodl-tmp/maggie/result/attack/pgd/preactresnet18-imagenetmixed10/20220813/00003-testacc-20.43/attack-imagenetmixed10-dataset/latent-attack-samples/test --batch_size 32 --epochs 40 --lr 0.01 >> /root/autodl-nas/maggie/mmat/log/ImagenetMixed10/defense/input-mixup-training/preactresnet18-imagenetmixed10-inputmixup-om-pgd-20220818.log 2>&1

# # PreActResNet18 cutmixup beta(1,1) against FGSM (eps=0.02)
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220818.py run --save_path /root/autodl-tmp/maggie/result --mode defense --defense_mode cutmixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode fgsm --attack_eps 0.02 --blackbox --exp_name preactresnet18-imagenetmixed10 --cla_model preactresnet18 --cla_network_pkl /root/autodl-tmp/maggie/result/train/cla-train/preactresnet18-imagenetmixed10/standard-trained-classifier-preactresnet18-on-clean-imagenetmixed10-epoch-0023-acc-90.47.pkl --dataset imagenetmixed10 --test_adv_dataset /root/autodl-tmp/maggie/result/attack/fgsm/preactresnet18-imagenetmixed10/20220813/00000-testacc-13.93/attack-imagenetmixed10-dataset/samples/test --batch_size 32 --epochs 40 --lr 0.01 >> /root/autodl-nas/maggie/mmat/log/ImagenetMixed10/defense/cut-mixup-training/preactresnet18-imagenetmixed10-cutmixup-fgsm-20220818.log 2>&1

# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220818.py run --save_path /root/autodl-tmp/maggie/result --mode defense --defense_mode cutmixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode fgsm --attack_eps 0.02 --blackbox --exp_name preactresnet18-imagenetmixed10 --cla_model preactresnet18 --cla_network_pkl /root/autodl-tmp/maggie/result/train/cla-train/preactresnet18-imagenetmixed10/standard-trained-classifier-preactresnet18-on-clean-imagenetmixed10-epoch-0023-acc-90.47.pkl --dataset imagenetmixed10 --test_adv_dataset /root/autodl-tmp/maggie/result/attack/fgsm/preactresnet18-imagenetmixed10/20220813/00000-testacc-13.93/attack-imagenetmixed10-dataset/samples/test --batch_size 32 --epochs 40 --lr 0.001 >> /root/autodl-nas/maggie/mmat/log/ImagenetMixed10/defense/cut-mixup-training/preactresnet18-imagenetmixed10-cutmixup-fgsm-20220818.log 2>&1

# # PreActResNet18 puzzlemixup beta(1,1) against FGSM (eps=0.02)
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220818.py run --save_path /root/autodl-tmp/maggie/result --mode defense --defense_mode puzzlemixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode fgsm --attack_eps 0.02 --blackbox --exp_name preactresnet18-imagenetmixed10 --cla_model preactresnet18 --cla_network_pkl /root/autodl-tmp/maggie/result/train/cla-train/preactresnet18-imagenetmixed10/standard-trained-classifier-preactresnet18-on-clean-imagenetmixed10-epoch-0023-acc-90.47.pkl --dataset imagenetmixed10 --test_adv_dataset /root/autodl-tmp/maggie/result/attack/fgsm/preactresnet18-imagenetmixed10/20220813/00000-testacc-13.93/attack-imagenetmixed10-dataset/samples/test --batch_size 32 --epochs 40 --lr 0.01 >> /root/autodl-nas/maggie/mmat/log/ImagenetMixed10/defense/puzzle-mixup-training/preactresnet18-imagenetmixed10-puzzlemixup-fgsm-20220818.log 2>&1

# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220818.py run --save_path /root/autodl-tmp/maggie/result --mode defense --defense_mode puzzlemixup --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --attack_mode fgsm --attack_eps 0.02 --blackbox --exp_name preactresnet18-imagenetmixed10 --cla_model preactresnet18 --cla_network_pkl /root/autodl-tmp/maggie/result/train/cla-train/preactresnet18-imagenetmixed10/standard-trained-classifier-preactresnet18-on-clean-imagenetmixed10-epoch-0023-acc-90.47.pkl --dataset imagenetmixed10 --test_adv_dataset /root/autodl-tmp/maggie/result/attack/fgsm/preactresnet18-imagenetmixed10/20220813/00000-testacc-13.93/attack-imagenetmixed10-dataset/samples/test --batch_size 32 --epochs 40 --lr 0.001 >> /root/autodl-nas/maggie/mmat/log/ImagenetMixed10/defense/puzzle-mixup-training/preactresnet18-imagenetmixed10-puzzlemixup-fgsm-20220818.log 2>&1