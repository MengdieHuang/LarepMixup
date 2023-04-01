source ~/.bashrc
source /home/xieyi/anaconda3/bin/activate mmat

# XXX-vanilla-cla-XXX
# XXX-dmat-seed1-cla-XXX

# eval vanilla wideresnet28_10 on cifar10: OM-FGSM (eps0.05-step0.05)
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20230331.py run --mode eval --exp_name wideresnet28_10-cifar10 --attack_mode fgsm --latentattack --attack_eps 0.05 --attack_eps_step 0.05 --whitebox --cla_model wideresnet28_10 --cla_network_pkl /home/data/maggie/result-newhome/train/cla-train/wideresnet28_10-cifar10/20230326/00002-testacc-0.9670/train-cifar10-dataset/standard-trained-classifier-wideresnet28_10-on-clean-cifar10-finished.pkl --gen_network_pkl /home/data/maggie/result-newhome/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --projected_testset /home/data/maggie/result-newhome/project/stylegan2ada-cifar10/20220627/00000/project-cifar10-testset --dataset cifar10 --test_adv_dataset /home/data/maggie/result-newhome/attack/fgsm/wideresnet28_10-cifar10/20230328/00003-om-eps05-step05-acc49.91/attack-cifar10-dataset/latent-attack-samples/test --batch_size 64 --seed 0 >> /home/maggie/mmat/log/CIFAR10/eval/montecarloexp/dmat-wideresnet28_10-cifar10-against-whiteomfgsm-eps0.05-step0.05-20230331.log 2>&1

# eval dmat-seed1 wideresnet28_10 on cifar10: OM-FGSM (eps0.05-step0.05)
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20230331.py run --mode eval --exp_name wideresnet28_10-cifar10 --attack_mode fgsm --latentattack --attack_eps 0.05 --attack_eps_step 0.05 --whitebox --cla_model wideresnet28_10 --cla_network_pkl /home/data/maggie/result-newhome/seed-1/defense/dmat/pgd/wideresnet28_10-cifar10/blackbox/20230329/00001-cleacc-96.81/adversarial-trained-classifier-wideresnet28_10-on-cifar10-epoch-0040.pkl --gen_network_pkl /home/data/maggie/result-newhome/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --projected_testset /home/data/maggie/result-newhome/project/stylegan2ada-cifar10/20220627/00000/project-cifar10-testset --dataset cifar10 --test_adv_dataset /home/data/maggie/result-newhome/attack/fgsm/wideresnet28_10-cifar10/20230328/00003-om-eps05-step05-acc49.91/attack-cifar10-dataset/latent-attack-samples/test --batch_size 64 --seed 1 >> /home/maggie/mmat/log/CIFAR10/eval/montecarloexp/dmat-wideresnet28_10-cifar10-against-whiteomfgsm-eps0.05-step0.05-20230331.log 2>&1

# eval dmat-seed2 wideresnet28_10 on cifar10: OM-FGSM (eps0.05-step0.05)
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20230331.py run --mode eval --exp_name wideresnet28_10-cifar10 --attack_mode fgsm --latentattack --attack_eps 0.05 --attack_eps_step 0.05 --whitebox --cla_model wideresnet28_10 --cla_network_pkl /home/data/maggie/result-newhome/seed-2/defense/dmat/pgd/wideresnet28_10-cifar10/blackbox/20230329/00001-cleacc-96.84/adversarial-trained-classifier-wideresnet28_10-on-cifar10-epoch-0040.pkl --gen_network_pkl /home/data/maggie/result-newhome/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --projected_testset /home/data/maggie/result-newhome/project/stylegan2ada-cifar10/20220627/00000/project-cifar10-testset --dataset cifar10 --test_adv_dataset /home/data/maggie/result-newhome/attack/fgsm/wideresnet28_10-cifar10/20230328/00003-om-eps05-step05-acc49.91/attack-cifar10-dataset/latent-attack-samples/test --batch_size 64 --seed 2 >> /home/maggie/mmat/log/CIFAR10/eval/montecarloexp/dmat-wideresnet28_10-cifar10-against-whiteomfgsm-eps0.05-step0.05-20230331.log 2>&1

# eval dmat-seed3 wideresnet28_10 on cifar10: OM-FGSM (eps0.05-step0.05)
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20230331.py run --mode eval --exp_name wideresnet28_10-cifar10 --attack_mode fgsm --latentattack --attack_eps 0.05 --attack_eps_step 0.05 --whitebox --cla_model wideresnet28_10 --cla_network_pkl /home/data/maggie/result-newhome/seed-3/defense/dmat/pgd/wideresnet28_10-cifar10/blackbox/20230329/00001-cleacc-96.95/adversarial-trained-classifier-wideresnet28_10-on-cifar10-epoch-0040.pkl --gen_network_pkl /home/data/maggie/result-newhome/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --projected_testset /home/data/maggie/result-newhome/project/stylegan2ada-cifar10/20220627/00000/project-cifar10-testset --dataset cifar10 --test_adv_dataset /home/data/maggie/result-newhome/attack/fgsm/wideresnet28_10-cifar10/20230328/00003-om-eps05-step05-acc49.91/attack-cifar10-dataset/latent-attack-samples/test --batch_size 64 --seed 3 >> /home/maggie/mmat/log/CIFAR10/eval/montecarloexp/dmat-wideresnet28_10-cifar10-against-whiteomfgsm-eps0.05-step0.05-20230331.log 2>&1

# eval dmat-seed4 wideresnet28_10 on cifar10: OM-FGSM (eps0.05-step0.05)
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20230331.py run --mode eval --exp_name wideresnet28_10-cifar10 --attack_mode fgsm --latentattack --attack_eps 0.05 --attack_eps_step 0.05 --whitebox --cla_model wideresnet28_10 --cla_network_pkl /home/data/maggie/result-newhome/seed-4/defense/dmat/pgd/wideresnet28_10-cifar10/blackbox/20230330/00001-cleacc-96.93/adversarial-trained-classifier-wideresnet28_10-on-cifar10-epoch-0040.pkl --gen_network_pkl /home/data/maggie/result-newhome/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --projected_testset /home/data/maggie/result-newhome/project/stylegan2ada-cifar10/20220627/00000/project-cifar10-testset --dataset cifar10 --test_adv_dataset /home/data/maggie/result-newhome/attack/fgsm/wideresnet28_10-cifar10/20230328/00003-om-eps05-step05-acc49.91/attack-cifar10-dataset/latent-attack-samples/test --batch_size 64 --seed 4 >> /home/maggie/mmat/log/CIFAR10/eval/montecarloexp/dmat-wideresnet28_10-cifar10-against-whiteomfgsm-eps0.05-step0.05-20230331.log 2>&1

# eval dmat-seed5 wideresnet28_10 on cifar10: OM-FGSM (eps0.05-step0.05)
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20230331.py run --mode eval --exp_name wideresnet28_10-cifar10 --attack_mode fgsm --latentattack --attack_eps 0.05 --attack_eps_step 0.05 --whitebox --cla_model wideresnet28_10 --cla_network_pkl /home/data/maggie/result-newhome/seed-5/defense/dmat/pgd/wideresnet28_10-cifar10/blackbox/20230330/00001-cleacc-96.84/adversarial-trained-classifier-wideresnet28_10-on-cifar10-epoch-0040.pkl --gen_network_pkl /home/data/maggie/result-newhome/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --projected_testset /home/data/maggie/result-newhome/project/stylegan2ada-cifar10/20220627/00000/project-cifar10-testset --dataset cifar10 --test_adv_dataset /home/data/maggie/result-newhome/attack/fgsm/wideresnet28_10-cifar10/20230328/00003-om-eps05-step05-acc49.91/attack-cifar10-dataset/latent-attack-samples/test --batch_size 64 --seed 5 >> /home/maggie/mmat/log/CIFAR10/eval/montecarloexp/dmat-wideresnet28_10-cifar10-against-whiteomfgsm-eps0.05-step0.05-20230331.log 2>&1

