source ~/.bashrc
source /home/xieyi/anaconda3/bin/activate mmat

# vgg19 cifar10 rmt 20210926000  cle+mix lr =0.1 beta(2,2)
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20210924.py run --mode defense --defense_mode rmt --beta_alpha 2 --attack_mode fgsm --attack_eps 0.3 --whitebox True --exp_name vgg19-cifar10 --cla_model vgg19 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/vgg19-cifar10/20210908/00002-testacc-0.8705/train-cifar10-dataset/standard-trained-classifier-vgg19-on-clean-cifar10-epoch-0013.pkl --gen_model stylegan2ada --gen_network_pkl /home/maggie/mmat/result/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_dataset /home/maggie/mmat/result/project/stylegan2ada-cifar10/20210913/00000/project-cifar10-trainset --adv_dataset /home/maggie/mmat/result/attack/fgsm/vgg19-cifar10/20210917/00000-attackacc-0.14540/attack-cifar10-dataset/samples --batch_size 256 --epochs 40 --lr 0.1 >> /home/maggie/mmat/log/vgg19-cifar10-rmt/vgg19-cifar10-rmt-20210926.log 2>&1

# vgg19 cifar10 rmt 20210926001  cle+mix lr =0.01 beta(2,2)
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20210924.py run --mode defense --defense_mode rmt --beta_alpha 2 --attack_mode fgsm --attack_eps 0.3 --whitebox True --exp_name vgg19-cifar10 --cla_model vgg19 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/vgg19-cifar10/20210908/00002-testacc-0.8705/train-cifar10-dataset/standard-trained-classifier-vgg19-on-clean-cifar10-epoch-0013.pkl --gen_model stylegan2ada --gen_network_pkl /home/maggie/mmat/result/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_dataset /home/maggie/mmat/result/project/stylegan2ada-cifar10/20210913/00000/project-cifar10-trainset --adv_dataset /home/maggie/mmat/result/attack/fgsm/vgg19-cifar10/20210917/00000-attackacc-0.14540/attack-cifar10-dataset/samples --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/vgg19-cifar10-rmt/vgg19-cifar10-rmt-20210926.log 2>&1

# vgg19 cifar10 rmt 20210926002  cle+mix lr =0.001 beta(2,2)
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20210924.py run --mode defense --defense_mode rmt --beta_alpha 2 --attack_mode fgsm --attack_eps 0.3 --whitebox True --exp_name vgg19-cifar10 --cla_model vgg19 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/vgg19-cifar10/20210908/00002-testacc-0.8705/train-cifar10-dataset/standard-trained-classifier-vgg19-on-clean-cifar10-epoch-0013.pkl --gen_model stylegan2ada --gen_network_pkl /home/maggie/mmat/result/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_dataset /home/maggie/mmat/result/project/stylegan2ada-cifar10/20210913/00000/project-cifar10-trainset --adv_dataset /home/maggie/mmat/result/attack/fgsm/vgg19-cifar10/20210917/00000-attackacc-0.14540/attack-cifar10-dataset/samples --batch_size 256 --epochs 40 --lr 0.001 >> /home/maggie/mmat/log/vgg19-cifar10-rmt/vgg19-cifar10-rmt-20210926.log 2>&1


# vgg19 cifar10 rmt 20210926000  cle+mix lr =0.1 beta(0.5, 0.5)
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20210924.py run --mode defense --defense_mode rmt --beta_alpha 2 --attack_mode fgsm --attack_eps 0.3 --whitebox True --exp_name vgg19-cifar10 --cla_model vgg19 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/vgg19-cifar10/20210908/00002-testacc-0.8705/train-cifar10-dataset/standard-trained-classifier-vgg19-on-clean-cifar10-epoch-0013.pkl --gen_model stylegan2ada --gen_network_pkl /home/maggie/mmat/result/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_dataset /home/maggie/mmat/result/project/stylegan2ada-cifar10/20210913/00000/project-cifar10-trainset --adv_dataset /home/maggie/mmat/result/attack/fgsm/vgg19-cifar10/20210917/00000-attackacc-0.14540/attack-cifar10-dataset/samples --batch_size 256 --epochs 40 --lr 0.1 >> /home/maggie/mmat/log/vgg19-cifar10-rmt/vgg19-cifar10-rmt-20210926.log 2>&1

# vgg19 cifar10 rmt 20210926001  cle+mix lr =0.01 beta(0.5, 0.5)
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20210924.py run --mode defense --defense_mode rmt --beta_alpha 2 --attack_mode fgsm --attack_eps 0.3 --whitebox True --exp_name vgg19-cifar10 --cla_model vgg19 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/vgg19-cifar10/20210908/00002-testacc-0.8705/train-cifar10-dataset/standard-trained-classifier-vgg19-on-clean-cifar10-epoch-0013.pkl --gen_model stylegan2ada --gen_network_pkl /home/maggie/mmat/result/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_dataset /home/maggie/mmat/result/project/stylegan2ada-cifar10/20210913/00000/project-cifar10-trainset --adv_dataset /home/maggie/mmat/result/attack/fgsm/vgg19-cifar10/20210917/00000-attackacc-0.14540/attack-cifar10-dataset/samples --batch_size 256 --epochs 40 --lr 0.01 >> /home/maggie/mmat/log/vgg19-cifar10-rmt/vgg19-cifar10-rmt-20210926.log 2>&1

# vgg19 cifar10 rmt 20210926002  cle+mix lr =0.001 beta(0.5, 0.5)
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20210924.py run --mode defense --defense_mode rmt --beta_alpha 2 --attack_mode fgsm --attack_eps 0.3 --whitebox True --exp_name vgg19-cifar10 --cla_model vgg19 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/vgg19-cifar10/20210908/00002-testacc-0.8705/train-cifar10-dataset/standard-trained-classifier-vgg19-on-clean-cifar10-epoch-0013.pkl --gen_model stylegan2ada --gen_network_pkl /home/maggie/mmat/result/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_dataset /home/maggie/mmat/result/project/stylegan2ada-cifar10/20210913/00000/project-cifar10-trainset --adv_dataset /home/maggie/mmat/result/attack/fgsm/vgg19-cifar10/20210917/00000-attackacc-0.14540/attack-cifar10-dataset/samples --batch_size 256 --epochs 40 --lr 0.001 >> /home/maggie/mmat/log/vgg19-cifar10-rmt/vgg19-cifar10-rmt-20210926.log 2>&1