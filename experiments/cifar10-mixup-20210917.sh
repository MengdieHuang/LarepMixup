source ~/.bashrc
source /home/xieyi/anaconda3/bin/activate mmat

# cifar10 dual convex miup 70000目标
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20210917.py run --mode interpolate --exp_name stylegan2ada-cifar10 --gen_model stylegan2ada --dataset cifar10 --gen_network_pkl /home/maggie/mmat/result/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --mix_mode basemixup --sample_mode uniformsampler --projected_dataset /home/maggie/mmat/result/project/stylegan2ada-cifar10/20210913/00000/project-cifar10-trainset --mix_img_num 70000 >> /home/maggie/mmat/log/cifar10-mixup/cifar10-dualconvex-mixup-20210917.log 2>&1
