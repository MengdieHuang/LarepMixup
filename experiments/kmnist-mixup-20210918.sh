source ~/.bashrc
source /home/xieyi/anaconda3/bin/activate mmat

# kmnist dual convex miup 70000目标
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20210918.py run --mode interpolate --exp_name stylegan2ada-kmnist --gen_model stylegan2ada --dataset kmnist --gen_network_pkl /home/maggie/mmat/result/train/gen-train/stylegan2ada-kmnist/20210829/00000/kmnist-auto1-batch64-ada-bgc-noresume/network-snapshot-020240.pkl --mix_mode basemixup --sample_mode uniformsampler --projected_dataset /home/maggie/mmat/result/project/stylegan2ada-kmnist/20210913/00000/project-kmnist-trainset --mix_img_num 70000 >> /home/maggie/mmat/log/kmnist-mixup/kmnist-dualconvex-mixup-20210918.log 2>&1