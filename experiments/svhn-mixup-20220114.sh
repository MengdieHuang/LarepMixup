source ~/.bashrc
source /home/xieyi/anaconda3/bin/activate mmat

# svhn dual convex miup 32
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220114.py run --mode interpolate --mix_w_num 2 --exp_name stylegan2ada-svhn --gen_model stylegan2ada --dataset svhn --gen_network_pkl /home/maggie/mmat/result/train/gen-train/stylegan2ada-svhn/20210718/00000/svhn-auto1-batch64-ada-bgc-noresume/network-snapshot-025000.pkl --mix_mode basemixup --sample_mode betasampler --beta_alpha 1.0 --projected_dataset /home/maggie/mmat/result/project/stylegan2ada-svhn/20210914/00000/project-svhn-trainset --mix_img_num 100 >> /home/maggie/mmat/log/svhn-mixup/svhn-mixup-20220114.log 2>&1

# svhn ternary convex miup 32
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220114.py run --mode interpolate --mix_w_num 3 --exp_name stylegan2ada-svhn --gen_model stylegan2ada --dataset svhn --gen_network_pkl /home/maggie/mmat/result/train/gen-train/stylegan2ada-svhn/20210718/00000/svhn-auto1-batch64-ada-bgc-noresume/network-snapshot-025000.pkl --mix_mode basemixup --sample_mode dirichletsampler --dirichlet_gama 1.0 --projected_dataset /home/maggie/mmat/result/project/stylegan2ada-svhn/20210914/00000/project-svhn-trainset --mix_img_num 100 >> /home/maggie/mmat/log/svhn-mixup/svhn-mixup-20220114.log 2>&1


#----------------------------------------------
# svhn dual mask miup 32
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220114.py run --mode interpolate --mix_w_num 2 --exp_name stylegan2ada-svhn --gen_model stylegan2ada --dataset svhn --gen_network_pkl /home/maggie/mmat/result/train/gen-train/stylegan2ada-svhn/20210718/00000/svhn-auto1-batch64-ada-bgc-noresume/network-snapshot-025000.pkl --mix_mode maskmixup --sample_mode bernoullisampler --projected_dataset /home/maggie/mmat/result/project/stylegan2ada-svhn/20210914/00000/project-svhn-trainset --mix_img_num 100 >> /home/maggie/mmat/log/svhn-mixup/svhn-mixup-20220114.log 2>&1

# svhn ternary mask miup 32
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220114.py run --mode interpolate --mix_w_num 3 --exp_name stylegan2ada-svhn --gen_model stylegan2ada --dataset svhn --gen_network_pkl /home/maggie/mmat/result/train/gen-train/stylegan2ada-svhn/20210718/00000/svhn-auto1-batch64-ada-bgc-noresume/network-snapshot-025000.pkl --mix_mode maskmixup --sample_mode bernoullisampler3 --dirichlet_gama 1.0 --projected_dataset /home/maggie/mmat/result/project/stylegan2ada-svhn/20210914/00000/project-svhn-trainset --mix_img_num 100 >> /home/maggie/mmat/log/svhn-mixup/svhn-mixup-20220114.log 2>&1