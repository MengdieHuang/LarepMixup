source ~/.bashrc
# source /home/xieyi/anaconda3/bin/activate mmat
source /root/miniconda3/bin/activate mmat

# project imagenetmixed10 trainset
#--------------------
# python tasklauncher-20220721.py run --save_path /root/autodl-tmp/maggie/result --exp_name stylegan2ada-imagenetmixed10 --mode project --gen_model stylegan2ada --dataset imagenetmixed10 --gen_network_pkl /root/autodl-tmp/maggie/result/train/gen-train/stylegan2ada-imagenetmixed10/network-snapshot-022937.pkl --batch_size 16 >> /root/autodl-nas/maggie/mmat/log/ImagenetMixed10/dataset-project/trainset/stylegan2ada-imagenetmixed10-trainset-project-20220721.log 2>&1


# # 0721 gan
# python tasklauncher-20220721.py run --save_path /root/autodl-tmp/maggie/result --exp_name stylegan2ada-imagenetmixed10 --mode project --num_steps 1000 --gen_model stylegan2ada --dataset imagenetmixed10 --gen_network_pkl /root/autodl-tmp/maggie/result/train/gen-train/stylegan2ada-imagenetmixed10/network-snapshot-006799.pkl --batch_size 16 >> /root/autodl-nas/maggie/mmat/log/ImagenetMixed10/dataset-project/trainset/stylegan2ada-imagenetmixed10-trainset-project-20220721.log 2>&1

# # 0721 gan
# python tasklauncher-20220721.py run --save_path /root/autodl-tmp/maggie/result --exp_name stylegan2ada-imagenetmixed10 --mode project --num_steps 1500 --gen_model stylegan2ada --dataset imagenetmixed10 --gen_network_pkl /root/autodl-tmp/maggie/result/train/gen-train/stylegan2ada-imagenetmixed10/network-snapshot-006799.pkl --batch_size 16 >> /root/autodl-nas/maggie/mmat/log/ImagenetMixed10/dataset-project/trainset/stylegan2ada-imagenetmixed10-trainset-project-20220721.log 2>&1

# # 0721 gan
# python tasklauncher-20220721.py run --save_path /root/autodl-tmp/maggie/result --exp_name stylegan2ada-imagenetmixed10 --mode project --num_steps 2000 --gen_model stylegan2ada --dataset imagenetmixed10 --gen_network_pkl /root/autodl-tmp/maggie/result/train/gen-train/stylegan2ada-imagenetmixed10/network-snapshot-006799.pkl --batch_size 16 >> /root/autodl-nas/maggie/mmat/log/ImagenetMixed10/dataset-project/trainset/stylegan2ada-imagenetmixed10-trainset-project-20220721.log 2>&1

# test

# 0721 gan
python tasklauncher-20220722.py run --save_path /root/autodl-tmp/maggie/result --exp_name stylegan2ada-imagenetmixed10 --mode project --num_steps 1500 --gen_model stylegan2ada --dataset imagenetmixed10 --gen_network_pkl /root/autodl-tmp/maggie/result/train/gen-train/stylegan2ada-imagenetmixed10/network-snapshot-006799.pkl --batch_size 512 >> /root/autodl-nas/maggie/mmat/log/ImagenetMixed10/dataset-project/trainset/stylegan2ada-imagenetmixed10-trainset-project-20220722.log 2>&1