source ~/.bashrc
# source /home/xieyi/anaconda3/bin/activate mmat
source /root/miniconda3/bin/activate mmat

# project imagenetmixed10 trainset
#--------------------
python tasklauncher-20220721.py run --save_path /root/autodl-tmp/maggie/result --exp_name stylegan2ada-imagenetmixed10 --mode project --gen_model stylegan2ada --dataset imagenetmixed10 --gen_network_pkl /root/autodl-tmp/maggie/result/train/gen-train/stylegan2ada-imagenetmixed10/20220714/00001/imagenetmixed10-auto2-batch128-ada-bgc-noresume/network-snapshot-022937.pkl >> /home/maggie/mmat/log/ImagenetMixed10/dataset-project/trainset/stylegan2ada-imagenetmixed10-trainset-project-20220721.log 2>&1
