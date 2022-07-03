source ~/.bashrc
# source /home/xieyi/anaconda3/bin/activate mmat
source /root/miniconda3/bin/activate mmat

# train stylegan2ada for imagenetmixed10
#--------------------
# python tasklauncher-20220702.py run --save_path /root/autodl-tmp/maggie/result --exp_name stylegan2ada-imagenetmixed10 --gen_model stylegan2ada --dataset imagenetmixed10 --data /root/autodl-tmp/maggie/data/imagenetmixed10/imagenetmixed104stylegan2ada/datasets/imagenetmixed10.zip --epochs 1 --batch_size 32 --mode train --train_mode gen-train --gpus 2 --workers 16 >> /root/autodl-nas/maggie/mmat/log/ImagenetMixed10/stylegan2ada-train/imagenetmixed10-stylegan2ada-train-20220703.log 2>&1

# python tasklauncher-20220702.py run --save_path /root/autodl-tmp/maggie/result --exp_name stylegan2ada-imagenetmixed10 --gen_model stylegan2ada --dataset imagenetmixed10 --data /root/autodl-tmp/maggie/data/imagenetmixed10/imagenetmixed104stylegan2ada/datasets/imagenetmixed10.zip --epochs 1 --batch_size 128 --mode train --train_mode gen-train --gpus 2 --workers 16 >> /root/autodl-nas/maggie/mmat/log/ImagenetMixed10/stylegan2ada-train/imagenetmixed10-stylegan2ada-train-2gpu-20220703.log 2>&1

python tasklauncher-20220702.py run --save_path /root/autodl-tmp/maggie/result --exp_name stylegan2ada-imagenetmixed10 --gen_model stylegan2ada --dataset imagenetmixed10 --data /root/autodl-tmp/maggie/data/imagenetmixed10/imagenetmixed104stylegan2ada/datasets/imagenetmixed10.zip --epochs 1 --batch_size 128 --mode train --train_mode gen-train --gpus 2 --workers 16 --pretrain_pkl_path /root/autodl-tmp/maggie/result/train/gen-train/stylegan2ada-imagenetmixed10/20220703/00000/imagenetmixed10-auto1-batch32-ada-bgc-noresume/network-snapshot-000080.pkl >> /root/autodl-nas/maggie/mmat/log/ImagenetMixed10/stylegan2ada-train/imagenetmixed10-stylegan2ada-train-2gpu-20220703.log 2>&1


