source ~/.bashrc
# source /home/xieyi/anaconda3/bin/activate mmat
source /root/miniconda3/bin/activate mmat

# ----------------preactresnet18----------------------
# preactresnet18 imagenetmixed10 stdftrain  lr = 0.01
python -u tasklauncher-20220722.py run --save_path /root/autodl-tmp/maggie/result --mode train --train_mode cla-train --exp_name preactresnet18-imagenetmixed10 --cla_model preactresnet18 --dataset imagenetmixed10 --batch_size 32 --epochs 40 --lr 0.01 >> /root/autodl-nas/maggie/mmat/log/ImagenetMixed10/classsifier-standard-train/preactresnet18-imagenetmixed10-stdtrain-20220722.log 2>&1

# preactresnet18 imagenetmixed10 stdftrain  lr = 0.001
python -u tasklauncher-20220722.py run --save_path /root/autodl-tmp/maggie/result --mode train --train_mode cla-train --exp_name preactresnet18-imagenetmixed10 --cla_model preactresnet18 --dataset imagenetmixed10 --batch_size 32 --epochs 40 --lr 0.001 >> /root/autodl-nas/maggie/mmat/log/ImagenetMixed10/classsifier-standard-train/preactresnet18-imagenetmixed10-stdtrain-20220722.log 2>&1

# preactresnet18 imagenetmixed10 stdftrain  lr = 0.01
python -u tasklauncher-20220722.py run --save_path /root/autodl-tmp/maggie/result --mode train --train_mode cla-train --exp_name preactresnet18-imagenetmixed10 --cla_model preactresnet18 --dataset imagenetmixed10 --batch_size 32 --epochs 40 --lr 0.01 >> /root/autodl-nas/maggie/mmat/log/ImagenetMixed10/classsifier-standard-train/preactresnet18-imagenetmixed10-stdtrain-20220722.log 2>&1

# preactresnet18 imagenetmixed10 stdftrain  lr = 0.001
python -u tasklauncher-20220722.py run --save_path /root/autodl-tmp/maggie/result --mode train --train_mode cla-train --exp_name preactresnet18-imagenetmixed10 --cla_model preactresnet18 --dataset imagenetmixed10 --batch_size 32 --epochs 40 --lr 0.001 >> /root/autodl-nas/maggie/mmat/log/ImagenetMixed10/classsifier-standard-train/preactresnet18-imagenetmixed10-stdtrain-20220722.log 2>&1

# ----------------preactresnet34----------------------
# preactresnet34 imagenetmixed10 stdftrain  lr = 0.01
python -u tasklauncher-20220722.py run --save_path /root/autodl-tmp/maggie/result --mode train --train_mode cla-train --exp_name preactresnet34-imagenetmixed10 --cla_model preactresnet34 --dataset imagenetmixed10 --batch_size 16 --epochs 40 --lr 0.01 >> /root/autodl-nas/maggie/mmat/log/ImagenetMixed10/classsifier-standard-train/preactresnet34-imagenetmixed10-stdtrain-20220722.log 2>&1

# preactresnet34 imagenetmixed10 stdftrain  lr = 0.001
python -u tasklauncher-20220722.py run --save_path /root/autodl-tmp/maggie/result --mode train --train_mode cla-train --exp_name preactresnet34-imagenetmixed10 --cla_model preactresnet34 --dataset imagenetmixed10 --batch_size 32 --epochs 40 --lr 0.001 >> /root/autodl-nas/maggie/mmat/log/ImagenetMixed10/classsifier-standard-train/preactresnet34-imagenetmixed10-stdtrain-20220722.log 2>&1

# preactresnet34 imagenetmixed10 stdftrain  lr = 0.01
python -u tasklauncher-20220722.py run --save_path /root/autodl-tmp/maggie/result --mode train --train_mode cla-train --exp_name preactresnet34-imagenetmixed10 --cla_model preactresnet34 --dataset imagenetmixed10 --batch_size 32 --epochs 40 --lr 0.01 >> /root/autodl-nas/maggie/mmat/log/ImagenetMixed10/classsifier-standard-train/preactresnet34-imagenetmixed10-stdtrain-20220722.log 2>&1

# preactresnet34 imagenetmixed10 stdftrain  lr = 0.001
python -u tasklauncher-20220722.py run --save_path /root/autodl-tmp/maggie/result --mode train --train_mode cla-train --exp_name preactresnet34-imagenetmixed10 --cla_model preactresnet34 --dataset imagenetmixed10 --batch_size 32 --epochs 40 --lr 0.001 >> /root/autodl-nas/maggie/mmat/log/ImagenetMixed10/classsifier-standard-train/preactresnet34-imagenetmixed10-stdtrain-20220722.log 2>&1

#----------------preactresnet50----------------------
# preactresnet50 imagenetmixed10 stdftrain  lr = 0.01
python -u tasklauncher-20220722.py run --save_path /root/autodl-tmp/maggie/result --mode train --train_mode cla-train --exp_name preactresnet50-imagenetmixed10 --cla_model preactresnet50 --dataset imagenetmixed10 --batch_size 16 --epochs 40 --lr 0.01 >> /root/autodl-nas/maggie/mmat/log/ImagenetMixed10/classsifier-standard-train/preactresnet50-imagenetmixed10-stdtrain-20220722.log 2>&1

# preactresnet50 imagenetmixed10 stdftrain  lr = 0.001
python -u tasklauncher-20220722.py run --save_path /root/autodl-tmp/maggie/result --mode train --train_mode cla-train --exp_name preactresnet50-imagenetmixed10 --cla_model preactresnet50 --dataset imagenetmixed10 --batch_size 32 --epochs 40 --lr 0.001 >> /root/autodl-nas/maggie/mmat/log/ImagenetMixed10/classsifier-standard-train/preactresnet50-imagenetmixed10-stdtrain-20220722.log 2>&1

# preactresnet50 imagenetmixed10 stdftrain  lr = 0.01
python -u tasklauncher-20220722.py run --save_path /root/autodl-tmp/maggie/result --mode train --train_mode cla-train --exp_name preactresnet50-imagenetmixed10 --cla_model preactresnet50 --dataset imagenetmixed10 --batch_size 32 --epochs 40 --lr 0.01 >> /root/autodl-nas/maggie/mmat/log/ImagenetMixed10/classsifier-standard-train/preactresnet50-imagenetmixed10-stdtrain-20220722.log 2>&1

# preactresnet50 imagenetmixed10 stdftrain  lr = 0.001
python -u tasklauncher-20220722.py run --save_path /root/autodl-tmp/maggie/result --mode train --train_mode cla-train --exp_name preactresnet50-imagenetmixed10 --cla_model preactresnet50 --dataset imagenetmixed10 --batch_size 32 --epochs 40 --lr 0.001 >> /root/autodl-nas/maggie/mmat/log/ImagenetMixed10/classsifier-standard-train/preactresnet50-imagenetmixed10-stdtrain-20220722.log 2>&1