import matplotlib.pyplot as plt
import csv
import numpy as np

plt.switch_backend('agg')

# 读取数据-dual mixup在adv上的准确率
with open("cifar10-alexnet-cle-adv-acc-black-fgsm.csv",'r') as f:
    data_csv = csv.reader(f)

    adversary_model = []            #   ResNet18  ResNet34    ResNet50    VGG19   DenseNet169 GoogleNet
    std_normal = []
    std_fgsm = []    
    repmix_p5_normal = []           #   beta(0.5,0.5)
    repmix_p5_fgsm = []
    repmix_2_normal = []            #   beta(2,2)
    repmix_2_fgsm = []


    next(data_csv)
    for row in data_csv:
        adversary_model.append(str(row[0]))
        std_normal.append(float(row[1]))
        std_fgsm.append(float(row[2]))
        repmix_p5_normal.append(float(row[3]))
        repmix_p5_fgsm.append(float(row[4]))
        repmix_2_normal.append(float(row[5]))
        repmix_2_fgsm.append(float(row[6]))



# # 设置符号
# beta_num = 946
# beta_str = chr(beta_num)
# gamma_num = 947
# gamma_str = chr(gamma_num)

# 绘图
# plt.figure(figsize = (6,8))


plt.subplots_adjust(hspace=0.4)

# plt.xlim(0, 40)
# plt.ylim(0.08, 0.24)
# # plt.ylim(0.09, 0.25)

plt.grid()
x = np.arange(len(adversary_model))  # x轴刻度标签位置
width = 0.25  # 柱子的宽度



plt.bar(x - width/2, std_normal, width, label=f'Standard trained AlexNet on normal samples' )
plt.bar(x + width/2, std_fgsm,width, label=f'Standard trained AlexNet on FGSM samples')
plt.bar(x - width/2, repmix_p5_normal, width, label=f'RepMixup trained AlexNet on normal samples' )
plt.bar(x + width/2, repmix_p5_fgsm,width, label=f'RepMixup trained AlexNet on FGSM samples')

# plt.plot(step,preactresnet34_dual_convex_adv, label=f'convex mixup', marker='s', markersize=3)
# plt.plot(step,preactresnet34_dual_mask_adv, label=f'binary mask mixup', marker='D', markersize=3)

plt.legend([f'Standard trained AlexNet on normal samples', 
f'Standard trained AlexNet on FGSM samples',
f'RepMixup trained AlexNet on normal samples', 
f'RepMixup trained AlexNet on FGSM samples'
],
# fontsize = 7, 
# loc='lower right') #   打出图例
# loc='upper right') #   打出图例
# loc='lower left') #   打出图例
loc='center right') #   打出图例

plt.title(f'Accuracy of Alexnet on Black-box Pixel-level Adversarial Attacks',fontsize=12, pad=12)
# plt.title(f'Classify White-box FGSM',pad=20)

# plt.xticks(x, label = adversary_model)

plt.xlabel('Adversary Model',fontsize=10)
plt.ylabel('Top1 accuracy(%) on black-box FGSM',fontsize=10)



plt.show()
savepath = f'/home/maggie/mmat/figure/black-fgsm'
savename = f'cifar10-alexnet-cle-adv-acc-black-fgsm-20211103'
plt.savefig(f'{savepath}/{savename}.png')
plt.close()


