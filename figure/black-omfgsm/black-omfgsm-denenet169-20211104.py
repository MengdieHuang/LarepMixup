import matplotlib.pyplot as plt
import csv
import numpy as np

plt.switch_backend('agg')

with open("cifar10-densenet169-cle-adv-acc-black-omfgsm.csv",'r') as f:
    data_csv = csv.reader(f)

    adversary_model = []            #   ResNet18  ResNet34    ResNet50    VGG19   DenseNet169 GoogleNet
    std_normal = []
    std_omfgsm = []    
    repmix_p5_normal = []           #   beta(0.5,0.5)
    repmix_p5_omfgsm = []
    repmix_2_normal = []            #   beta(2,2)
    repmix_2_omfgsm = []


    next(data_csv)
    for row in data_csv:
        adversary_model.append(str(row[0]))
        std_normal.append(float(row[1]))
        std_omfgsm.append(float(row[2]))
        repmix_p5_normal.append(float(row[3]))
        repmix_p5_omfgsm.append(float(row[4]))
        repmix_2_normal.append(float(row[5]))
        repmix_2_omfgsm.append(float(row[6]))

# 设置符号
beta_num = 946
beta_str = chr(beta_num)

width = 0.1  # 柱子的宽度
x = np.arange(len(adversary_model))  # x轴刻度标签位置
print("x :",x)
print("adversary_model:",adversary_model)

#---------------------------左边 clean-----------------
# fig = plt.figure(figsize = (7,6))
ax1 = plt.subplot()

ax1.bar(x - 5*(width/2), std_normal, width, color= 'dodgerblue', label=f'Cle-Standard train' )
ax1.bar(x - 3*(width/2), repmix_p5_normal, width, color= 'deepskyblue', label=f'Cle-RepMixup({beta_str}={0.5:.1f}) train' )
ax1.bar(x - 1*(width/2), repmix_2_normal, width, color= 'lightskyblue', label=f'Cle-RepMixup({beta_str}={2:.1f}) train' )

plt.xticks(x, adversary_model,fontsize = 9)
# plt.xticks(x, adversary_model, fontsize = 10, rotation=15)

ax1.set_ylim(70, 90)
ax1.set_ylabel('Top1 accuracy(%) on clean testset',fontsize=10)
ax1.set_xlabel('Adversary Model',fontsize=10)

y_major_locator=plt.MultipleLocator(2)       #   把y轴的刻度间隔设置为10，并存在变量里
ax1.yaxis.set_major_locator(y_major_locator)     #   把y轴的主刻度设置为10的倍数

#---------------------------右边 omfgsm-----------------
ax2 = ax1.twinx()

ax2.bar(x + 1*(width/2), std_omfgsm, width, color= 'orangered', label=f'Adv-Standard train')
ax2.bar(x + 3*(width/2), repmix_p5_omfgsm,width, color= 'darkorange', label=f'Adv-RepMixup({beta_str}={0.5:.1f}) train')
ax2.bar(x + 5*(width/2), repmix_2_omfgsm,width, color= 'gold', label=f'Adv-RepMixup({beta_str}={2:.1f}) train')


ax2.set_ylim(10, 30)
ax2.set_ylabel('Top1 accuracy(%) on black-box OM-FGSM',fontsize=10)

y_major_locator=plt.MultipleLocator(2)              
ax2.yaxis.set_major_locator(y_major_locator)     

#---------图例---------
handles1, labels1 = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()
plt.legend(handles1+handles2, labels1+labels2, fontsize = 9, loc='lower right')
# plt.legend(handles1+handles2, labels1+labels2, fontsize = 7, loc='upper right')

#---------标题---------
# plt.title(f'Accuracy of DenseNet169 on Black-box Representation-level Adversarial Attacks',fontsize=12, pad=12)
plt.title(f'DenseNet169 against Black-box Representation-level Adversarial Attacks',fontsize=12, pad=12)

plt.show()
savepath = f'/home/maggie/mmat/figure/black-omfgsm'
savename = f'cifar10-densenet169-cle-adv-acc-black-omfgsm-20211104'
plt.savefig(f'{savepath}/{savename}.png')
plt.close()


