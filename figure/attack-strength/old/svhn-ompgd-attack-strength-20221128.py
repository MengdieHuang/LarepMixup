import matplotlib.pyplot as plt
import csv

plt.switch_backend('agg')

# 读取不同扰动强度下的svhn adv准确率
with open("svhn-ompgd-attack-strengths-20221128.csv",'r') as f:
    data_csv = csv.reader(f)

    step = []
    AlexNet_adv = []
    ResNet18_adv = []    
    ResNet34_adv = []
    ResNet50_adv = []    
    VGG19_adv = []
    DenseNet169_adv = []
    GoogleNet_adv = []

    next(data_csv)
    for row in data_csv:
        step.append(str(float(row[0])))

        AlexNet_adv.append(float(row[1]))
        ResNet18_adv.append(float(row[2]))
        ResNet34_adv.append(float(row[3]))
        ResNet50_adv.append(float(row[4]))
        VGG19_adv.append(float(row[5]))
        DenseNet169_adv.append(float(row[6]))
        GoogleNet_adv.append(float(row[7]))

plt.plot(step,AlexNet_adv, label=f'AlexNet', marker='o', markersize=4)
plt.plot(step,ResNet18_adv, label=f'ResNet18', marker='p', markersize=4)
plt.plot(step,ResNet34_adv, label=f'ResNet34', marker='^', markersize=4)
plt.plot(step,ResNet50_adv, label=f'ResNet50', marker='p', markersize=4)
plt.plot(step,VGG19_adv, label=f'VGG19', marker='^', markersize=4)
plt.plot(step,DenseNet169_adv, label=f'DenseNet169', marker='p', markersize=4)
plt.plot(step,GoogleNet_adv, label=f'GoogleNet', marker='^', markersize=4)

plt.legend([f'AlexNet', 
f'ResNet18',
f'ResNet34',
f'ResNet50',
f'VGG19',
f'DenseNet169',
f'GoogleNet'],
fontsize = 8, 
loc='upper right') #   打出图例

# plt.title(f'Accuracy Improvement by LarepMixup on OM-PGD Attacks',fontsize=12, pad=12)
plt.title(f'Improved Accuracy from LarepMixup on SVHN OM-PGD Attacks ',fontsize=12, pad=12)

# plt.title(f'Classify White-box FGSM',pad=20)
plt.xlabel('Zeta',fontsize=10)
plt.ylabel('Increased Accuracy',fontsize=10)


# x_major_locator=plt.MultipleLocator(2)          #   把x轴的刻度间隔设置为1，并存在变量里
# y_major_locator=plt.MultipleLocator(0.02)       #   把y轴的刻度间隔设置为10，并存在变量里
# ax=plt.gca()                                    #   ax为两条坐标轴的实例
# ax.xaxis.set_major_locator(x_major_locator)     #   把x轴的主刻度设置为1的倍数
# ax.yaxis.set_major_locator(y_major_locator)     #   把y轴的主刻度设置为10的倍数


plt.show()
savepath = f'/home/maggie/mmat/figure/attack-strength'
savename = f'svhn-ompgd-attack-strengths-20221128'
plt.savefig(f'{savepath}/{savename}.png')
plt.close()


