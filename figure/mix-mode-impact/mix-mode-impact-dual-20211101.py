import matplotlib.pyplot as plt
import csv

plt.switch_backend('agg')

# 读取数据-dual mixup在adv上的准确率
with open("cifar10-preactresnet18-whitebox-fgsm-acc-dual-20211028.csv",'r') as f:
    data_csv = csv.reader(f)

    step = []
    preactresnet18_dual_convex_adv = []
    preactresnet18_dual_mask_adv = []    

    next(data_csv)
    for row in data_csv:
        step.append(str(int(row[0])))
        preactresnet18_dual_convex_adv.append(float(row[1]))
        preactresnet18_dual_mask_adv.append(float(row[2]))


with open("cifar10-preactresnet34-whitebox-fgsm-acc-dual-20211030.csv",'r') as f:
    data_csv = csv.reader(f)
 
    step = []
    preactresnet34_dual_convex_adv = []
    preactresnet34_dual_mask_adv = []

    next(data_csv)
    for row in data_csv:
        step.append(str(int(row[0])))
        preactresnet34_dual_convex_adv.append(float(row[1]))
        preactresnet34_dual_mask_adv.append(float(row[2]))

  
# 读取数据-dual mixup在cle上的准确率
with open("cifar10-preactresnet18-clean-acc-dual-20211028.csv",'r') as f:
    data_csv = csv.reader(f)

    step = []
    preactresnet18_dual_convex_cle = []
    preactresnet18_dual_mask_cle = []    

    next(data_csv)
    for row in data_csv:
        step.append(str(int(row[0])))
        preactresnet18_dual_convex_cle.append(float(row[1]))
        preactresnet18_dual_mask_cle.append(float(row[2]))


with open("cifar10-preactresnet34-clean-acc-dual-20211030.csv",'r') as f:
    data_csv = csv.reader(f)
 
    step = []
    preactresnet34_dual_convex_cle = []
    preactresnet34_dual_mask_cle = []

    next(data_csv)
    for row in data_csv:
        step.append(str(int(row[0])))
        preactresnet34_dual_convex_cle.append(float(row[1]))
        preactresnet34_dual_mask_cle.append(float(row[2]))

# 设置符号
beta_num = 946
beta_str = chr(beta_num)
gamma_num = 947
gamma_str = chr(gamma_num)

# 绘图
plt.figure(figsize = (6,8))

#  dual---------------------------------------
plt.subplot(2,1,1)
# plt.subplots_adjust(wspace=0.1)
plt.subplots_adjust(hspace=0.4)

plt.xlim(0, 40)
plt.ylim(0.08, 0.24)
# plt.ylim(0.09, 0.25)

plt.grid()
plt.plot(step,preactresnet18_dual_convex_adv, label=f'convex mixup', marker='o', markersize=3)
plt.plot(step,preactresnet18_dual_mask_adv, label=f'binary mask mixup', marker='p', markersize=3)

plt.plot(step,preactresnet34_dual_convex_adv, label=f'convex mixup', marker='s', markersize=3)
plt.plot(step,preactresnet34_dual_mask_adv, label=f'binary mask mixup', marker='D', markersize=3)

plt.legend([f'PreActResNet18 convex mixup', 
f'PreActResNet18 binary mask mixup',
f'PreActResNet34 convex mixup', 
f'PreActResNet34 binary mask mixup'
],
fontsize = 7, 
loc='lower right') #   打出图例
# loc='upper left') #   打出图例
# loc='lower left') #   打出图例

plt.title(f'Impact of mix mode to the dual mixup',fontsize=12, pad=12)
# plt.title(f'Classify White-box FGSM',pad=20)
plt.xlabel('Epoch',fontsize=10)
plt.ylabel('Top1 accuracy(%) on white-box FGSM',fontsize=10)


x_major_locator=plt.MultipleLocator(2)          #   把x轴的刻度间隔设置为1，并存在变量里
y_major_locator=plt.MultipleLocator(0.02)       #   把y轴的刻度间隔设置为10，并存在变量里
ax=plt.gca()                                    #   ax为两条坐标轴的实例
ax.xaxis.set_major_locator(x_major_locator)     #   把x轴的主刻度设置为1的倍数
ax.yaxis.set_major_locator(y_major_locator)     #   把y轴的主刻度设置为10的倍数

#  dual---------------------------------------
plt.subplot(2,1,2)
# plt.subplots_adjust(wspace=0.1)
plt.subplots_adjust(hspace=0.4)

plt.xlim(0, 40)
plt.ylim(0.74, 0.90)

plt.grid()
plt.plot(step,preactresnet18_dual_convex_cle, label=f'convex mixup', marker='o', markersize=3)
plt.plot(step,preactresnet18_dual_mask_cle, label=f'binary mask mixup', marker='p', markersize=3)

plt.plot(step,preactresnet34_dual_convex_cle, label=f'convex mixup', marker='s', markersize=3)
plt.plot(step,preactresnet34_dual_mask_cle, label=f'binary mask mixup', marker='D', markersize=3)

plt.legend([f'PreActResNet18 convex mixup', 
f'PreActResNet18 binary mask mixup',
f'PreActResNet34 convex mixup', 
f'PreActResNet34 binary mask mixup'
],
fontsize = 7, 
loc='lower right') #   打出图例
# loc='upper left') #   打出图例
# loc='lower left') #   打出图例

plt.title(f'Impact of mix mode to the dual mixup',fontsize=12, pad=12)
# plt.title(f'Classify White-box FGSM',pad=20)
plt.xlabel('Epoch',fontsize=10)
plt.ylabel('Top1 accuracy(%) on clean testset',fontsize=10)


x_major_locator=plt.MultipleLocator(2)          #   把x轴的刻度间隔设置为1，并存在变量里
y_major_locator=plt.MultipleLocator(0.02)       #   把y轴的刻度间隔设置为10，并存在变量里
ax=plt.gca()                                    #   ax为两条坐标轴的实例
ax.xaxis.set_major_locator(x_major_locator)     #   把x轴的主刻度设置为1的倍数
ax.yaxis.set_major_locator(y_major_locator)     #   把y轴的主刻度设置为10的倍数



plt.show()
savepath = f'/home/maggie/mmat/figure/mix-mode-impact'
savename = f'cifar10-preactresnet18-preactresnet34-adv-cle-acc-dual-20211101'
plt.savefig(f'{savepath}/{savename}.png')
plt.close()


