import matplotlib.pyplot as plt
import csv

plt.switch_backend('agg')

# 读取数据
with open("cifar10-preactresnet18-clean-acc-convex-20211028.csv",'r') as f:
    data_csv = csv.reader(f)

    step = []
    preactresnet18_convex_dual = []
    preactresnet18_convex_ternary = []    

    next(data_csv)
    for row in data_csv:
        step.append(str(int(row[0])))
        preactresnet18_convex_dual.append(float(row[1]))
        preactresnet18_convex_ternary.append(float(row[2]))

with open("cifar10-preactresnet18-clean-acc-mask-20211028.csv",'r') as f:
    data_csv = csv.reader(f)

    step = []
    preactresnet18_mask_dual = []
    preactresnet18_mask_ternary = []    

    next(data_csv)
    for row in data_csv:
        step.append(str(int(row[0])))
        preactresnet18_mask_dual.append(float(row[1]))
        preactresnet18_mask_ternary.append(float(row[2]))

# with open("cifar10-preactresnet34-clean-acc-dual-20211027.csv",'r') as f:
#     data_csv = csv.reader(f)
 
#     step = []
#     preactresnet34_dual_convex = []
#     preactresnet34_dual_mask = []

#     next(data_csv)
#     for row in data_csv:
#         step.append(str(int(row[0])))
#         preactresnet34_dual_convex.append(float(row[1]))
#         preactresnet34_dual_mask.append(float(row[2]))

# with open("cifar10-preactresnet34-clean-acc-ternary-20211027.csv",'r') as f:
#     data_csv = csv.reader(f)
 
#     step = []
#     preactresnet34_ternary_convex = []
#     preactresnet34_ternary_mask = []

#     next(data_csv)
#     for row in data_csv:
#         step.append(str(int(row[0])))
#         preactresnet34_ternary_convex.append(float(row[1]))
#         preactresnet34_ternary_mask.append(float(row[2]))

# 设置符号
beta_num = 946
beta_str = chr(beta_num)
gamma_num = 947
gamma_str = chr(gamma_num)

# 绘图
plt.figure(figsize = (6,8))

#  beta distribution---------------------------------------
plt.subplot(2,1,1)
# plt.subplots_adjust(wspace=0.1)
plt.subplots_adjust(hspace=0.4)

plt.xlim(0, 40)
plt.ylim(0.78, 0.89)

plt.grid()
plt.plot(step,preactresnet18_convex_dual, label=f'dual mixup', marker='o', markersize=3)
plt.plot(step,preactresnet18_convex_ternary, label=f'ternary mixup', marker='p', markersize=3)

# plt.plot(step,preactresnet34_dual_convex, label=f'convex mixup', marker='s', markersize=3)
# plt.plot(step,preactresnet34_dual_mask, label=f'binary mask mixup', marker='D', markersize=3)

plt.legend([f'PreActResNet18 dual mixup', 
f'PreActResNet18 ternary mixup'],
fontsize = 7, 
loc='lower right') #   打出图例
# loc='upper left') #   打出图例
# loc='lower left') #   打出图例

plt.title(f'Impact of source number to the convex mixup',fontsize=12, pad=12)
# plt.title(f'Classify White-box FGSM',pad=20)
plt.xlabel('Epoch',fontsize=10)
plt.ylabel('Top1 accuracy(%) on clean testset',fontsize=10)


x_major_locator=plt.MultipleLocator(2)          #   把x轴的刻度间隔设置为1，并存在变量里
y_major_locator=plt.MultipleLocator(0.01)       #   把y轴的刻度间隔设置为10，并存在变量里
ax=plt.gca()                                    #   ax为两条坐标轴的实例
ax.xaxis.set_major_locator(x_major_locator)     #   把x轴的主刻度设置为1的倍数
ax.yaxis.set_major_locator(y_major_locator)     #   把y轴的主刻度设置为10的倍数


#  ternary mixup------------------------------------------
plt.subplot(2,1,2)
# plt.subplots_adjust(wspace=0.1)
plt.subplots_adjust(hspace=0.4)

plt.xlim(0, 40)
plt.ylim(0.78, 0.89)

plt.grid()
plt.plot(step,preactresnet18_mask_dual, label=f'dual mixup', marker='o', markersize=3)
plt.plot(step,preactresnet18_mask_ternary, label=f'ternary mixup', marker='p', markersize=3)

# plt.plot(step,preactresnet34_ternary_convex, label=f'convex mixup', marker='s', markersize=3)
# plt.plot(step,preactresnet34_ternary_mask, label=f'binary mask mixup', marker='D', markersize=3)

plt.legend([f'PreActResNet18 dual mixup',
f'PreActResNet18 ternary mixup'],
fontsize = 7, 
loc='lower right') # .打出图例
# loc='upper left') #   打出图例
# loc='lower left') #   打出图例

plt.title(f'Impact of source number to the binary mask mixup',fontsize=12, pad=12)
# plt.title(f'Classify White-box FGSM',pad=20)
plt.xlabel('Epoch',fontsize=10)
plt.ylabel('Top1 accuracy(%) on clean testset',fontsize=10)

x_major_locator=plt.MultipleLocator(2)          #   把x轴的刻度间隔设置为1，并存在变量里
y_major_locator=plt.MultipleLocator(0.01)       #   把y轴的刻度间隔设置为10，并存在变量里
ax=plt.gca()                                    #   ax为两条坐标轴的实例
ax.xaxis.set_major_locator(x_major_locator)     #   把x轴的主刻度设置为1的倍数
ax.yaxis.set_major_locator(y_major_locator)     #   把y轴的主刻度设置为10的倍数

plt.show()
savepath = f'/home/maggie/mmat/figure/source-num'
savename = f'cifar10-preactresnet18-clean-acc-convex-mask-20211028'
plt.savefig(f'{savepath}/{savename}.png')
plt.close()


