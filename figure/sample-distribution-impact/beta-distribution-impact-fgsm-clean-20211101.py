import matplotlib.pyplot as plt
import csv

plt.switch_backend('agg')

# 读取数据-beta分布下的adv准确率
with open("cifar10-preactresnet18-whitebox-fgsm-acc-beta-20211027.csv",'r') as f:
    data_csv = csv.reader(f)

    step = []
    preactresnet18_beta_low_adv = []
    preactresnet18_beta_mid_adv = []    
    preactresnet18_beta_high_adv = []

    next(data_csv)
    for row in data_csv:
        step.append(str(int(row[0])))
        preactresnet18_beta_low_adv.append(float(row[1]))
        preactresnet18_beta_mid_adv.append(float(row[2]))
        preactresnet18_beta_high_adv.append(float(row[3]))

with open("cifar10-preactresnet34-whitebox-fgsm-acc-beta-20211027.csv",'r') as f:
    data_csv = csv.reader(f)

    step = []
    preactresnet34_beta_low_adv = []
    preactresnet34_beta_mid_adv = []
    preactresnet34_beta_high_adv = []

    next(data_csv)
    for row in data_csv:
        step.append(str(int(row[0])))
        preactresnet34_beta_low_adv.append(float(row[1]))
        preactresnet34_beta_mid_adv.append(float(row[2]))
        preactresnet34_beta_high_adv.append(float(row[3]))

# 读取数据-beta分布下的cle准确率
with open("cifar10-preactresnet18-clean-acc-beta-20211027.csv",'r') as f:
    data_csv = csv.reader(f)

    step = []
    preactresnet18_beta_low_cle = []
    preactresnet18_beta_mid_cle = []    
    preactresnet18_beta_high_cle = []

    next(data_csv)
    for row in data_csv:
        step.append(str(int(row[0])))
        preactresnet18_beta_low_cle.append(float(row[1]))
        preactresnet18_beta_mid_cle.append(float(row[2]))
        preactresnet18_beta_high_cle.append(float(row[3]))


with open("cifar10-preactresnet34-clean-acc-beta-20211027.csv",'r') as f:
    data_csv = csv.reader(f)

    step = []
    preactresnet34_beta_low_cle = []
    preactresnet34_beta_mid_cle = []
    preactresnet34_beta_high_cle = []

    next(data_csv)
    for row in data_csv:
        step.append(str(int(row[0])))
        preactresnet34_beta_low_cle.append(float(row[1]))
        preactresnet34_beta_mid_cle.append(float(row[2]))
        preactresnet34_beta_high_cle.append(float(row[3]))



# 设置符号
beta_num = 946
beta_str = chr(beta_num)
gamma_num = 947
gamma_str = chr(gamma_num)

# 绘图
plt.figure(figsize = (6,8))

#  fgsm上的准确率---------------------------------------
plt.subplot(2,1,1)
# plt.subplots_adjust(wspace=0.1)
plt.subplots_adjust(hspace=0.4)

plt.xlim(0, 40)
plt.ylim(0.08, 0.28)

plt.grid()
plt.plot(step,preactresnet18_beta_low_adv, label=f'{beta_str}=({0.5:.1f},{0.5:.1f})', marker='o', markersize=3)
plt.plot(step,preactresnet18_beta_mid_adv, label=f'{beta_str}=({1:.1f},{1:.1f})', marker='p', markersize=3)
plt.plot(step,preactresnet18_beta_high_adv, label=f'{beta_str}=({2:.1f},{2:.1f})', marker='^', markersize=3)

plt.plot(step,preactresnet34_beta_low_adv, label=f'{beta_str}=({0.5:.1f},{0.5:.1f})', marker='s', markersize=3)
plt.plot(step,preactresnet34_beta_mid_adv, label=f'{beta_str}=({1:.1f},{1:.1f})', marker='D', markersize=3)
plt.plot(step,preactresnet34_beta_high_adv, label=f'{beta_str}=({2:.1f},{2:.1f})', marker='*', markersize=3)


# plt.plot(step,beta_low, label=f'{beta_str}=({0.5:.1f},{0.5:.1f})', linestyle='-')
# plt.plot(step,beta_high, label=f'{beta_str}=({2:.1f},{2:.1f})', linestyle='--') 

plt.legend([f'PreActResNet18 {beta_str}=({0.5:.1f},{0.5:.1f})', 
f'PreActResNet18 {beta_str}=({1:.1f},{1:.1f})',
f'PreActResNet18 {beta_str}=({2:.1f},{2:.1f})',
f'PreActResNet34 {beta_str}=({0.5:.1f},{0.5:.1f})',
f'PreActResNet34 {beta_str}=({1:.1f},{1:.1f})',
f'PreActResNet34 {beta_str}=({2:.1f},{2:.1f})'],
fontsize = 7, 
loc='lower right') #   打出图例
# loc='upper left') #   打出图例
# loc='lower left') #   打出图例

plt.title(f'Impact of beta({beta_str}) distribution to the dual convex mixup',fontsize=12, pad=12)
# plt.title(f'Classify White-box FGSM',pad=20)
plt.xlabel('Epoch',fontsize=10)
plt.ylabel('Top1 accuracy(%) on white-box FGSM',fontsize=10)


x_major_locator=plt.MultipleLocator(2)          #   把x轴的刻度间隔设置为1，并存在变量里
y_major_locator=plt.MultipleLocator(0.02)       #   把y轴的刻度间隔设置为10，并存在变量里
ax=plt.gca()                                    #   ax为两条坐标轴的实例
ax.xaxis.set_major_locator(x_major_locator)     #   把x轴的主刻度设置为1的倍数
ax.yaxis.set_major_locator(y_major_locator)     #   把y轴的主刻度设置为10的倍数


#  clean上的准确率---------------------------------------
plt.subplot(2,1,2)
# plt.subplots_adjust(wspace=0.1)
plt.subplots_adjust(hspace=0.4)

plt.xlim(0, 40)
plt.ylim(0.76, 0.89)

plt.grid()
plt.plot(step,preactresnet18_beta_low_cle, label=f'{beta_str}=({0.5:.1f},{0.5:.1f})', marker='o', markersize=3)
plt.plot(step,preactresnet18_beta_mid_cle, label=f'{beta_str}=({1:.1f},{1:.1f})', marker='p', markersize=3)
plt.plot(step,preactresnet18_beta_high_cle, label=f'{beta_str}=({2:.1f},{2:.1f})', marker='^', markersize=3)

plt.plot(step,preactresnet34_beta_low_cle, label=f'{beta_str}=({0.5:.1f},{0.5:.1f})', marker='s', markersize=3)
plt.plot(step,preactresnet34_beta_mid_cle, label=f'{beta_str}=({1:.1f},{1:.1f})', marker='D', markersize=3)
plt.plot(step,preactresnet34_beta_high_cle, label=f'{beta_str}=({2:.1f},{2:.1f})', marker='*', markersize=3)

plt.legend([f'PreActResNet18 {beta_str}=({0.5:.1f},{0.5:.1f})', 
f'PreActResNet18 {beta_str}=({1:.1f},{1:.1f})',
f'PreActResNet18 {beta_str}=({2:.1f},{2:.1f})',
f'PreActResNet34 {beta_str}=({0.5:.1f},{0.5:.1f})',
f'PreActResNet34 {beta_str}=({1:.1f},{1:.1f})',
f'PreActResNet34 {beta_str}=({2:.1f},{2:.1f})'],
fontsize = 7, 
loc='lower right') #   打出图例
# loc='upper left') #   打出图例
# loc='lower left') #   打出图例

plt.title(f'Impact of beta({beta_str}) distribution to the dual convex mixup',fontsize=12, pad=12)
# plt.title(f'Classify White-box FGSM',pad=20)
plt.xlabel('Epoch',fontsize=10)
plt.ylabel('Top1 accuracy(%) on clean testset',fontsize=10)


x_major_locator=plt.MultipleLocator(2)          #   把x轴的刻度间隔设置为1，并存在变量里
y_major_locator=plt.MultipleLocator(0.01)       #   把y轴的刻度间隔设置为10，并存在变量里
ax=plt.gca()                                    #   ax为两条坐标轴的实例
ax.xaxis.set_major_locator(x_major_locator)     #   把x轴的主刻度设置为1的倍数
ax.yaxis.set_major_locator(y_major_locator)     #   把y轴的主刻度设置为10的倍数



plt.show()
savepath = f'/home/maggie/mmat/figure/sample-distribution-impact'
savename = f'cifar10-preactresnet18-preactresnet34-adv-cle-acc-beta-20211101'
plt.savefig(f'{savepath}/{savename}.png')
plt.close()


