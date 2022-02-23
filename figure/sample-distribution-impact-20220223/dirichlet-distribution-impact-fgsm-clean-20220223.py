import matplotlib.pyplot as plt
import csv

plt.switch_backend('agg')


# 读取数据-dirichlet分布下的adv准确率
with open("cifar10-preactresnet18-whitebox-fgsm-acc-dirichlet-20220223.csv",'r') as f:
    data_csv = csv.reader(f)

    step = []
    preactresnet18_dirichlet_low_adv = []
    preactresnet18_dirichlet_mid_adv = []
    preactresnet18_dirichlet_high_adv = []

    next(data_csv)
    for row in data_csv:
        step.append(str(int(row[0])))
        preactresnet18_dirichlet_low_adv.append(float(row[1]))
        preactresnet18_dirichlet_mid_adv.append(float(row[2]))
        preactresnet18_dirichlet_high_adv.append(float(row[3]))


with open("cifar10-preactresnet34-whitebox-fgsm-acc-dirichlet-20220223.csv",'r') as f:
    data_csv = csv.reader(f)

    step = []
    preactresnet34_dirichlet_low_adv = []
    preactresnet34_dirichlet_mid_adv = []
    preactresnet34_dirichlet_high_adv = []

    next(data_csv)
    for row in data_csv:
        step.append(str(int(row[0])))
        preactresnet34_dirichlet_low_adv.append(float(row[1]))
        preactresnet34_dirichlet_mid_adv.append(float(row[2]))
        preactresnet34_dirichlet_high_adv.append(float(row[3]))


# 读取数据-dirichlet分布下的cle准确率
with open("cifar10-preactresnet18-clean-acc-dirichlet-20220223.csv",'r') as f:
    data_csv = csv.reader(f)

    step = []
    preactresnet18_dirichlet_low_cle = []
    preactresnet18_dirichlet_mid_cle = []
    preactresnet18_dirichlet_high_cle = []

    next(data_csv)
    for row in data_csv:
        step.append(str(int(row[0])))
        preactresnet18_dirichlet_low_cle.append(float(row[1]))
        preactresnet18_dirichlet_mid_cle.append(float(row[2]))
        preactresnet18_dirichlet_high_cle.append(float(row[3]))


with open("cifar10-preactresnet34-clean-acc-dirichlet-20220223.csv",'r') as f:
    data_csv = csv.reader(f)

    step = []
    preactresnet34_dirichlet_low_cle = []
    preactresnet34_dirichlet_mid_cle = []
    preactresnet34_dirichlet_high_cle = []

    next(data_csv)
    for row in data_csv:
        step.append(str(int(row[0])))
        preactresnet34_dirichlet_low_cle.append(float(row[1]))
        preactresnet34_dirichlet_mid_cle.append(float(row[2]))
        preactresnet34_dirichlet_high_cle.append(float(row[3]))


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
# plt.ylim(0.08, 0.28)
plt.ylim(0.08, 0.30)

plt.grid()
plt.plot(step,preactresnet18_dirichlet_low_adv, label=f'{gamma_str}=({0.5},{0.5},{0.5})', marker='o', markersize=3)
plt.plot(step,preactresnet18_dirichlet_mid_adv, label=f'{gamma_str}=({1.0},{1.0},{1.0})', marker='p', markersize=3)
plt.plot(step,preactresnet18_dirichlet_high_adv, label=f'{gamma_str}=({10},{10},{10})', marker='^', markersize=3)

plt.plot(step,preactresnet34_dirichlet_low_adv, label=f'{gamma_str}=({0.5},{0.5},{0.5})', marker='s', markersize=3)
plt.plot(step,preactresnet34_dirichlet_mid_adv, label=f'{gamma_str}=({1.0},{1.0},{1.0})', marker='D', markersize=3)
plt.plot(step,preactresnet34_dirichlet_high_adv, label=f'{gamma_str}=({10},{10},{10})', marker='*', markersize=3)

plt.legend([f'PreActResNet18 {gamma_str}=({0.5},{0.5},{0.5})',        # gamma=5换成gamma=0.5
f'PreActResNet18 {gamma_str}=({1.0},{1.0},{1.0})',
f'PreActResNet18 {gamma_str}=({10},{10},{10})',
f'PreActResNet34 {gamma_str}=({0.5},{0.5},{0.5})',
f'PreActResNet34 {gamma_str}=({1.0},{1.0},{1.0})',
f'PreActResNet34 {gamma_str}=({10},{10},{10})'
],
fontsize = 8, 
loc='lower right') # .打出图例
# loc='upper left') #   打出图例
# loc='lower left') #   打出图例

plt.title(f'Impact of dirichlet({gamma_str}) distribution to the ternary convex mixup',fontsize=12, pad=12)
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
# plt.ylim(0.75, 0.89)

plt.grid()
plt.plot(step,preactresnet18_dirichlet_low_cle, label=f'{gamma_str}=({0.5},{0.5},{0.5})', marker='o', markersize=3)
plt.plot(step,preactresnet18_dirichlet_mid_cle, label=f'{gamma_str}=({1.0},{1.0},{1.0})', marker='p', markersize=3)
plt.plot(step,preactresnet18_dirichlet_high_cle, label=f'{gamma_str}=({10},{10},{10})', marker='^', markersize=3)

plt.plot(step,preactresnet34_dirichlet_low_cle, label=f'{gamma_str}=({0.5},{0.5},{0.5})', marker='s', markersize=3)
plt.plot(step,preactresnet34_dirichlet_mid_cle, label=f'{gamma_str}=({1.0},{1.0},{1.0})', marker='D', markersize=3)
plt.plot(step,preactresnet34_dirichlet_high_cle, label=f'{gamma_str}=({10},{10},{10})', marker='*', markersize=3)

plt.legend([f'PreActResNet18 {gamma_str}=({0.5},{0.5},{0.5})',    # gamma=5换成gamma=0.5
f'PreActResNet18 {gamma_str}=({1.0},{1.0},{1.0})',
f'PreActResNet18 {gamma_str}=({10},{10},{10})',
f'PreActResNet34 {gamma_str}=({0.5},{0.5},{0.5})',
f'PreActResNet34 {gamma_str}=({1.0},{1.0},{1.0})',
f'PreActResNet34 {gamma_str}=({10},{10},{10})'
],
fontsize = 8, 
loc='lower right') # .打出图例
# loc='upper left') #   打出图例
# loc='lower left') #   打出图例

plt.title(f'Impact of dirichlet({gamma_str}) distribution to the ternary convex mixup',fontsize=12, pad=12)
# plt.title(f'Classify White-box FGSM',pad=20)
plt.xlabel('Epoch',fontsize=10)
plt.ylabel('Top1 accuracy(%) on clean testset',fontsize=10)

x_major_locator=plt.MultipleLocator(2)          #   把x轴的刻度间隔设置为1，并存在变量里
y_major_locator=plt.MultipleLocator(0.01)       #   把y轴的刻度间隔设置为10，并存在变量里
ax=plt.gca()                                    #   ax为两条坐标轴的实例
ax.xaxis.set_major_locator(x_major_locator)     #   把x轴的主刻度设置为1的倍数
ax.yaxis.set_major_locator(y_major_locator)     #   把y轴的主刻度设置为10的倍数

plt.show()
savepath = f'/home/maggie/mmat/figure/sample-distribution-impact-20220223'
savename = f'cifar10-preactresnet18-preactresnet34-adv-cle-acc-dirichlet-20220223'
plt.savefig(f'{savepath}/{savename}.png')
plt.close()


