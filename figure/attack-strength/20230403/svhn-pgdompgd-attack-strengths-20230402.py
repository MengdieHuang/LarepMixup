import matplotlib.pyplot as plt
import csv

plt.switch_backend('agg')

# 读取不同模型在不同扰动强度下的cifar10 PGD准确率
with open("svhn-pgd-attack-strengths-20230402.csv",'r') as f:
    data_csv = csv.reader(f)

    step = []        # 0 0.02 0.05 0.1 0.2 0.3
    
    AlexNet_adv = []
    RMT_AlexNet_adv = []
    
    ResNet18_adv = []    
    RMT_ResNet18_adv = []    
    
    ResNet34_adv = []
    RMT_ResNet34_adv = []
    
    ResNet50_adv = []    
    RMT_ResNet50_adv = []    
    
    VGG19_adv = []
    RMT_VGG19_adv = []
    
    DenseNet169_adv = []
    RMT_DenseNet169_adv = []
    
    GoogleNet_adv = []
    RMT_GoogleNet_adv = []

    next(data_csv)
    for row in data_csv:
        step.append(str(float(row[0])))

        AlexNet_adv.append(float(row[1]))
        RMT_AlexNet_adv.append(float(row[2]))
        
        ResNet18_adv.append(float(row[3]))
        RMT_ResNet18_adv.append(float(row[4]))
        
        ResNet34_adv.append(float(row[5]))
        RMT_ResNet34_adv.append(float(row[6]))
        
        ResNet50_adv.append(float(row[7]))
        RMT_ResNet50_adv.append(float(row[8]))
        
        
        VGG19_adv.append(float(row[9]))
        RMT_VGG19_adv.append(float(row[10]))
        
        DenseNet169_adv.append(float(row[11]))
        RMT_DenseNet169_adv.append(float(row[12]))
        
        GoogleNet_adv.append(float(row[13]))
        RMT_GoogleNet_adv.append(float(row[14]))

#-------------ompgd--------
with open("svhn-ompgd-attack-strengths-20230402.csv",'r') as f:
    data_csv = csv.reader(f)

    step = []        # 0 0.02 0.05 0.1 0.2 0.3
    
    AlexNet_omadv = []
    RMT_AlexNet_omadv = []
    
    ResNet18_omadv = []    
    RMT_ResNet18_omadv = []    
    
    ResNet34_omadv = []
    RMT_ResNet34_omadv = []
    
    ResNet50_omadv = []    
    RMT_ResNet50_omadv = []    
    
    VGG19_omadv = []
    RMT_VGG19_omadv = []
    
    DenseNet169_omadv = []
    RMT_DenseNet169_omadv = []
    
    GoogleNet_omadv = []
    RMT_GoogleNet_omadv = []

    next(data_csv)
    for row in data_csv:
        step.append(str(float(row[0])))

        AlexNet_omadv.append(float(row[1]))
        RMT_AlexNet_omadv.append(float(row[2]))
        
        ResNet18_omadv.append(float(row[3]))
        RMT_ResNet18_omadv.append(float(row[4]))
        
        ResNet34_omadv.append(float(row[5]))
        RMT_ResNet34_omadv.append(float(row[6]))
        
        ResNet50_omadv.append(float(row[7]))
        RMT_ResNet50_omadv.append(float(row[8]))
        
        
        VGG19_omadv.append(float(row[9]))
        RMT_VGG19_omadv.append(float(row[10]))
        
        DenseNet169_omadv.append(float(row[11]))
        RMT_DenseNet169_omadv.append(float(row[12]))
        
        GoogleNet_omadv.append(float(row[13]))
        RMT_GoogleNet_omadv.append(float(row[14]))
#-----------------


#-------alexnet-------------
plt.plot(step, AlexNet_adv, label=f'Vanilla on PGD', marker='o', markersize=4, linestyle='-', color='b')
plt.plot(step, RMT_AlexNet_adv, label=f'LarepMixup (Ours) on PGD', marker='s', markersize=4, linestyle='-',color='r')

plt.plot(step, AlexNet_omadv, label=f'Vanilla on OM-PGD', marker='*', markersize=5, linestyle='--', color='b')
plt.plot(step, RMT_AlexNet_omadv, label=f'LarepMixup (Ours) on OM-PGD', marker='^', markersize=4, linestyle='--',color='r')

plt.legend([f'Vanilla on PGD', 
f'LarepMixup (Ours) on PGD',
f'Vanilla on OM-PGD',
f'LarepMixup (Ours) on OM-PGD',
],
fontsize = 8, 
loc='upper right') #   打出图例

plt.title(f'Accuracy of AlexNet on SVHN under Different Attack Budgets',fontsize=10, pad=12)
plt.xlabel('Budget',fontsize=10)
plt.ylabel('Top-1 Accuracy',fontsize=10)

plt.show()
savepath = f'/home/maggie/mmat/figure/attack-strength/20230403'
savename = f'svhn-Alexnet-pgd-ompgd-attack-strengths-20230403'
plt.savefig(f'{savepath}/{savename}.png')
plt.close()

#-------ResNet18-------------
# plt.plot(step,ResNet18_adv, label=f'ResNet18', marker='p', markersize=4)

plt.plot(step, ResNet18_adv, label=f'Vanilla on PGD', marker='o', markersize=4, linestyle='-', color='b')
plt.plot(step, RMT_ResNet18_adv, label=f'LarepMixup (Ours) on PGD', marker='s', markersize=4, linestyle='-',color='r')

plt.plot(step, ResNet18_omadv, label=f'Vanilla on OM-PGD', marker='*', markersize=5, linestyle='--', color='b')
plt.plot(step, RMT_ResNet18_omadv, label=f'LarepMixup (Ours) on OM-PGD', marker='^', markersize=4, linestyle='--',color='r')

plt.legend([f'Vanilla on PGD', 
f'LarepMixup (Ours) on PGD',
f'Vanilla on OM-PGD',
f'LarepMixup (Ours) on OM-PGD',
],
fontsize = 8, 
loc='upper right') #   打出图例

plt.title(f'Accuracy of ResNet18 on SVHN under Different Attack Budgets',fontsize=10, pad=12)
plt.xlabel('Budget',fontsize=10)
plt.ylabel('Top-1 Accuracy',fontsize=10)

plt.show()
savepath = f'/home/maggie/mmat/figure/attack-strength/20230403'
savename = f'svhn-ResNet18-pgd-attack-strengths-20230403'
plt.savefig(f'{savepath}/{savename}.png')
plt.close()

#-------ResNet34-------------
# plt.plot(step,ResNet34_adv, label=f'ResNet34', marker='p', markersize=4)

plt.plot(step, ResNet34_adv, label=f'Vanilla on PGD', marker='o', markersize=4, linestyle='-', color='b')
plt.plot(step, RMT_ResNet34_adv, label=f'LarepMixup (Ours) on PGD', marker='s', markersize=4, linestyle='-',color='r')

plt.plot(step, ResNet34_omadv, label=f'Vanilla on OM-PGD', marker='*', markersize=5, linestyle='--', color='b')
plt.plot(step, RMT_ResNet34_omadv, label=f'LarepMixup (Ours) on OM-PGD', marker='^', markersize=4, linestyle='--',color='r')

plt.legend([f'Vanilla on PGD', 
f'LarepMixup (Ours) on PGD',
f'Vanilla on OM-PGD',
f'LarepMixup (Ours) on OM-PGD',
],
fontsize = 8, 
loc='upper right') #   打出图例

plt.title(f'Accuracy of ResNet34 on SVHN under Different Attack Budgets',fontsize=10, pad=12)
plt.xlabel('Budget',fontsize=10)
plt.ylabel('Top-1 Accuracy',fontsize=10)

plt.show()
savepath = f'/home/maggie/mmat/figure/attack-strength/20230403'
savename = f'svhn-ResNet34-pgd-attack-strengths-20230403'
plt.savefig(f'{savepath}/{savename}.png')
plt.close()

#-------ResNet50-------------
# plt.plot(step,ResNet50_adv, label=f'ResNet50', marker='p', markersize=4)

plt.plot(step, ResNet50_adv, label=f'Vanilla on PGD', marker='o', markersize=4, linestyle='-', color='b')
plt.plot(step, RMT_ResNet50_adv, label=f'LarepMixup (Ours) on PGD', marker='s', markersize=4, linestyle='-',color='r')

plt.plot(step, ResNet50_omadv, label=f'Vanilla on OM-PGD', marker='*', markersize=5, linestyle='--', color='b')
plt.plot(step, RMT_ResNet50_omadv, label=f'LarepMixup (Ours) on OM-PGD', marker='^', markersize=4, linestyle='--',color='r')

plt.legend([f'Vanilla on PGD', 
f'LarepMixup (Ours) on PGD',
f'Vanilla on OM-PGD',
f'LarepMixup (Ours) on OM-PGD',
],
fontsize = 8, 
loc='upper right') #   打出图例

plt.title(f'Accuracy of ResNet50 on SVHN under Different Attack Budgets',fontsize=10, pad=12)
plt.xlabel('Budget',fontsize=10)
plt.ylabel('Top-1 Accuracy',fontsize=10)

plt.show()
savepath = f'/home/maggie/mmat/figure/attack-strength/20230403'
savename = f'svhn-ResNet50-pgd-attack-strengths-20230403'
plt.savefig(f'{savepath}/{savename}.png')
plt.close()

#-------VGG19-------------
# plt.plot(step,VGG19_adv, label=f'VGG19', marker='p', markersize=4)

plt.plot(step, VGG19_adv, label=f'Vanilla on PGD', marker='o', markersize=4, linestyle='-', color='b')
plt.plot(step, RMT_VGG19_adv, label=f'LarepMixup (Ours) on PGD', marker='s', markersize=4, linestyle='-',color='r')

plt.plot(step, VGG19_omadv, label=f'Vanilla on OM-PGD', marker='*', markersize=5, linestyle='--', color='b')
plt.plot(step, RMT_VGG19_omadv, label=f'LarepMixup (Ours) on OM-PGD', marker='^', markersize=4, linestyle='--',color='r')

plt.legend([f'Vanilla on PGD', 
f'LarepMixup (Ours) on PGD',
f'Vanilla on OM-PGD',
f'LarepMixup (Ours) on OM-PGD',
],
fontsize = 8, 
loc='upper right') #   打出图例

plt.title(f'Accuracy of VGG19 on SVHN under Different Attack Budgets',fontsize=10, pad=12)
plt.xlabel('Budget',fontsize=10)
plt.ylabel('Top-1 Accuracy',fontsize=10)

plt.show()
savepath = f'/home/maggie/mmat/figure/attack-strength/20230403'
savename = f'svhn-VGG19-pgd-attack-strengths-20230403'
plt.savefig(f'{savepath}/{savename}.png')
plt.close()

#-------DenseNet169-------------
# plt.plot(step,DenseNet169_adv, label=f'DenseNet169', marker='p', markersize=4)

plt.plot(step, DenseNet169_adv, label=f'Vanilla on PGD', marker='o', markersize=4, linestyle='-', color='b')
plt.plot(step, RMT_DenseNet169_adv, label=f'LarepMixup (Ours) on PGD', marker='s', markersize=4, linestyle='-',color='r')

plt.plot(step, DenseNet169_omadv, label=f'Vanilla on OM-PGD', marker='*', markersize=5, linestyle='--', color='b')
plt.plot(step, RMT_DenseNet169_omadv, label=f'LarepMixup (Ours) on OM-PGD', marker='^', markersize=4, linestyle='--',color='r')

plt.legend([f'Vanilla on PGD', 
f'LarepMixup (Ours) on PGD',
f'Vanilla on OM-PGD',
f'LarepMixup (Ours) on OM-PGD',
],
fontsize = 8, 
loc='upper right') #   打出图例

plt.title(f'Accuracy of DenseNet169 on SVHN under Different Attack Budgets',fontsize=10, pad=12)
plt.xlabel('Budget',fontsize=10)
plt.ylabel('Top-1 Accuracy',fontsize=10)

plt.show()
savepath = f'/home/maggie/mmat/figure/attack-strength/20230403'
savename = f'svhn-DenseNet169-pgd-attack-strengths-20230403'
plt.savefig(f'{savepath}/{savename}.png')
plt.close()

#-------GoogleNet-------------
# plt.plot(step,GoogleNet_adv, label=f'GoogleNet', marker='p', markersize=4)

plt.plot(step, GoogleNet_adv, label=f'Vanilla on PGD', marker='o', markersize=4, linestyle='-', color='b')
plt.plot(step, RMT_GoogleNet_adv, label=f'LarepMixup (Ours) on PGD', marker='s', markersize=4, linestyle='-',color='r')

plt.plot(step, GoogleNet_omadv, label=f'Vanilla on OM-PGD', marker='*', markersize=5, linestyle='--', color='b')
plt.plot(step, RMT_GoogleNet_omadv, label=f'LarepMixup (Ours) on OM-PGD', marker='^', markersize=4, linestyle='--',color='r')

plt.legend([f'Vanilla on PGD', 
f'LarepMixup (Ours) on PGD',
f'Vanilla on OM-PGD',
f'LarepMixup (Ours) on OM-PGD',
],
fontsize = 8, 
loc='upper right') #   打出图例

plt.title(f'Accuracy of GoogleNet on SVHN under Different Attack Budgets',fontsize=10, pad=12)
plt.xlabel('Budget',fontsize=10)
plt.ylabel('Top-1 Accuracy',fontsize=10)

plt.show()
savepath = f'/home/maggie/mmat/figure/attack-strength/20230403'
savename = f'svhn-GoogleNet-pgd-attack-strengths-20230403'
plt.savefig(f'{savepath}/{savename}.png')
plt.close()

# plt.plot(step,ResNet34_adv, label=f'ResNet34', marker='^', markersize=4)

# plt.plot(step,ResNet50_adv, label=f'ResNet50', marker='p', markersize=4)

# plt.plot(step,VGG19_adv, label=f'VGG19', marker='^', markersize=4)

# plt.plot(step,GoogleNet_adv, label=f'GoogleNet', marker='p', markersize=4)

# plt.plot(step,GoogleNet_adv, label=f'GoogleNet', marker='^', markersize=4)

# plt.legend([f'AlexNet', 
# f'ResNet18',
# f'ResNet34',
# f'ResNet50',
# f'VGG19',
# f'DenseNet169',
# f'GoogleNet'],
# fontsize = 8, 
# loc='upper right') #   打出图例

# plt.title(f'Accuracy of AlexNet on SVHN under Different Attack Budgets',fontsize=10, pad=12)
# plt.xlabel('Budget',fontsize=10)
# plt.ylabel('Top-1 Accuracy',fontsize=10)

# # x_major_locator=plt.MultipleLocator(2)          #   把x轴的刻度间隔设置为1，并存在变量里
# # y_major_locator=plt.MultipleLocator(0.02)       #   把y轴的刻度间隔设置为10，并存在变量里
# # ax=plt.gca()                                    #   ax为两条坐标轴的实例
# # ax.xaxis.set_major_locator(x_major_locator)     #   把x轴的主刻度设置为1的倍数
# # ax.yaxis.set_major_locator(y_major_locator)     #   把y轴的主刻度设置为10的倍数


# plt.show()
# savepath = f'/home/maggie/mmat/figure/attack-strength/20230403'
# savename = f'svhn-Alexnet-pgd-attack-strengths-20230403'
# plt.savefig(f'{savepath}/{savename}.png')
# plt.close()


