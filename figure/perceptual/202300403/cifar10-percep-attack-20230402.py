import matplotlib.pyplot as plt
import csv
import numpy as np

plt.switch_backend('agg')

# 读取不同模型在不同扰动强度下的cifar10 PGD准确率
with open("cifar10-perceptual-attack-20230402.csv",'r') as f:
    data_csv = csv.reader(f)

    step = []        # clean Fog Snow Elastic JPEG

    
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
        # step.append(str(float(row[0])))
        step.append(str(row[0]))

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

# #-------------ompgd--------
# with open("cifar10-ompgd-attack-strengths-20230402.csv",'r') as f:
#     data_csv = csv.reader(f)

#     step = []        # 0 0.02 0.05 0.1 0.2 0.3
    
#     AlexNet_omadv = []
#     RMT_AlexNet_omadv = []
    
#     ResNet18_omadv = []    
#     RMT_ResNet18_omadv = []    
    
#     ResNet34_omadv = []
#     RMT_ResNet34_omadv = []
    
#     ResNet50_omadv = []    
#     RMT_ResNet50_omadv = []    
    
#     VGG19_omadv = []
#     RMT_VGG19_omadv = []
    
#     DenseNet169_omadv = []
#     RMT_DenseNet169_omadv = []
    
#     GoogleNet_omadv = []
#     RMT_GoogleNet_omadv = []

#     next(data_csv)
#     for row in data_csv:
#         step.append(str(float(row[0])))

#         AlexNet_omadv.append(float(row[1]))
#         RMT_AlexNet_omadv.append(float(row[2]))
        
#         ResNet18_omadv.append(float(row[3]))
#         RMT_ResNet18_omadv.append(float(row[4]))
        
#         ResNet34_omadv.append(float(row[5]))
#         RMT_ResNet34_omadv.append(float(row[6]))
        
#         ResNet50_omadv.append(float(row[7]))
#         RMT_ResNet50_omadv.append(float(row[8]))
        
        
#         VGG19_omadv.append(float(row[9]))
#         RMT_VGG19_omadv.append(float(row[10]))
        
#         DenseNet169_omadv.append(float(row[11]))
#         RMT_DenseNet169_omadv.append(float(row[12]))
        
#         GoogleNet_omadv.append(float(row[13]))
#         RMT_GoogleNet_omadv.append(float(row[14]))
# #-----------------


#-------alexnet-------------
size = 5
x = np.arange(size)
# print(x) #[0 1 2 3 4]
total_width, n = 0.5, 2
width = total_width / n #0.25

plt.bar(x-(width/2) , AlexNet_adv, label=f'Vanilla', width=width)
plt.bar(x+(width/2), RMT_AlexNet_adv, label=f'LarepMixup (Ours)', width=width)
x_labels = step
plt.xticks(x, x_labels)

# 功能2
for i, j in zip(x, AlexNet_adv):
    plt.text(i-(width/2), j + 0.02, "%.2f" % j, ha="center", va="bottom", fontsize=6)
for i, j in zip(x, RMT_AlexNet_adv):
    plt.text(i+(width/2), j + 0.02, "%.2f" % j, ha="center", va="bottom", fontsize=6)

plt.legend([f'Vanilla', 
f'LarepMixup (Ours)',
],
fontsize = 8, 
loc='upper right') #   打出图例

plt.title(f'Accuracy of AlexNet on CIFAR-10 under Different Perceptual Attacks',fontsize=10, pad=12)
plt.xlabel('Epsilon',fontsize=10)
plt.ylabel('Top-1 Accuracy',fontsize=10)

plt.show()
savepath = f'/home/maggie/mmat/figure/perceptual/202300403'
savename = f'cifar10-Alexnet-perceptual-attack-20230403'
plt.savefig(f'{savepath}/{savename}.png')
plt.close()

#-------ResNet18-------------
size = 5
x = np.arange(size)
# print(x) #[0 1 2 3 4]
total_width, n = 0.5, 2
width = total_width / n #0.25

plt.bar(x-(width/2) , ResNet18_adv, label=f'Vanilla', width=width)
plt.bar(x+(width/2), RMT_ResNet18_adv, label=f'LarepMixup (Ours)', width=width)
x_labels = step
plt.xticks(x, x_labels)

# 功能2
for i, j in zip(x, ResNet18_adv):
    plt.text(i-(width/2), j + 0.02, "%.2f" % j, ha="center", va="bottom", fontsize=6)
for i, j in zip(x, RMT_ResNet18_adv):
    plt.text(i+(width/2), j + 0.02, "%.2f" % j, ha="center", va="bottom", fontsize=6)

plt.legend([f'Vanilla', 
f'LarepMixup (Ours)',
],
fontsize = 8, 
loc='upper right') #   打出图例

plt.title(f'Accuracy of ResNet18 on CIFAR-10 under Different Perceptual Attacks',fontsize=10, pad=12)
plt.xlabel('Epsilon',fontsize=10)
plt.ylabel('Top-1 Accuracy',fontsize=10)

plt.show()
savepath = f'/home/maggie/mmat/figure/perceptual/202300403'
savename = f'cifar10-ResNet18-perceptual-attack-20230403'
plt.savefig(f'{savepath}/{savename}.png')
plt.close()

#-------ResNet34-------------
size = 5
x = np.arange(size)
# print(x) #[0 1 2 3 4]
total_width, n = 0.5, 2
width = total_width / n #0.25

plt.bar(x-(width/2) , ResNet34_adv, label=f'Vanilla', width=width)
plt.bar(x+(width/2), RMT_ResNet34_adv, label=f'LarepMixup (Ours)', width=width)
x_labels = step
plt.xticks(x, x_labels)

# 功能2
for i, j in zip(x, ResNet34_adv):
    plt.text(i-(width/2), j + 0.02, "%.2f" % j, ha="center", va="bottom", fontsize=6)
for i, j in zip(x, RMT_ResNet34_adv):
    plt.text(i+(width/2), j + 0.02, "%.2f" % j, ha="center", va="bottom", fontsize=6)

plt.legend([f'Vanilla', 
f'LarepMixup (Ours)',
],
fontsize = 8, 
loc='upper right') #   打出图例

plt.title(f'Accuracy of ResNet34 on CIFAR-10 under Different Perceptual Attacks',fontsize=10, pad=12)
plt.xlabel('Epsilon',fontsize=10)
plt.ylabel('Top-1 Accuracy',fontsize=10)

plt.show()
savepath = f'/home/maggie/mmat/figure/perceptual/202300403'
savename = f'cifar10-ResNet34-perceptual-attack-20230403'
plt.savefig(f'{savepath}/{savename}.png')
plt.close()

#-------ResNet50-------------
size = 5
x = np.arange(size)
# print(x) #[0 1 2 3 4]
total_width, n = 0.5, 2
width = total_width / n #0.25

plt.bar(x-(width/2) , ResNet50_adv, label=f'Vanilla', width=width)
plt.bar(x+(width/2), RMT_ResNet50_adv, label=f'LarepMixup (Ours)', width=width)
x_labels = step
plt.xticks(x, x_labels)

# 功能2
for i, j in zip(x, ResNet50_adv):
    plt.text(i-(width/2), j + 0.02, "%.2f" % j, ha="center", va="bottom", fontsize=6)
for i, j in zip(x, RMT_ResNet50_adv):
    plt.text(i+(width/2), j + 0.02, "%.2f" % j, ha="center", va="bottom", fontsize=6)

plt.legend([f'Vanilla', 
f'LarepMixup (Ours)',
],
fontsize = 8, 
loc='upper right') #   打出图例

plt.title(f'Accuracy of ResNet50 on CIFAR-10 under Different Perceptual Attacks',fontsize=10, pad=12)
plt.xlabel('Epsilon',fontsize=10)
plt.ylabel('Top-1 Accuracy',fontsize=10)

plt.show()
savepath = f'/home/maggie/mmat/figure/perceptual/202300403'
savename = f'cifar10-ResNet50-perceptual-attack-20230403'
plt.savefig(f'{savepath}/{savename}.png')
plt.close()

#-------VGG19-------------
size = 5
x = np.arange(size)
# print(x) #[0 1 2 3 4]
total_width, n = 0.5, 2
width = total_width / n #0.25

plt.bar(x-(width/2) , VGG19_adv, label=f'Vanilla', width=width)
plt.bar(x+(width/2), RMT_VGG19_adv, label=f'LarepMixup (Ours)', width=width)
x_labels = step
plt.xticks(x, x_labels)

# 功能2
for i, j in zip(x, VGG19_adv):
    plt.text(i-(width/2), j + 0.02, "%.2f" % j, ha="center", va="bottom", fontsize=6)
for i, j in zip(x, RMT_VGG19_adv):
    plt.text(i+(width/2), j + 0.02, "%.2f" % j, ha="center", va="bottom", fontsize=6)

plt.legend([f'Vanilla', 
f'LarepMixup (Ours)',
],
fontsize = 8, 
loc='upper right') #   打出图例

plt.title(f'Accuracy of VGG19 on CIFAR-10 under Different Perceptual Attacks',fontsize=10, pad=12)
plt.xlabel('Epsilon',fontsize=10)
plt.ylabel('Top-1 Accuracy',fontsize=10)

plt.show()
savepath = f'/home/maggie/mmat/figure/perceptual/202300403'
savename = f'cifar10-VGG19-perceptual-attack-20230403'
plt.savefig(f'{savepath}/{savename}.png')
plt.close()

#-------DenseNet169-------------
size = 5
x = np.arange(size)
# print(x) #[0 1 2 3 4]
total_width, n = 0.5, 2
width = total_width / n #0.25

plt.bar(x-(width/2) , DenseNet169_adv, label=f'Vanilla', width=width)
plt.bar(x+(width/2), RMT_DenseNet169_adv, label=f'LarepMixup (Ours)', width=width)
x_labels = step
plt.xticks(x, x_labels)

# 功能2
for i, j in zip(x, DenseNet169_adv):
    plt.text(i-(width/2), j + 0.02, "%.2f" % j, ha="center", va="bottom", fontsize=6)
for i, j in zip(x, RMT_DenseNet169_adv):
    plt.text(i+(width/2), j + 0.02, "%.2f" % j, ha="center", va="bottom", fontsize=6)

plt.legend([f'Vanilla', 
f'LarepMixup (Ours)',
],
fontsize = 8, 
loc='upper right') #   打出图例

plt.title(f'Accuracy of DenseNet169 on CIFAR-10 under Different Perceptual Attacks',fontsize=10, pad=12)
plt.xlabel('Epsilon',fontsize=10)
plt.ylabel('Top-1 Accuracy',fontsize=10)

plt.show()
savepath = f'/home/maggie/mmat/figure/perceptual/202300403'
savename = f'cifar10-DenseNet169-perceptual-attack-20230403'
plt.savefig(f'{savepath}/{savename}.png')
plt.close()

#-------GoogleNet-------------
size = 5
x = np.arange(size)
# print(x) #[0 1 2 3 4]
total_width, n = 0.5, 2
width = total_width / n #0.25

plt.bar(x-(width/2) , GoogleNet_adv, label=f'Vanilla', width=width)
plt.bar(x+(width/2), RMT_GoogleNet_adv, label=f'LarepMixup (Ours)', width=width)
x_labels = step
plt.xticks(x, x_labels)

# 功能2
for i, j in zip(x, GoogleNet_adv):
    plt.text(i-(width/2), j + 0.02, "%.2f" % j, ha="center", va="bottom", fontsize=6)
for i, j in zip(x, RMT_GoogleNet_adv):
    plt.text(i+(width/2), j + 0.02, "%.2f" % j, ha="center", va="bottom", fontsize=6)

plt.legend([f'Vanilla', 
f'LarepMixup (Ours)',
],
fontsize = 8, 
loc='upper right') #   打出图例

plt.title(f'Accuracy of GoogleNet on CIFAR-10 under Different Perceptual Attacks',fontsize=10, pad=12)
plt.xlabel('Epsilon',fontsize=10)
plt.ylabel('Top-1 Accuracy',fontsize=10)

plt.show()
savepath = f'/home/maggie/mmat/figure/perceptual/202300403'
savename = f'cifar10-GoogleNet-perceptual-attack-20230403'
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

# plt.title(f'Accuracy of AlexNet on CIFAR-10 under Different Attack Budgets',fontsize=10, pad=12)
# plt.xlabel('Epsilon',fontsize=10)
# plt.ylabel('Top-1 Accuracy',fontsize=10)

# # x_major_locator=plt.MultipleLocator(2)          #   把x轴的刻度间隔设置为1，并存在变量里
# # y_major_locator=plt.MultipleLocator(0.02)       #   把y轴的刻度间隔设置为10，并存在变量里
# # ax=plt.gca()                                    #   ax为两条坐标轴的实例
# # ax.xaxis.set_major_locator(x_major_locator)     #   把x轴的主刻度设置为1的倍数
# # ax.yaxis.set_major_locator(y_major_locator)     #   把y轴的主刻度设置为10的倍数


# plt.show()
# savepath = f'/home/maggie/mmat/figure/attack-strength/20230403'
# savename = f'cifar10-Alexnet-pgd-attack-strengths-20230403'
# plt.savefig(f'{savepath}/{savename}.png')
# plt.close()


