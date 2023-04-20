import matplotlib.pyplot as plt
import csv

plt.switch_backend('agg')

with open("run-rmt-cifar10-dataset_tensorboard-log-epoch_cost_time-tag-epoch_cost_time.csv",'r') as f:
    data_csv = csv.reader(f)

    # step = []        # 0 0.02 0.05 0.1 0.2 0.3
    cur_epoch_cost_time=[]

    next(data_csv)
    for row in data_csv:
        # step.append(str(float(row[1])))
        cur_epoch_cost_time.append(float(row[2]))

with open("dmat-run-tensorboard-log-epoch_cost_time-tag-epoch_cost_time.csv",'r') as f:
    data_csv = csv.reader(f)

    # step = []        # 0 0.02 0.05 0.1 0.2 0.3
    dmat_cur_epoch_cost_time=[]

    next(data_csv)
    for row in data_csv:
        # step.append(str(float(row[1])))
        dmat_cur_epoch_cost_time.append(float(row[2]))
                
#-------alexnet-------------
step = list(range(len(cur_epoch_cost_time)))

plt.plot(step, cur_epoch_cost_time, label=f'LarepMixup (Ours)', linestyle='-', color='b')
plt.plot(step, dmat_cur_epoch_cost_time, label=f'DMAT', linestyle='-', color='r')


# plt.plot(step, RMT_AlexNet_adv, label=f'LarepMixup (Ours) on PGD', marker='*', markersize=4, linestyle='-',color='r')

# plt.plot(step, AlexNet_omadv, label=f'Vanilla on OM-PGD', marker='o', markersize=4, linestyle='--', color='b')
# plt.plot(step, RMT_AlexNet_omadv, label=f'LarepMixup (Ours) on OM-PGD', marker='s', markersize=4, linestyle='--',color='r')

# plt.legend()

plt.legend([
f'LarepMixup (Ours)', 
f'DMAT',
# f'DMAT',
# f'LarepMixup (Ours) on OM-PGD',
],
fontsize = 8, 
loc='upper right') #   打出图例

plt.title(f'Each Epoch Cost Time for Robust Training WideResNet28-10',fontsize=10, pad=12)
plt.xlabel('Epoch Index',fontsize=10)
plt.ylabel('Cost Time (seconds)',fontsize=10)

plt.show()
savepath = f'/home/maggie/mmat/figure/costtime'
savename = f'cifar10-wideresnet-rmt-epoch-costtime-compare-20230403'
plt.savefig(f'{savepath}/{savename}.png')
plt.close()



















# #----------------------------------------------------------------

# # 读取不同模型在不同扰动强度下的cifar10 PGD准确率
# with open("run-rmt-cifar10-dataset_tensorboard-log-epoch_total_epo_cost_time-tag-epoch_total_epo_cost_time.csv",'r') as f:
#     data_csv = csv.reader(f)

#     step = []        # 0 0.02 0.05 0.1 0.2 0.3
#     stack_epoch_cost_time=[]

#     next(data_csv)
#     for row in data_csv:
#         step.append(str(float(row[1])))
#         stack_epoch_cost_time.append(float(row[2]))
        

# step = list(range(len(cur_epoch_cost_time)))


# plt.plot(step, stack_epoch_cost_time, linestyle='-', color='b')

# # plt.plot(step, RMT_AlexNet_adv, label=f'LarepMixup (Ours) on PGD', marker='*', markersize=4, linestyle='-',color='r')

# # plt.plot(step, AlexNet_omadv, label=f'Vanilla on OM-PGD', marker='o', markersize=4, linestyle='--', color='b')
# # plt.plot(step, RMT_AlexNet_omadv, label=f'LarepMixup (Ours) on OM-PGD', marker='s', markersize=4, linestyle='--',color='r')

# # plt.legend()

# # plt.legend([f'Vanilla on PGD', 
# # f'LarepMixup (Ours) on PGD',
# # f'Vanilla on OM-PGD',
# # f'LarepMixup (Ours) on OM-PGD',
# # ],
# # fontsize = 8, 
# # loc='upper right') #   打出图例

# plt.title(f'Total Cost Time for LarepMixup Training WideResNet28-10',fontsize=10, pad=12)
# plt.xlabel('Epochs',fontsize=10)
# plt.ylabel('Cost Time (seconds)',fontsize=10)

# plt.show()
# savepath = f'/home/maggie/mmat/figure/costtime'
# savename = f'cifar10-wideresnet-rmt-total-costtime-20230403'
# plt.savefig(f'{savepath}/{savename}.png')
# plt.close()