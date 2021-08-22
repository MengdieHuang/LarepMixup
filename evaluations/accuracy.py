"""
Author: maggie
Date:   2021-06-15
Place:  Xidian University
@copyright
"""

import torch
import torch.utils.data

def EvaluateAccuracy(classifier,classify_loss,test_dataloader:torch.utils.data.DataLoader):
    
    #   计算准确率
    testset_total_num = len(test_dataloader.dataset)
    # print("flag B test set total_num:",testset_total_num)                                                       #   test set total_num: 10000 样本总数
    # print("flag B test_dataloader.len:",len(test_dataloader))                                                   #   test_dataloader.len: 40 batch数目（batchsize256）

    # print("flag B test_dataloader.dataset[0][0][0]:",test_dataloader.dataset[0][0][0])                                #   归一化的
    # print("flag B test_dataloader.dataset[0][1]:",test_dataloader.dataset[0][1])                                  #   
    # print("flag B test_dataloader.dataset[1][1]:",test_dataloader.dataset[1][1])                                  #   
    # print("flag B test_dataloader.dataset[2][1]:",test_dataloader.dataset[2][1])                                  #   

    epoch_correct_num = 0
    epoch_total_loss = 0
    
    # print("test_dataloader.dataset.data:",test_dataloader.dataset.data[:2])

    for batch_index,(images, labels) in enumerate(test_dataloader):
        # if batch_index == 0:
        #     print("flag B images[0][0]:",images[0][0])                                                                       #     images.shape: torch.Size([256, 3, 32, 32])
        #     print("flag B labels[0]:",labels[0])                                                                         #   labels.shape: torch.Size([256])

        imgs = images.cuda()
        labs = labels.cuda()
        
        # if batch_index == 0:
        #     print("flag B imgs[0][0]:",images[0][0])                                                                        #   归一化的 但和test_dataloader.dataset[0][0]不一样
        #     print("flag B labs[0]:",labs[0])                                                                             #   labs.shape: torch.Size([256])


        with torch.no_grad():
            output = classifier(imgs)
            # print("output:",output)
            loss = classify_loss(output,labs)
            _, predicted_label_index = torch.max(output.data, 1)    #torch.max()这个函数返回的是两个值，第一个值是具体的value，第二个值是value所在的index   
            # print("predicted_label_index:",predicted_label_index)                                                           #   predicted_label_index: tensor([1, 4, 0, 6, 0, 0, 7, 8, 0, 3, 3, 0, 7, 4, 9, 3], device='cuda:0')
            # print("labs:",labs)                                                                                             #   labs: tensor([1, 6, 8, 2, 8, 8, 5, 0, 1, 1, 9, 0, 7, 4, 1, 2], device='cuda:0')
            batch_same_num = (predicted_label_index == labs).sum().item()
            epoch_correct_num += batch_same_num
            epoch_total_loss += loss

    #------当前epoch分类模型在测试集整体上的准确率--------------------(测试集只用1轮)
    test_accuracy = epoch_correct_num / testset_total_num
    test_loss = epoch_total_loss / len(test_dataloader)                                                         #   除以batch num
    # print("测试样本总数：",testset_total_num)
    # print("预测正确总数：",epoch_correct_num)
    # print("预测总损失：",epoch_total_loss)
    return test_accuracy,test_loss 
    