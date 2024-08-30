import time
from tools_wzy import model_run
import os
import sys

import multiprocessing
# 创建子进程必须添加这一行这个
import torch.nn as nn
import torch.nn.functional as F
class CustomModel(nn.Module):
    def __init__(self, pretrained_model):
        super(CustomModel, self).__init__()
        self.pretrained_model = pretrained_model

    def forward(self, x):
        outputs = self.pretrained_model(x)
        probabilities = F.softmax(outputs, dim=1)
        return probabilities

if __name__ == '__main__':
    multiprocessing.freeze_support()

    # 单个move
    model_path = r"D:\Zhiyuan\pic_data_20231229\YGC\HUZ\20240201\YGC_enb0_C3_E23_0.00480_0.99809_0.06319_0.98830.pt"
    data_folder = r"D:\Program Files\wxwork\WXWork\1688857814425476\Cache\File\2024-03\HUZ_YGC\1"
    # model_run.classify_move(model_path, data_folder, image_size=224, batch_size=64, conf_threshold=0.9, num_workers=4, noNM=True)

    # 单个check
    model_path = r"D:\Zhiyuan\models\xie\CSC_PTC_PBC_enb0_C4_E48_0.00079_0.99970_0.01056_0.99878.pt"
    data_folder = r"\\192.168.102.206\share\pic_data\CSC_PTC_PBC\PVG\20240514"
    model_run.classify_check(model_path, data_folder, image_size=224, batch_size=32, conf_threshold=0.9, num_workers=2, noNM=False)

    # 多个check，选择最好测试精度和最低测试损失的
    data_folder_folder = r"D:\Program Files\wxwork\WXWork\1688857814425476\Cache\File\2023-12\nanyang_20231212"
    # for folder in os.listdir(data_folder_folder):
    #     data_folder = os.path.join(data_folder_folder, folder)
    #     print("==========================")
    #     print(data_folder)
    #     best_test_acc = 0
    #     best_test_loss = 100
    #     best_pt = ""
    #     for file in os.listdir(data_folder):
    #         if file.endswith('.pt'):
    #             ptsplit = file.split('.pt')[0].split('_')
    #             test_acc = float(ptsplit[-1])
    #             test_loss = float(ptsplit[-2])
    #             if test_acc > best_test_acc:
    #                 best_test_acc = test_acc
    #                 best_test_loss = test_loss
    #                 best_pt = file
    #             elif test_acc == best_test_acc:
    #                 if best_test_loss < test_loss:
    #                     best_test_loss = test_loss
    #                     best_pt = file
    #     if best_pt is not None:
    #         print(best_pt)
    #         model_run.classify_check(os.path.join(data_folder, best_pt), data_folder, image_size=224, batch_size=64, conf_threshold=0.9, num_workers=4)



