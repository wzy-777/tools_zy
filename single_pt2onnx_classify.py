from tools_wzy import model_run
import os
import multiprocessing
import torch.nn as nn
import torch.nn.functional as F




# 创建子进程必须添加这一行这个
if __name__ == '__main__':
    multiprocessing.freeze_support()

# ========================================================================================================================== #
    # 把pt转为onnx
    pt_path = r"D:\Zhiyuan\video\record_all_finall\classify\crop\model\v2\vest_enb0_C3_E20_0.00483_0.99906_0.01215_0.99811_noNM_softmax.pt"
    model_run.pt2onnx_for_classify(pt_path)

    # 多个用这个，选择最好测试精度和最低测试损失的
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
    #         model_run.pt2onnx_for_classify(os.path.join(data_folder, best_pt))
