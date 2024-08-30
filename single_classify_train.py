import datetime
from tools_wzy import model_run
import multiprocessing
import time
import os


# 创建子进程必须添加这一行这个
if __name__ == '__main__':

    event_name = "vest"
    data_folder = r"D:\Zhiyuan\models\record_vest"
    model_out_folder = r"D:\Zhiyuan\models\record_vest\train_result"
    model_run.classify_train(event_name, data_folder, model_out_folder, epochs=60, batch_size=16, lr=9e-5, weight_decay=1e-5,
                             log_txt="log.txt", num_workers=8, max_no_better=20, noNM=False)
    # model_run.classify_train(event_name, data_folder, model_out_folder, epochs=60, batch_size=16, lr=9e-5, weight_decay=1e-5,
    #                          log_txt="log.txt", num_workers=4, max_no_better=20, noNM=True)
    # model_run.classify_train_v2(event_name, data_folder, model_out_folder, epochs=10, batch_size=32, lr=2e-4, weight_decay=1e-5,
    #                          log_txt="log.txt", num_workers=8)



    # 运行多个训练，可以用下面这个
    # folder_folder = r"D:\Program Files\wxwork\WXWork\1688857814425476\Cache\File\2023-12\nanyang_20231212"
    # for folder in os.listdir(folder_folder):
    #     print("当前时间：", datetime.datetime.now().strftime("%H:%M:%S"))
    #     data_folder = os.path.join(folder_folder, folder)
    #     # 参数如下
    #     event_name = folder
    #     num_class = model_run.check_sequential_folders(data_folder + '/train')
    #     model_out_folder = data_folder
    #     print(f'event_name: {event_name}\nnum_class: {num_class}\ndata_folder: {data_folder}\nmodel_out_folder: {model_out_folder}')
    #     model_path = model_run.classify_train(event_name, data_folder, model_out_folder, epochs=80, batch_size=16, lr=2e-4,
    #                              weight_decay=1e-5, log_txt="log.txt", num_workers=4)
        # 这里不推荐自动检查最后一个，推荐自己筛选过后再检查
        # model_run.classify_check(model_path, data_folder, image_size=224, batch_size=64, conf_threshold=0.9, num_workers=4)
