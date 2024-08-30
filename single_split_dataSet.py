from tools_wzy import yolo_need
from tools_wzy import model_run
import multiprocessing
import os


def split_folder_folder_classify_dataSet():
    # 划分分类数据集，如果一个文件夹中有多个事件的文件夹，类似于：/ningbo20231113/PCC用这个。
    folder_folder = r"D:\Program Files\wxwork\WXWork\1688857814425476\Cache\File\2023-12\nanyang_20231212"
    test_ratio = 0.15
    for folder in os.listdir(folder_folder):
        top_path = os.path.join(folder_folder, folder)
        num_class = model_run.check_sequential_folders(top_path)
        print(top_path)
        print(num_class)
        model_run.split_classify_train_test(top_path, test_ratio=test_ratio, num_class=num_class)


def split_classify_dataSet():
    # 划分分类数据集
    top_path = r"D:\Zhiyuan\pic_data_20231229\HCM\MFM\20240408"
    num_class = model_run.check_sequential_folders(top_path)
    test_ratio = 0.2
    model_run.split_classify_train_test(top_path, test_ratio=test_ratio, num_class=num_class)


def split_detect_dataSet():
    # 划分yolo数据集
    xmls_folder = r'D:\Zhiyuan\图片采集\data\外包采集图像\zhaoenbo\Annotations'
    # class_list = ["1HCM", "2FJ", "3LQ", "4XLC", "5PBC", "6QYC", "7CSC", "8JYC", "9CBC", "10DYC", "11LJC", "12ZT", "13REN", "14PCC",
    #               "15QWSC", "16PTC"]
    # class_list = ['FJ', 'person', 'cone', 'KTC', 'QYC',
    #               'CSC', 'PTC', 'JYC', 'KCM', 'HCM',
    #               'LQ', 'PCC', 'workladder', 'XLC', 'PBC',
    #               'YDC', 'other']
    class_list = ['shouji', 'diexingbaozhawu', 'fenmobaozhawu', 'juxingbaozhawu', 'taocidao', 'jinshudao', 'dahuoji', 'yeti', 'zidan', 'qiang', 'luyinbi', 'fangxingluyinbi', 'xiaoluosidao', 'zhijiadao', 'zhusheqi', 'banshou', 'yanhe', 'zhibi', 'jinshuUpan', 'suliaoUpan', 'xisheng', 'shexiangtou']
    # 7, 2, 1
    trainval = 0.9
    train = 0.777
    data_yaml_name = 'all.yaml'

    ylabels_folder = yolo_need.xmls_to_yolo(xmls_folder, class_list)  # 把所有xml文件转为txt的label
    dataSet_folder = yolo_need.split_train_val(xmls_folder, trainval, train)  # 根据xml文件划分图片数据集
    yaml_path = yolo_need.gen_yaml(dataSet_folder, class_list, data_yaml_name)  # 产生对应data的yaml文件
    # yaml_path_wsl = file_to_wsl(yaml_path)  # 产生对应wsl的文件


if __name__ == '__main__':
    # split_folder_folder_classify_dataSet()
    # split_classify_dataSet()
    split_detect_dataSet()
    pass
