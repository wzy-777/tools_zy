import os

from tools_wzy import data_need

if __name__ == '__main__':
    # 根据txt文件重命名图片，给图片写上描述，方便人员标注
    pics_folder = r"D:\Zhiyuan\图片采集\20240805 - 副本"
    txt_path = r"D:\Zhiyuan\work_tools_py2\temp.txt"
    # data_need.rename_pics_with_txt(pics_folder, txt_path, '.bmp', 2)

    # 标好一个文件之后，将其复制到对应所有图片，这样就相当于给所有图片粗略标了一次，然后修改位置就行了。
    pics_folder = r"D:\Zhiyuan\图片采集\20240805 - 副本"
    jsonA_path = r"D:\Zhiyuan\图片采集\20240805_copy\A.json"
    jsonB_path = r"D:\Zhiyuan\图片采集\20240805_copy\B.json"
    # data_need.json_one_2_more(pics_folder, jsonA_path, jsonB_path)

    # 划分图片和json文件到train, val, test文件夹，同时修改json文件为coco需要的格式
    labelme_folder = r"D:\Zhiyuan\图片采集\20240805_copy2\A"
    ratio = (0.85, 0.1, 0.05)
    fixed_bbox = [1, 50, 510, 1022]
    # data_folder = data_need.split_labelme(labelme_folder, ratio)
    # for datai in ['train', 'val', 'test']:
    #     data_need.labelmes2coco(os.path.join(data_folder, datai), rename=True, fixed_bbox=fixed_bbox)

