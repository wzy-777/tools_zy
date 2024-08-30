import os
import shutil

def categorize_files(folder_path):
    images_folder = os.path.join(folder_path, 'images')
    annotations_folder = os.path.join(folder_path, 'Annotations')

    # 创建子文件夹
    os.makedirs(images_folder, exist_ok=True)
    os.makedirs(annotations_folder, exist_ok=True)

    # 遍历文件夹中的所有文件
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)

        if file_name.endswith('.xml'):
            # 如果是.xml文件，则移动到Annotations文件夹
            shutil.move(file_path, os.path.join(annotations_folder, file_name))
        elif os.path.isfile(file_path):
            # 如果是图片文件，则移动到images文件夹
            shutil.move(file_path, os.path.join(images_folder, file_name))

# 例子用法
folder_to_categorize = r'D:\Program Files\wechat\WeChat Files\wxid_6n8fdu2n730n22\FileStorage\File\2024-01\Finall_20230107\PVG_20230701\2000\2000\all'
# categorize_files(folder_to_categorize)


