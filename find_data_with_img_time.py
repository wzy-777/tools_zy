
import os
import shutil
from datetime import datetime

class rgb:
    def __init__(self, modificationTime, path):
        self.mTime = modificationTime
        self.path = path

def format_time(timestamp):
    ff = '%Y-%m-%d %H:%M:%S'
    ff = '%H:%M:%S'
    if isinstance(timestamp, (list, tuple)):
        res = [datetime.fromtimestamp(ti).strftime(ff) for ti in timestamp]
        return tuple(res)
    formatted_datetime = datetime.fromtimestamp(timestamp).strftime(ff)
    return formatted_datetime

def get_data_A_B_List(data_folder, data_format='.data'):
    Alist = []
    Blist = []
    for data in os.listdir(data_folder):
        if data.endswith(data_format):
            file_path = '/'.join([data_folder, data])
            modificationTime = os.path.getmtime(file_path)
            # print(f"{modificationTime}: {file_path}")
            if 'A' in data:
                Alist.append(rgb(modificationTime, file_path))
            elif 'B' in data:
                Blist.append(rgb(modificationTime, file_path))
    return Alist, Blist


def move_data_with_img_time(img_folder, Alist, Blist, img_format, data_format, operation=None):
    """根据bmp文件的创建时间，找对应的data文件。

    Args:
        img_folder (_type_): 含有bmp文件夹
        img_format: '.bmp'
        data_format: '.data'
        operation: None, 'move', 'copy'
    """
    not_ok_list = []
    bmp_num = [0, 0]
    for bmp in os.listdir(img_folder):
        if bmp.endswith(img_format):
            data_name = bmp.replace(img_format, data_format)
            new_file_path = os.path.join(img_folder, data_name)
            if os.path.exists(new_file_path):
                continue
            data_path = ''
            img_path = os.path.join(img_folder, bmp)
            modificationTime = os.path.getmtime(img_path)
            cnt = 0
            if 'A' in bmp:
                bmp_num[0] += 1
                for datai in Alist:
                    time_gap = abs(modificationTime - datai.mTime)
                    # print(time_gap)
                    if time_gap < 2.5:
                        data_path = datai.path
                        cnt += 1
            elif 'B' in bmp:
                bmp_num[1] += 1
                for datai in Blist:
                    time_gap = abs(modificationTime - datai.mTime)
                    # print(time_gap)
                    if time_gap < 2:
                        data_path = datai.path
                        cnt += 1
            if cnt > 1:
                not_ok_list.append(bmp)
                print(bmp, cnt, 'not ok')
            elif cnt == 1:
                if operation=='move':
                    print(data_path, '-->', new_file_path)
                    shutil.move(data_path, new_file_path)
                elif operation=='copy':
                    print(data_path, '-->', new_file_path)
                    shutil.copy(data_path, new_file_path)
            else:
                not_ok_list.append(bmp)
                print(bmp, cnt, 'not ok')
    print(bmp_num)
    if not_ok_list:
        print('=' * 99, 'not ok: ')
        for not_ok in not_ok_list:
            print(not_ok)
    
    
if __name__ == "__main__":
    """
    功能是根据图片的时间，找到时间相近的data，然后合并到一个文件夹，同时进行重命名。
    rgb_folder 里面是一系列的data
    folder_folder 里面的folder里面 是一系列的img

    """
    
    rgb_folder = r'D:\Zhiyuan\图片采集\data\外包采集图像\原始归档\20240820\rgb\data'
    folder_folder = r'D:\Zhiyuan\图片采集\data\外包采集图像\原始归档\20240820\夏季左侧'
    
    Alist, Blist = get_data_A_B_List(rgb_folder, '.data')
    print(len(Alist), len(Blist))
    for folder in os.listdir(folder_folder):
        folderi = os.path.join(folder_folder, folder)
        move_data_with_img_time(folderi, Alist, Blist, '.bmp', '.data', 'move')