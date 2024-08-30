from tools_wzy import file_processing

import os
import shutil

import cv2
import numpy as np
import multiprocessing as mp

if __name__ == '__main__':
    mp.set_start_method('spawn')
    pool = mp.Pool(12)

    xml_folder = r'D:\Zhiyuan\video\record_all_finall\detect\Annotations'
    img_folder = r'D:\Zhiyuan\video\record_all_finall\detect\images'
    output_folder = r'D:\Zhiyuan\video\record_all_finall\classify_crop'
    x1, y1, x2, y2 = [2400, 500, 3300, 1200]
    position = y1, y2, x1, x2  # ymin, ymax, xmin, xmax flight = 850, 450, 1250, 750 x1, y1, x2, y2 = flight
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 根据xml文件裁图
    for filename in os.listdir(xml_folder):
        pool.apply_async(file_processing.crop_from_xml, args=(os.path.join(xml_folder, filename), img_folder, output_folder, '1'))

    # 根据位置裁图
    # for filename in os.listdir(img_folder):
    #     pool.apply_async(file_processing.crop_from_position, args=(os.path.join(img_folder, filename), position, output_folder, filename))

    pool.close()
    pool.join()
