import os
import shutil
import hashlib
from PIL import Image
# from skimage.metrics import structural_similarity
import cv2
import numpy as np
import copy
import sys
import multiprocessing as mp
import xml.etree.ElementTree as ET
import json
import pandas as pd


# 移动所有图片到指定文件夹
def move_images(source_folder, destination_folder, file_format=None):
    # 遍历指定文件夹中的所有子文件夹
    for root, dirs, files in os.walk(source_folder):
        for dir_name in dirs:
            # 拼接子文件夹的路径
            dir_path = os.path.join(root, dir_name)
            # 遍历子文件夹中的所有文件
            for file_name in os.listdir(dir_path):
                # 获取文件的完整路径
                file_path = os.path.join(dir_path, file_name)
                # 判断是否是图片文件
                if file_format is None:
                    file_format = ('.png', '.jpg', '.jpeg', '.gif', '.bmp')
                if file_name.lower().endswith(file_format):
                    # 移动文件
                    try:
                        shutil.move(file_path, destination_folder)
                    except OSError:
                        # 目标路径已经存在同名文件，修改文件名并重试
                        path_basename, ext = os.path.splitext(file_path)
                        index = 1
                        while True:
                            new_file_path = f"{path_basename}_{index}{ext}"
                            if not os.path.exists(new_file_path):
                                shutil.move(file_path, new_file_path)
                                break
                            index += 1

# 将超多的图片平均划分到多个文件夹中
def split_pics_to_folders(src_dir, dst_dir, x):
    img_extensions = ['.jpg']
    img_files = []
    for root, dirs, files in os.walk(src_dir):
        for file in files:
            img_files.append(os.path.join(root, file))

    # 将图像文件按顺序分组，每组包含 x 个图像文件
    img_groups = [img_files[i:i + x] for i in range(0, len(img_files), x)]

    # 在目标文件夹中创建子文件夹并移动图像文件
    for i, group in enumerate(img_groups):
        # 创建子文件夹
        folder_name = "f_{}".format(i + 1)
        folder_path = os.path.join(dst_dir, folder_name)
        os.makedirs(folder_path)

        # 移动图像文件到子文件夹中
        for file_path in group:
            file_name = os.path.basename(file_path)
            dst_file_path = os.path.join(folder_path, file_name)
            shutil.move(file_path, dst_file_path)

# 删除 folder2 中与 folder1 重复的图像文件
def remove_same_pic(folder1, folder2_for_remove):
    # 获取 folder1 中所有图像文件的哈希值
    folder1_images_hashes = {}
    for filename in os.listdir(folder1):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            with open(os.path.join(folder1, filename), 'rb') as f:
                hash_value = hashlib.md5(f.read()).hexdigest()
            folder1_images_hashes[hash_value] = filename

    # 删除 folder2 中与 folder1 重复的图像文件
    deleted_count = 0
    for filename in os.listdir(folder2_for_remove):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            with open(os.path.join(folder2_for_remove, filename), 'rb') as f:
                hash_value = hashlib.md5(f.read()).hexdigest()
            if hash_value in folder1_images_hashes:
                os.remove(os.path.join(folder2_for_remove, filename))
                deleted_count += 1

    print(f'Deleted {deleted_count} duplicate images from {folder2_for_remove}.')

# 删除尺寸小的图片
def delete_small_pics(folder_path, min_width, min_height):
    # 存储要删除的文件路径列表
    to_delete = []
    # 遍历文件夹及其子文件夹中所有图片文件
    for root, dirs, files in os.walk(folder_path):
        for file_name in files:
            if file_name.lower().endswith((".jpg", ".jpeg", ".png")):
                file_path = os.path.join(root, file_name)
                try:
                    # 打开并读取文件
                    with Image.open(file_path) as img:
                        # 获取图片尺寸
                        width, height = img.size
                        # 如果图片像素宽度小于 min_width 或高度小于 min_height，则添加该文件到待删除列表中
                        if width < min_width or height < min_height:
                            to_delete.append(file_path)
                            print(f"Added {file_path} to delete list")
                except Exception as e:
                    print(f"Failed to process file {file_path}: {e}")

    # 删除待删除的文件
    for file_path in to_delete:
        try:
            os.remove(file_path)
            print(f"Deleted {file_path}")
        except Exception as e:
            print(f"Failed to delete file {file_path}: {e}")


# 
def filter_sub_process(src_folder, dst_folder, files, similar, method, size=[500, 500]):
    # size是[宽, 高]
    # ymin, ymax, xmin, xmax = [350, 1080, 600, 1700]
    print('%s\t==>\t%s' % (src_folder, dst_folder))
    os.mkdir(dst_folder)
    files.sort()
    if len(files):
        while True:
            i = 0
            img1 = cv2.imread(os.path.join(src_folder, files[i]))
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            # --------- 比较图片中的一部分 ----------
            # img1 = img1[ymin:ymax, xmin:xmax]
            # cv2.imshow("image", img1)  # 显示图片，后面会讲解
            # cv2.waitKey(0)  # 等待按键
            if img1 is not None:
                src_file = os.path.join(src_folder, files[i])
                dst_file = os.path.join(dst_folder, files[i])
                shutil.copy(src_file, dst_file)
                img1 = cv2.resize(img1, size)
                break
            i += 1
        img1 = cv2.GaussianBlur(img1, (5, 5), 0)
    for file in files:
        src_file = os.path.join(src_folder, file)
        img2 = cv2.imread(src_file)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        # -------- 比较图片中的一部分 -----------
        # img2 = img2[ymin:ymax, xmin:xmax]
        if img2 is not None:
            img2 = cv2.resize(img2, size)
            img2 = cv2.GaussianBlur(img2, (5, 5), 0)
            # print(img2.shape)
            (ssim, diff) = structural_similarity(img1, img2, full=True)
            if ssim < similar:
                dst_file = os.path.join(dst_folder, file)
                if method == 'copy':
                    shutil.copy(src_file, dst_file)
                elif method == 'move':
                    shutil.move(src_file, dst_file)
                img1 = img2
    print('\t==>\t%s\t---------- over' % dst_folder)

# 筛选图片
def filter_mp(src_top_folder, similar, method='copy', size=[500, 500], multi_process=1, dst_top_folder=None):
    src_top_folder = os.path.normpath(src_top_folder)
    if dst_top_folder is None:
        dst_top_folder = src_top_folder + '_filtered_' + str(similar)
    f_len = len(src_top_folder)
    if os.path.exists(dst_top_folder):
        shutil.rmtree(dst_top_folder)

    pool = mp.Pool(multi_process)

    for src_folder, dirs, files in os.walk(src_top_folder):
        dst_folder = os.path.join(dst_top_folder, src_folder[f_len + 1:])
        pool.apply_async(filter_sub_process, args=(src_folder, dst_folder, files, similar, method, size))

    pool.close()
    pool.join()

# 根据xml裁剪图片
def crop_from_xml(xml_file, img_folder, output_folder, object_name):
    if xml_file.endswith('.xml'):
        # 解析xml文件
        tree = ET.parse(xml_file)
        root = tree.getroot()

        # 查找所有name为'vest'的object标签
        for obj in root.findall('object'):
            name = obj.find('name').text
            if name != object_name:
                continue

            # 获取图片路径和坐标信息
            fname_temp = os.path.basename(xml_file).split('.')[0]
            img_path = os.path.join(img_folder, fname_temp + '.jpg')
            xmin = int(obj.find('bndbox/xmin').text)
            ymin = int(obj.find('bndbox/ymin').text)
            xmax = int(obj.find('bndbox/xmax').text)
            ymax = int(obj.find('bndbox/ymax').text)
            
            crop_from_position(img_path, [ymin, ymax, xmin, xmax], output_folder, f'{fname_temp}_{xmin}_{ymin}.jpg')

# 根据位置裁剪图片
def crop_from_position(img_path, position, output_folder, crop_name):
    if img_path.endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
        img = cv2.imread(img_path)
        ymin, ymax, xmin, xmax = position
        cropped_img = img[ymin:ymax, xmin:xmax]
        # 保存裁剪后的图片
        output_filename = os.path.join(output_folder, crop_name)
        cv2.imwrite(output_filename, cropped_img)


# 一个json文件
def get_apron_poly_from_json(json_file, name=None):
    with open(json_file, 'r') as f:
        json_data = json.load(f)
        shapes = json_data['shapes']
        if name is None:
            name = os.path.basename(json_file).split('.')[0]
        config_poly_map = {name: [(int(point[0]), int(point[1])) for point in item['points']] for item in shapes}
        print(config_poly_map)
        return config_poly_map


# folder下有多个文件夹，每个文件夹下有多个json文件
def get_apron_polys_from_folder_foler(folder):
    config_poly_map_merged = {}
    for subfolder in os.listdir(folder):
        allfolder = os.path.join(folder, subfolder)
        for file in os.listdir(allfolder):
            if file.lower().endswith('.json'):
                config_poly_map_i = get_apron_poly_from_json(os.path.join(allfolder, file),
                                                             name=subfolder + '/' + os.path.basename(file).split('.')[0])
                config_poly_map_merged.update(config_poly_map_i)
    print('-' * 100)
    print('merged config_poly_map: ')
    print(config_poly_map_merged)


# folder下有多个json文件
def get_apron_polys_from_folder(folder):
    config_poly_map_merged = {}
    for file in os.listdir(folder):
        if file.lower().endswith('.json'):
            config_poly_map_i = get_apron_poly_from_json(os.path.join(folder, file))
            config_poly_map_merged.update(config_poly_map_i)
    print('-' * 100)
    print(config_poly_map_merged)
    return config_poly_map_merged


def read_and_merge_excel(folder_path, output_sheet, sheet_name='整体覆盖率'):
    for i, file in enumerate(os.listdir(folder_path)):
        file_path = os.path.join(folder_path, file)
        print(i, file)
        if i == 0:
            sheet_all = pd.read_excel(file_path, sheet_name=sheet_name)
            sheet_all['文件名'] = os.path.splitext(file)[0]
        if i > 0:
            sheet_i = pd.read_excel(file_path, sheet_name=sheet_name)
            sheet_i['文件名'] = os.path.splitext(file)[0]
            sheet_all = pd.concat([sheet_all, sheet_i], ignore_index=True)
    cols = ['文件名'] + [col for col in sheet_all.columns if col != '文件名']
    sheet_all = sheet_all[cols]
    sheet_all.to_excel(output_sheet, index=False)

def convert_images_to_jpg(folder_path):
    # 支持的图片格式
    supported_formats = ('.webp', '.png', '.bmp', '.gif', '.jpeg')
    # 遍历指定文件夹
    for filename in os.listdir(folder_path):
        # 检查文件格式
        if filename.lower().endswith(supported_formats):
            try:
                # 文件路径
                file_path = os.path.join(folder_path, filename)
                # 打开图片
                with Image.open(file_path) as img:
                    # 转换图片到JPG
                    # 忽略透明通道，因为JPG不支持透明度
                    if img.mode in ('RGBA', 'LA', 'P'):
                        img = img.convert("RGB")
                    # 新的文件名和路径
                    new_filename = os.path.splitext(filename)[0] + '.jpg'
                    new_file_path = os.path.join(folder_path, new_filename)
                    # 保存转换后的图片
                    img.save(new_file_path, 'JPEG')
                    print(f"Converted {filename} to {new_filename}")
            except Exception as e:
                print(f"Failed to convert {filename}: {e}")

def save_video_interval_frames(video_path, save_dir, interval=0.2):
    # 从视频路径中提取视频名称（不包括扩展名）
    video_name = os.path.basename(video_path)
    video_name = os.path.splitext(video_name)[0]

    # 打开视频文件
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Error: Could not open video.")
            return

        # 获取视频的FPS（每秒帧数）
        fps = cap.get(cv2.CAP_PROP_FPS)
        # 计算间隔帧数
        frame_interval = int(fps * interval)

        frame_count = 0
        saved_frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 每隔frame_interval帧保存一帧
            if frame_count % frame_interval == 0:
                save_path = os.path.join(save_dir, f"{video_name}_frame_{saved_frame_count}.jpg")
                # cv2.imwrite(save_path, frame)
                cv2.imencode('.jpg', frame)[1].tofile(save_path)
                print(f"Saved {save_path}")
                saved_frame_count += 1

            frame_count += 1

        cap.release()
    except Exception as e:
        print('Error!!!')
    print("Done.")

def save_videos_interval_frames(video_folder, save_dir, interval=0.2):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    for file in os.listdir(video_folder):
        if file.endswith('.mp4'):
            save_video_interval_frames(os.path.join(video_folder, file), save_dir, interval=interval)
