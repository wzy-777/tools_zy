import os
import shutil
# import hashlib
# from PIL import Image
# from skimage.metrics import structural_similarity
# import cv2
# import numpy as np
# import copy
# import sys
# import multiprocessing as mp
import xml.etree.ElementTree as ET
# import json
# import pandas as pd


def move_images(source_folder, destination_folder, file_format=None):
    """
    将一个文件夹下的所有图片移动到新的文件夹（注意：两个输入不能有嵌套关系）,
    src_dir和dst_dir可以写一样的

    参数:
    source_folder (str): 源文件夹路径。
    destination_folder (str): 目标文件夹路径。
    file_format (str, optional): 需要移动的文件格式，file_format默认为('.png', '.jpg', '.jpeg', '.gif', '.bmp')

    返回:
    None
    """
    if file_format is None:
        file_format = ('.png', '.jpg', '.jpeg', '.gif', '.bmp')
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    # 遍历指定文件夹中的所有子文件夹
    for root, dirs, files in os.walk(source_folder):
        for file_name in files:
            # 获取文件的完整路径
            file_path = os.path.join(root, file_name)
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

def replace_labelname(xml_folder, old_names=[], new_names=[], output_folder=None, all_xml=True, old_new=None):
    """
    替换xml文件中的类名

    Args:
        xml_folder (str): xml文件所在文件夹，可以为多级的父目录
        old_names (list): 旧标签名
        new_names (list): 新标签名
        output_folder (str, optional): 修改标签后的xml放置的文件夹. Defaults to None.
        all_xml (bool, optional): 是否遍历包含子文件夹的所有xml文件，否则只遍历指定目录下的. Defaults to True.
    """
    def rename_and_write(xml_file):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for obj in root.findall('object'):
            name = obj.find('name').text
            if name in old_new:
                obj.find('name').text = old_new[name]
                print(name + '->' + old_new[name])
        if output_folder is None:
            tree.write(xml_file, xml_declaration=True, encoding='utf-8')
        else:
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            tree.write(os.path.join(output_folder, file_name), xml_declaration=True, encoding='utf-8') 
            
    if old_new is None:            
        if len(old_names) != len(new_names):
            print("name mismatch")
            return
        old_new = {}
        for old_name, new_name in zip(old_names, new_names):
            old_new[old_name] = new_name
    if all_xml:
        # 遍历xml_folder下，包含子文件夹下的xml文件
        for root, dirs, files in os.walk(xml_folder):
            for file_name in files:
                if file_name.endswith('.xml'):
                    # 获取文件的完整路径
                    xml_file = os.path.join(root, file_name)
                    rename_and_write(xml_file)
    else:
        # 
        for file_name in os.listdir(xml_folder):
            if file_name.endswith('.xml'): 
                xml_file = os.path.join(xml_folder, file_name)
                rename_and_write(xml_file)

def find_A_with_B(b_path, a_folder, a_ext, out_folder, method='copy'):
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    if not a_ext.startswith('.'):
        a_ext = '.' + a_ext
    b_base_name = os.path.basename(b_path)
    b_name_no_ext = os.path.splitext(b_base_name)[0]
    a_name = b_name_no_ext + a_ext
    a_path = os.path.join(a_folder, a_name)
    out_path = os.path.join(out_folder, os.path.basename(a_path))
    if not os.path.exists(a_path):
        return
    if method == 'copy':
        shutil.copy(a_path, out_path)
    elif method == 'move':
        shutil.move(a_path, out_path)

# 根据xml文件找图像
def find_img_with_xml(xml_path, img_folder, img_ext, out_folder, method='copy'):
    find_A_with_B(xml_path, img_folder, img_ext, out_folder, method)

# 根据图像找xml文件
def find_xml_with_img(img_path, xml_folder, xml_ext, out_folder, method='copy'):
    find_A_with_B(img_path, xml_folder, xml_ext, out_folder, method)

def find_imgs_with_xmls(xmls_folder, img_folder, img_ext, out_folder, method='copy'):
    for xml_name in os.listdir(xmls_folder):
        xml_path = os.path.join(xmls_folder, xml_name)
        find_A_with_B(xml_path, img_folder, img_ext, out_folder, method)
        
def find_xmls_with_imgs(imgs_folder, xml_folder, xml_ext, out_folder, method='copy'):
    for img_name in os.listdir(imgs_folder):
        img_path = os.path.join(imgs_folder, img_name)
        find_A_with_B(img_path, xml_folder, xml_ext, out_folder, method)

def find_As_with_Bs(B_folder, A_folder, A_ext, out_folder, method='copy'):
    for B_name in os.listdir(B_folder):
        B_path = os.path.join(A_folder, B_name)
        find_A_with_B(B_path, A_folder, A_ext, out_folder, method)