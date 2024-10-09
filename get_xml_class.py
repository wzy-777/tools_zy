import os
import xml.etree.ElementTree as ET

img_formats = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo',]
    
def get_xml_folder_category(voc_root):
    category_set = set()
    for xml_file in os.listdir(voc_root):
        if not xml_file.endswith('.xml'):
            continue
        xml_path = os.path.join(voc_root, xml_file)
        tree = ET.parse(xml_path)  
        root = tree.getroot()  
        for obj in root.iter('object'):  
            category_name = obj.find('name').text
            category_set.add(category_name)
            if category_name == 'zhesheqi':
                print(xml_path)
    return category_set

def get_xml_folder_folder_category(folder_folder):
    category = set()
    for folder in os.listdir(folder_folder):
        xml_folder = os.path.join(folder_folder, folder)
        category_i = get_xml_folder_category(xml_folder)
        category = category.union(category_i)
    return category

def get_xml_folder_folder_folder_category(folder_folder_folder):
    category = set()
    for folder_folder in os.listdir(folder_folder_folder):
        xml_folder_folder = os.path.join(folder_folder_folder, folder_folder)
        for folder in os.listdir(xml_folder_folder):
            xml_folder = os.path.join(xml_folder_folder, folder)
            category_i = get_xml_folder_category(xml_folder)
            category = category.union(category_i)
    return category

if __name__ == '__main__':
    # folder_folder = r"D:\Zhiyuan\图片采集\data\外包采集图像\20240910_审核完成\winter_left"
    # category = get_xml_folder_folder_category(folder_folder)
    # print(folder_folder, '\n', len(category), '\n', category)
    
    folder_folder_folder = r"D:\Zhiyuan\图片采集\data\外包采集图像\20240913_审核完成"
    category = get_xml_folder_folder_folder_category(folder_folder_folder)
    print(folder_folder_folder, '\n', len(category), '\n', category)
    
    # folder = r"D:\Zhiyuan\图片采集\data\外包采集图像\20240910_审核完成\winter_left"
    # category = get_xml_folder_category(folder)
    # print(folder, '\n', len(category), '\n', category)