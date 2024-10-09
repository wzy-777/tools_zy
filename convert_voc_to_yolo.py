import json
import os
import PIL.Image as Image
from tqdm import tqdm
import xml.etree.ElementTree as ET
import argparse  

img_formats = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo',]

def parse_xml_file(xml_file, category_dict):
    tree = ET.parse(xml_file)  
    root = tree.getroot()  
    annotations = []
    for obj in root.iter('object'):  
        difficult = int(obj.find('difficult').text)  
        bndbox = obj.find('bndbox')  
        xmin = int(bndbox.find('xmin').text)  
        ymin = int(bndbox.find('ymin').text)  
        xmax = int(bndbox.find('xmax').text)  
        ymax = int(bndbox.find('ymax').text)  
        categeory_name = obj.find('name').text
        # if categeory_name not in category_dict.keys():
        #     continue
        categeory = category_dict[categeory_name]
        anno = {
            "x1":xmin,
            "y1":ymin,
            "x2":xmax,
            "y2":ymax,
            "category_name":categeory_name,
            "category":categeory,
            "difficult":difficult
        }
        annotations.append(anno)
    return annotations

"""
    将voc文件转换为yolo格式, 
"""
def make_yolo_from_voc(voc_root, save_root, img_root, category_dict):
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    # 解析voc的xml文件，转换为yolo格式标注
    for xml_file in tqdm(os.listdir(voc_root)):
        if not xml_file.endswith('.xml'):
            continue
        xml_path = os.path.join(voc_root, xml_file)
        annotations = parse_xml_file(xml_path, category_dict)
        if len(annotations) == 0:
            print(xml_path)
            continue
        img_name = xml_file.replace('.xml', '.bmp')
        img_path = os.path.join(img_root, img_name)
        img = Image.open(img_path)
        width, height = img.size
        # 将voc的标注转换为yolo的标注
        yolo_annotations = []
        for anno in annotations:
            x1, y1, x2, y2 = anno['x1'], anno['y1'], anno['x2'], anno['y2']
            x_center = (x1 + x2) / 2 / width
            y_center = (y1 + y2) / 2 / height
            w = (x2 - x1) / width
            h = (y2 - y1) / height
            yolo_anno = [anno['category'], x_center, y_center, w, h]
            yolo_annotations.append(yolo_anno)
        # 将yolo的标注保存到txt文件中
        txt_name = img_name.replace('.bmp', '.txt')
        txt_path = os.path.join(save_root, txt_name)
        with open(txt_path, 'w') as f:
            for yolo_anno in yolo_annotations:
                f.write(' '.join(map(str, yolo_anno)) + '\n')
        #print(f'{xml_path} -> {txt_path}')
    print('Done!')
 

if __name__ == '__main__':
    # category_dict = {
    #     "Metaknives": 0,
    #     "Gun": 1,
    #     "Lighter": 2,
    #     "Bullet": 3,
    #     "Liquid": 4
    # }
    # ch = ['手机', '碟形爆炸物', '粉末爆炸物', '矩形爆炸物', '陶瓷刀', '金属刀', '打火机', '液体', '子弹', '枪', '录音笔', '方形录音笔', '小螺丝刀', '指甲刀', '注射器', '麻绳', '扳手', '烟盒', '纸币', '金属U盘', '塑料U盘', '细绳', '摄像头']

    pinyin = ['tou', 'shouji', 'diexingbaozhawu', 'fenmobaozhawu', 'juxingbaozhawu', 'taocidao', 'jinshudao', 'dahuoji', 'yeti', 'zidan', 'qiang', 'luyinbi', 'fangxingluyinbi', 'xiaoluosidao', 'zhijiadao', 'zhusheqi', 'masheng', 'banshou', 'yanhe', 'zhibi', 'Upan', 'xisheng', 'shexiangtou']
    category_dict = {pinyin[i]: i for i in range(len(pinyin))}
    print(len(pinyin))
    voc_root = "/data/things/20240823/xml"
    img_root = "/data/things/20240823/bmp"
    save_root = "/home/wangzhiyuan/data/things/20240823/labels"
    make_yolo_from_voc(voc_root, save_root, img_root, category_dict)
    
    