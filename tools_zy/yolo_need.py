# coding:utf-8

import os
import random
from xml.dom.minidom import Document
import cv2
import fileinput
import glob
import shutil
import xml.etree.ElementTree as ET
import onnxruntime as ort
import numpy as np
from PIL import Image
from torchvision import transforms


# xml_to_yolo把文件夹下所有xml转换为yolo格式的label，
# split_train_val，获取所有xml文件路径，然后将其按照一定比例划分为train, val, test

def convert(size, box):
    dw = 1. / (size[0])
    dh = 1. / (size[1])
    x = (box[0] + box[1]) / 2.0 - 1
    y = (box[2] + box[3]) / 2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return x, y, w, h


def xml_to_yolo(xml_folder, xml_name, labels_folder, class_list=None):
    if class_list is None:
        class_list = ["person"]
    in_file = open(xml_folder + '/' + xml_name, encoding='UTF-8')
    out_file = open(labels_folder + '/' + xml_name[:-3] + 'txt', 'w')  # 这边写txt是要写到label这个文件夹中的。
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        # difficult = obj.find('Difficult').text
        cls = obj.find('name').text
        if cls not in class_list or int(difficult) == 1:
            continue
        cls_id = class_list.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
             float(xmlbox.find('ymax').text))
        b1, b2, b3, b4 = b
        # 标注越界修正
        if b2 > w:
            b2 = w
        if b4 > h:
            b4 = h
        b = (b1, b2, b3, b4)
        bb = convert((w, h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')


def xmls_to_yolo(xmls_folder, class_list=None):
    if class_list is None:
        class_list = ['xxx_class']
    # 创建label文件夹
    up_folder = os.path.dirname(xmls_folder)
    labels_folder = up_folder + '/labels'
    os.makedirs(labels_folder, exist_ok=True)

    # 对xml_folder下所有文件进行转换
    xml_names = [f.name for f in os.scandir(xmls_folder) if f.is_file()]
    for xml_name in xml_names:
        xml_to_yolo(xmls_folder, xml_name, labels_folder, class_list)
    print("xml文件转换成功：" + labels_folder)
    return labels_folder


def split_train_val(xmls_folder, trainval_proportion=0.9, train_proportion=0.8):
    # trainval_proportion: 训练集和验证集占总体的比例，剩下的是测试集
    # train_proportion: 训练集占train_val比例，可自己进行调整

    # 创建label文件夹
    up_folder = os.path.dirname(xmls_folder)
    split_folder = up_folder + '/dataSet_path'
    images_folder = up_folder + '/images'
    os.makedirs(split_folder, exist_ok=True)

    # 读取所有xml文件名，拼接到
    xmls_names = [os.path.join(images_folder, f.name[:-4] + '.jpg') for f in os.scandir(xmls_folder) if f.is_file()]
    # 随机划分
    num = len(xmls_names)
    list_index = range(num)
    tv = int(num * trainval_proportion)
    tr = int(tv * train_proportion)
    trainval = random.sample(list_index, tv)
    train = random.sample(trainval, tr)

    # 写入四个文件
    file_trainval = open(split_folder + '/trainval.txt', 'w')
    file_test = open(split_folder + '/test.txt', 'w')
    file_train = open(split_folder + '/train.txt', 'w')
    file_val = open(split_folder + '/val.txt', 'w')

    for i in list_index:
        name = xmls_names[i][:-4] + '.jpg\n'
        if i in trainval:
            file_trainval.write(name)
            if i in train:
                file_train.write(name)
            else:
                file_val.write(name)
        else:
            file_test.write(name)

    file_trainval.close()
    file_train.close()
    file_val.close()
    file_test.close()
    print("数据集划分成功：" + split_folder)
    return split_folder


def gen_yaml(dataSet_folder, class_list, yaml_name='dataset.yaml'):
    import yaml

    # 数据集配置参数
    params = {
        'path': dataSet_folder,
        'train': dataSet_folder + '/train.txt',
        'val': dataSet_folder + '/val.txt',
        'test': dataSet_folder + '/test.txt',
        'nc': len(class_list),
        'names': {i: category for i, category in enumerate(class_list)}
    }

    # 将数据集配置参数写入 YAML 文件
    yaml_path = os.path.dirname(dataSet_folder) + '/' + yaml_name
    yaml.Dumper.ignore_aliases = lambda *args: True  # 设置默认流参数以保留键值对的顺序
    with open(yaml_path, 'w') as f:
        yaml.dump(params, f, sort_keys=False)
    print("data yaml文件生成成功：" + yaml_path)
    return yaml_path


def file_to_wsl(yaml_path):
    # 拆分文件路径
    folder_path, yaml_name_with_extension = os.path.split(yaml_path)
    # 复制出ydataSet_path并修改
    dataSet_folder = folder_path + '\\dataSet_path'
    dataSet_folder_wsl = folder_path + '\\dataSet_path_wsl'
    os.makedirs(dataSet_folder_wsl, exist_ok=True)
    shutil.copy(dataSet_folder + '\\test.txt', dataSet_folder_wsl + '\\test.txt')
    shutil.copy(dataSet_folder + '\\train.txt', dataSet_folder_wsl + '\\train.txt')
    shutil.copy(dataSet_folder + '\\trainval.txt', dataSet_folder_wsl + '\\trainval.txt')
    shutil.copy(dataSet_folder + '\\val.txt', dataSet_folder_wsl + '\\val.txt')
    replace_for_wsl(dataSet_folder_wsl + '\\test.txt')
    replace_for_wsl(dataSet_folder_wsl + '\\train.txt')
    replace_for_wsl(dataSet_folder_wsl + '\\trainval.txt')
    replace_for_wsl(dataSet_folder_wsl + '\\val.txt')
    # 写yaml_wsl
    yaml_name, yaml_extension = os.path.splitext(yaml_name_with_extension)
    with open(yaml_path, 'r') as file:
        content = file.read()
    content = content.replace('\\', '/')
    content = content.replace('D:', '/mnt/d')
    content = content.replace('dataSet_path', 'dataSet_path_wsl')
    # 将替换后的内容写入原始文件
    with open(os.path.join(folder_path, yaml_name + '_wsl' + yaml_extension), 'w') as file:
        file.write(content)


def replace_for_wsl(file_path):
    with open(file_path, 'r') as file:
        content = file.read()

    # 将 \ 替换为 /
    content = content.replace('\\', '/')
    # 将 D: 替换为 /mnt/d
    content = content.replace('D:', '/mnt/d')

    # 将替换后的内容写入原始文件
    with open(file_path, 'w') as file:
        file.write(content)


def get_single_class_label_txt(input_folder, output_folder, find_number, replace_number=0):
    # 创建输出文件夹（如果不存在）
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历所有的 .txt 文件
    for filename in os.listdir(input_folder):
        if filename.endswith('.txt'):
            # 构建输入和输出文件名
            input_file = os.path.join(input_folder, filename)
            output_file = os.path.join(output_folder, filename)

            # 替换开头为 find_number 的行中的第一个数字，改为replace_number，并保存到输出文件中
            with open(input_file, 'r') as f1, open(output_file, 'w') as f2:
                for line in f1.readlines():
                    fields = line.strip().split(' ')
                    if fields[0] == str(find_number):
                        new_line = str(replace_number) + ' ' + ' '.join(fields[1:])
                        f2.write(new_line + '\n')


def get_single_class_label_xml(input_folder, output_folder, find_object, rename=None):
    # 遍历输入文件夹中的所有xml文件

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    xml_filenames = [filename for filename in os.listdir(input_folder) if filename.endswith('.xml')]

    for filename in xml_filenames:
        # 解析xml文件
        xml_file = os.path.join(input_folder, filename)
        print(xml_file)
        tree = ET.parse(xml_file)
        root = tree.getroot()

        # 删除object标签中内容不为vest的标签
        for obj in root.findall('object'):
            name = obj.find('name').text
            if name != find_object:
                root.remove(obj)
            elif rename is not None:
                obj.find('name').text = rename

        if len(root.findall('object')) == 0:
            continue
        # 将修改后的xml文件保存到输出文件夹中
        output_file = os.path.join(output_folder, filename)
        tree.write(output_file)


def merge_multi_classes_label_xml(image_path, xmls_folder_list, output_folder, single_name=None, element_folder='xxx'):
    # 遍历images_folder文件夹中的所有图片文件名
    # xml_filenames = [filename for filename in os.listdir(images_folder) if filename.endswith('.jpg')]
    # print(xml_filenames)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    name = os.path.basename(image_path)
    picfolder = os.path.dirname(image_path)
    # 读取图片和txt列表
    img = cv2.imread(os.path.join(picfolder, name[0:-4] + ".jpg"))
    # print(os.path.join(picfolder, name[0:-4] + ".jpg"))
    Pheight, Pwidth, Pdepth = img.shape
    xmlBuilder = Document()
    # 创建annotation标签
    annotation = xmlBuilder.createElement("annotation")
    xmlBuilder.appendChild(annotation)
    # folder标签
    folder = xmlBuilder.createElement("folder")
    folderContent = xmlBuilder.createTextNode(element_folder)
    folder.appendChild(folderContent)
    annotation.appendChild(folder)
    # filename标签
    filename = xmlBuilder.createElement("filename")
    filenameContent = xmlBuilder.createTextNode(name[0:-4] + ".jpg")
    filename.appendChild(filenameContent)
    annotation.appendChild(filename)
    # size标签
    size = xmlBuilder.createElement("size")
    # size子标签width
    width = xmlBuilder.createElement("width")
    widthContent = xmlBuilder.createTextNode(str(Pwidth))
    width.appendChild(widthContent)
    size.appendChild(width)
    # size子标签height
    height = xmlBuilder.createElement("height")
    heightContent = xmlBuilder.createTextNode(str(Pheight))
    height.appendChild(heightContent)
    size.appendChild(height)
    # size子标签depth
    depth = xmlBuilder.createElement("depth")
    depthContent = xmlBuilder.createTextNode(str(Pdepth))
    depth.appendChild(depthContent)
    size.appendChild(depth)
    annotation.appendChild(size)
    for xmls_folder in xmls_folder_list:
        xml_file = os.path.join(xmls_folder, name[0:-4] + ".xml")
        if not os.path.exists(xml_file):
            continue
        with open(xml_file, 'r') as f:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            for object_elem in root.iter('object'):
                new_object_elem = xmlBuilder.createElement("object")
                # 复制子元素到新的<object>元素中
                for child_elem in object_elem:
                    if child_elem.tag == 'bndbox':
                        new_bndbox_elem = xmlBuilder.createElement(child_elem.tag)
                        # 复制<bndbox>的子元素和数值
                        for bndbox_child_elem in child_elem:
                            new_bndbox_child_elem = xmlBuilder.createElement(bndbox_child_elem.tag)
                            new_bndbox_child_elem.appendChild(xmlBuilder.createTextNode(bndbox_child_elem.text))
                            new_bndbox_elem.appendChild(new_bndbox_child_elem)
                        new_object_elem.appendChild(new_bndbox_elem)
                    else:
                        if child_elem.tag == 'name' and isinstance(single_name, str):
                            new_child_elem = xmlBuilder.createElement(child_elem.tag)
                            new_child_elem.appendChild(xmlBuilder.createTextNode(single_name))
                            new_object_elem.appendChild(new_child_elem)
                        else:
                            new_child_elem = xmlBuilder.createElement(child_elem.tag)
                            new_child_elem.appendChild(xmlBuilder.createTextNode(child_elem.text))
                            new_object_elem.appendChild(new_child_elem)
                annotation.appendChild(new_object_elem)

    f = open(os.path.join(output_folder, name[0:-4] + ".xml"), 'w')
    print(os.path.join(output_folder, name[0:-4] + ".xml"))
    xmlBuilder.writexml(f, indent='\t', newl='\n', addindent='\t', encoding='utf-8')
    f.close()


def find_imgs_with_xml(xmls_folder, xml_path, out_folder, image_ext='jpg', method='copy'):
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    fname = os.path.basename(xml_path)
    xml_path = os.path.join(xmls_folder, fname[0:-3] + image_ext)
    xml_to_path = os.path.join(out_folder, fname[0:-3] + image_ext)
    if method == 'copy':
        shutil.copy(xml_path, xml_to_path)
    elif method == 'move':
        shutil.move(xml_path, xml_to_path)

def find_xmls_with_img(image_folder, imgs_path, out_folder, xml_ext='xml', method='copy'):
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    fname = os.path.basename(imgs_path)
    # print(fname[0:-3])
    img_path = os.path.join(image_folder, fname[0:-3] + xml_ext)
    img_to_path = os.path.join(out_folder, fname[0:-3] + xml_ext)
    if method == 'copy':
        shutil.copy(img_path, img_to_path)
    elif method == 'move':
        shutil.move(img_path, img_to_path)

def txt2xml(txtpath, picfolder, dict, xmlfolder, element_folder="xxx_folder"):  # 读取txt路径，xml保存路径，数据集图片所在路径
    if not os.path.exists(xmlfolder):
        os.makedirs(xmlfolder)
    name = os.path.basename(txtpath)
    # 读取图片和txt列表
    img = cv2.imread(os.path.join(picfolder, name[0:-4] + ".jpg"))
    # print(os.path.join(picfolder, name[0:-4] + ".jpg"))
    Pheight, Pwidth, Pdepth = img.shape

    with open(txtpath, 'r') as f:
        txtList = f.readlines()
        xmlBuilder = Document()
        # 创建annotation标签
        annotation = xmlBuilder.createElement("annotation")
        xmlBuilder.appendChild(annotation)
        # folder标签
        folder = xmlBuilder.createElement("folder")
        folderContent = xmlBuilder.createTextNode(element_folder)
        folder.appendChild(folderContent)
        annotation.appendChild(folder)
        # filename标签
        filename = xmlBuilder.createElement("filename")
        filenameContent = xmlBuilder.createTextNode(name[0:-4] + ".jpg")
        filename.appendChild(filenameContent)
        annotation.appendChild(filename)
        # size标签
        size = xmlBuilder.createElement("size")
        # size子标签width
        width = xmlBuilder.createElement("width")
        widthContent = xmlBuilder.createTextNode(str(Pwidth))
        width.appendChild(widthContent)
        size.appendChild(width)
        # size子标签height
        height = xmlBuilder.createElement("height")
        heightContent = xmlBuilder.createTextNode(str(Pheight))
        height.appendChild(heightContent)
        size.appendChild(height)
        # size子标签depth
        depth = xmlBuilder.createElement("depth")
        depthContent = xmlBuilder.createTextNode(str(Pdepth))
        depth.appendChild(depthContent)
        size.appendChild(depth)
        annotation.appendChild(size)

        for i in txtList:
            oneline = i.strip().split(" ")

            object = xmlBuilder.createElement("object")
            picname = xmlBuilder.createElement("name")
            nameContent = xmlBuilder.createTextNode(dict[int(oneline[0])])
            picname.appendChild(nameContent)
            object.appendChild(picname)
            pose = xmlBuilder.createElement("pose")
            poseContent = xmlBuilder.createTextNode("Unspecified")
            pose.appendChild(poseContent)
            object.appendChild(pose)
            truncated = xmlBuilder.createElement("truncated")
            truncatedContent = xmlBuilder.createTextNode("0")
            truncated.appendChild(truncatedContent)
            object.appendChild(truncated)
            difficult = xmlBuilder.createElement("difficult")
            difficultContent = xmlBuilder.createTextNode("0")
            difficult.appendChild(difficultContent)
            object.appendChild(difficult)
            bndbox = xmlBuilder.createElement("bndbox")
            xmin = xmlBuilder.createElement("xmin")
            mathData = int(((float(oneline[1])) * Pwidth + 1) - (float(oneline[3])) * 0.5 * Pwidth)
            xminContent = xmlBuilder.createTextNode(str(mathData))
            xmin.appendChild(xminContent)
            bndbox.appendChild(xmin)
            ymin = xmlBuilder.createElement("ymin")
            mathData = int(((float(oneline[2])) * Pheight + 1) - (float(oneline[4])) * 0.5 * Pheight)
            yminContent = xmlBuilder.createTextNode(str(mathData))
            ymin.appendChild(yminContent)
            bndbox.appendChild(ymin)
            xmax = xmlBuilder.createElement("xmax")
            mathData = int(((float(oneline[1])) * Pwidth + 1) + (float(oneline[3])) * 0.5 * Pwidth)
            xmaxContent = xmlBuilder.createTextNode(str(mathData))
            xmax.appendChild(xmaxContent)
            bndbox.appendChild(xmax)
            ymax = xmlBuilder.createElement("ymax")
            mathData = int(((float(oneline[2])) * Pheight + 1) + (float(oneline[4])) * 0.5 * Pheight)
            ymaxContent = xmlBuilder.createTextNode(str(mathData))
            ymax.appendChild(ymaxContent)
            bndbox.appendChild(ymax)
            object.appendChild(bndbox)

            annotation.appendChild(object)

        f = open(os.path.join(xmlfolder, name[0:-4] + ".xml"), 'w')
        print(os.path.join(xmlfolder, name[0:-4] + ".xml"))
        xmlBuilder.writexml(f, indent='\t', newl='\n', addindent='\t', encoding='utf-8')
        f.close()


# def replace_labelname(xml_folder, old_name, new_name):
#     files = glob.glob(os.path.join(xml_folder, "*.xml"))
#     for file in files:
#         print(file)
#         with fileinput.FileInput(file, inplace=True) as f:
#             for line in f:
#                 print(line.replace(old_name, new_name), end='')

# def replace_labelname(xml_folder, old_name, new_name, output_folder=None):
#     # 遍历xml_folder下所有xml文件
#     for file_name in os.listdir(xml_folder):
#         if file_name.endswith('.xml'):
#             xml_file = os.path.join(xml_folder, file_name)
#             tree = ET.parse(xml_file)
#             root = tree.getroot()
#             for obj in root.findall('object'):
#                 name = obj.find('name').text
#                 if name == old_name:
#                     obj.find('name').text = new_name
#                     print(name + '->' + obj.find('name').text)
#             if output_folder is None:
#                 tree.write(xml_file, xml_declaration=True, encoding='utf-8')
#             else:
#                 if not os.path.exists(output_folder):
#                     os.makedirs(output_folder)
#                 tree.write(os.path.join(output_folder, file_name), xml_declaration=True, encoding='utf-8')


def replace_labelname(xml_folder, old_names, new_names, output_folder=None):
    if len(old_names) != len(new_names):
        print("name mismatch")
        return
    old_new = {}
    for old_name, new_name in zip(old_names, new_names):
        old_new[old_name] = new_name
    # 遍历xml_folder下所有xml文件
    for file_name in os.listdir(xml_folder):
        if file_name.endswith('.xml'):
            xml_file = os.path.join(xml_folder, file_name)
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


def merge_xml(from_folder, to_folder):
    for file_name in os.listdir(from_folder):
        from_path = os.path.join(from_folder, file_name)
        to_path = os.path.join(to_folder, file_name)
        if os.path.exists(to_path):
            from_tree = ET.parse(from_path)
            from_root = from_tree.getroot()
            objects = from_root.findall("object")

            to_tree = ET.parse(to_path)
            annotation = to_tree.getroot()  # annotation就是根目录，

            for obj in objects:
                annotation.append(obj)
            to_tree.write(to_path, encoding="utf-8", xml_declaration=True)
            os.remove(from_path)
        else:
            shutil.move(from_path, to_path)

def change_xml_labels_using_cls_onnx(xml_file, img_folder, object_name, onnx_path, classes, crop_output_folder, noNM=False):
    image_size = 224  # 根据您的模型需要设置
    if noNM:
        test_transforms = transforms.Compose([
            transforms.Resize(size=(image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0, 0, 0], [0.229, 0.224, 0.225])
        ])
    else:
        test_transforms = transforms.Compose([
            transforms.Resize(size=(image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    sess = ort.InferenceSession(onnx_path)
    if not os.path.exists(crop_output_folder):
        os.makedirs(crop_output_folder)
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
            position = [ymin, ymax, xmin, xmax]
            crop_name = f'{fname_temp}_{xmin}_{ymin}.jpg'
            img = cv2.imread(img_path)
            cropped_img = img[ymin:ymax, xmin:xmax]
            # 保存裁剪后的图片
            crop_path = os.path.join(crop_output_folder, crop_name)
            cv2.imwrite(crop_path, cropped_img)
            # 加载图像并应用预处理
            img = Image.open(crop_path).convert("RGB")
            img_transformed = test_transforms(img)
            img_transformed = np.expand_dims(img_transformed.numpy(), 0)  # 添加批次维度

            # 进行推理
            input_name = sess.get_inputs()[0].name
            output = sess.run(None, {input_name: img_transformed})

            print(f"output: {output}")
            # 输出是softmax概率，获取最可能的类别
            predicted_class = np.argmax(output[0])
            print(f"class: {predicted_class}")

            # 将xml中的类别name修改为预测的类别，然后保存
            obj.find('name').text = str(classes[predicted_class])
        tree.write(xml_file)
def change_xmls_labels_using_cls_onnx(xmls_folder, img_folder, object_name, onnx_path, classes, crop_output_folder, noNM=False):
    for filename in os.listdir(xmls_folder):
        xml_file = os.path.join(xmls_folder, filename)
        change_xml_labels_using_cls_onnx(xml_file, img_folder, object_name, onnx_path, classes, crop_output_folder, noNM=noNM)

if __name__ == '__main__':
    # xmls_folder = r'F:\Zhiyuan\pic_annotations_data\Annotations'
    # class_list = ["person"]
    # labels_folder = xmls_to_yolo(xmls_folder, class_list)
    # dataSet_folder = split_train_val(xmls_folder, 0.5, 0.5)
    # yaml_path = gen_yaml(dataSet_folder, class_list, 'test.yaml')

    txtfolder = r"D:\Zhiyuan\pic_data\pytest\detect_result2\labels"
    picfolder = r"D:\Zhiyuan\pic_data\pytest\detect_result2"
    dict = {'0': "vest"}
    xmlfolder = r"D:\Zhiyuan\pic_data\pytest\detect_result2\xml"
