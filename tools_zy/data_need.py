import copy
import json
import os
import shutil
import random
import time


def rename_pics_with_txt(pics_folder: str, txt_path: str, end: str = '.jpg', factor: int = 1) -> None:
    """
    使用给定的文本文件中的名称，重命名指定文件夹中的图片文件。
    图片名称不同，new_name规则要修改
    参数:
    pics_folder (str): 包含图片文件的文件夹路径。
    txt_path (str): 包含新文件名的文本文件路径。
    end (str, optional): 新文件名的扩展名。默认为 '.jpg'。

    返回:
    None
    """
    pics_name = [pici for pici in os.listdir(pics_folder) if pici.endswith(end)]
    with open(txt_path, 'r', encoding='utf-8') as file:
        names_fix = file.read().split('\n')
    if len(names_fix) * factor == len(pics_name):
        for i in range(len(pics_name)):
            # 根据不同图片名称，修改这里的new_name规则
            # ['20240805152733382', 'SIS', '002', 'admin', 'A', 'down', 'MCITY-2024-08-05-15-27-39-205.bmp']
            tt, _, _, _, AB, _, _ = pics_name[i].split('_')
            new_name = tt + '_' + AB + '_' + names_fix[i // 2] + '.bmp'

            os.rename(os.path.join(pics_folder, pics_name[i]), os.path.join(pics_folder, new_name))

def json_one_2_more(pics_folder: str, jsonA_path: str, jsonB_path: str):
    """
    把一个labelme的标注json文件，复制到所有的图片中。
    """
    pathA = pics_folder + '/A'
    pathB = pics_folder + '/B'
    if not os.path.exists(pathA):
        os.makedirs(pathA)
    if not os.path.exists(pathB):
        os.makedirs(pathB)
    # 移动图片到A、B子文件夹。
    for ci in os.listdir(pics_folder):
        if not ci.endswith('.bmp'):
            continue
        if 'A' in ci:
            shutil.move('/'.join([pics_folder, ci]), pathA)
        elif 'B' in ci:
            shutil.move('/'.join([pics_folder, ci]), pathB)
    # 复制A、B json文件
    with open(jsonA_path, 'r', encoding='utf-8') as file:
        dictA = json.load(file)
    with open(jsonB_path, 'r', encoding='utf-8') as file:
        dictB = json.load(file)
    for ci in os.listdir(pathA):
        dictA['imagePath'] = ci
        with open('/'.join([pathA, ci[:-4] + '.json']), 'w', encoding='utf-8') as file:
            json.dump(dictA, file, ensure_ascii=False, indent=4)
    for ci in os.listdir(pathB):
        dictB['imagePath'] = ci
        with open('/'.join([pathB, ci[:-4] + '.json']), 'w', encoding='utf-8') as file:
            json.dump(dictB, file, ensure_ascii=False, indent=4)

def labelmes2coco(data_folder: str, rename=True, fixed_bbox=None) -> str:
    pic_folder = f'{data_folder}/images'
    new_pic_folder = f'{data_folder}/images_new'
    json_folder = f'{data_folder}/labelme_json'
    new_json_folder = f'{data_folder}/labelme_json_new'
    os.makedirs(new_pic_folder, exist_ok=True)
    os.makedirs(new_json_folder, exist_ok=True)

    # ========== categories ==========
    def get_categories(json_file: str = 'tools_wzy/categories_person.json') -> dict:
        with open(json_file, 'r', encoding='utf-8') as file:
            categories = json.load(file)
            return categories
    categorie_i = get_categories()
    categories = [categorie_i]
    keypoints = [0] * 3 * len(categories[0]['keypoints'])
    # ========== images and annotations ==========
    images = []
    annotations = []
    for json_file in os.listdir(json_folder):
        json_path = os.path.join(json_folder, json_file)
        pic_name = json_file.split('.')[0] + '.bmp'
        with open(json_path, 'r', encoding='utf-8') as file:
            dict_labelme = json.load(file)
            # ========== images ==========
            file_id = int(dict_labelme['imagePath'].split('_')[0][4:])
            if rename:
                pic_name_new = str(file_id) + '.jpg'
            else:
                pic_name_new = json_file.replace('.json', '.jpg')
            shutil.copy(f'{pic_folder}/{pic_name}', new_pic_folder + '/' + pic_name_new)
            images_i = {'height': dict_labelme['imageHeight'],
                        'width': dict_labelme['imageWidth'],
                        'file_name': pic_name_new,
                        'id': file_id}
            images.append(images_i)
            # ========== annotations ==========
            shapes = dict_labelme['shapes']
            num_keypoints = 0
            for shape_i in shapes:
                if shape_i['shape_type'] == 'rectangle':
                    if fixed_bbox is None:
                        (x1, y1), (x2, y2) = shape_i['points']
                        x1, x2 = (x1, x2) if x1 <= x2 else (x2, x1)
                        y1, y2 = (y1, y2) if y1 <= y2 else (y2, y1)
                        bbox = [x1, y1, x2 - x1, y2 - y1]
                    else:
                        bbox = fixed_bbox
                elif shape_i['shape_type'] == 'point':
                    idx = int(shape_i['label'])
                    keypoints[(idx - 1) * 3 + 0] = shape_i['points'][0][0]
                    keypoints[(idx - 1) * 3 + 1] = shape_i['points'][0][1]
                    keypoints[(idx - 1) * 3 + 2] = 2
                    if shape_i['group_id'] is not None:
                        keypoints[(idx - 1) * 3 + 2] = int(shape_i['group_id'])
                    num_keypoints = num_keypoints + 1
            shutil.copy(f'{json_folder}/{json_file}', new_json_folder + '/' + pic_name_new.replace('.jpg', '.json'))
            annotation_i = {'segmentation': [[]],
                            'num_keypoints': num_keypoints,
                            'area': int(bbox[2] * bbox[3]),
                            'iscrowd': 0,
                            'keypoints': copy.deepcopy(keypoints),
                            'image_id': file_id,
                            'bbox': bbox,
                            'category_id': 1,
                            'id': file_id
                            }
            annotations.append(annotation_i)
    dict_coco = {'images': images,
                 'annotations': annotations,
                 'categories': categories}
    # 生成json文件
    coco_json_file_path = f'{data_folder}/coco.json'
    print(f'coco_json_file_path: {coco_json_file_path}')
    if os.path.exists(coco_json_file_path):
        os.remove(coco_json_file_path)
    with open(coco_json_file_path, 'w', encoding='utf-8') as file:
        json.dump(dict_coco, file, ensure_ascii=False, indent=4)
    return data_folder


def split_labelme(labelme_folder: str, ratio: tuple = (0.85, 0.1, 0.05)):
    random.seed(time.time())

    # 获取源文件夹中的所有JPG文件
    pics = [f for f in os.listdir(labelme_folder) if f.split('.')[-1] in ['jpg', 'bmp']]
    # 随机打乱并划分JPG文件列表
    random.shuffle(pics)
    total_files = len(pics)
    train_ratio, val_ratio, test_ratio = ratio
    train_cnt, val_cnt = int(total_files * train_ratio), int(total_files * val_ratio)
    train_jpgs = pics[:train_cnt]
    val_jpgs = pics[train_cnt:train_cnt + val_cnt]
    test_jpgs = pics[train_cnt + val_cnt:]

    def move_files(pics, target_dir):
        os.makedirs(target_dir, exist_ok=True)
        img_folder = os.path.join(target_dir, 'images')
        os.makedirs(img_folder, exist_ok=True)
        json_folder = os.path.join(target_dir, 'labelme_json')
        os.makedirs(json_folder, exist_ok=True)

        for pic in pics:
            json_file = pic.replace(pic.split('.')[-1], 'json')
            shutil.copy(os.path.join(labelme_folder, pic), img_folder)
            if os.path.exists(os.path.join(labelme_folder, json_file)):
                shutil.copy(os.path.join(labelme_folder, json_file), json_folder)

    # 移动文件到相应的目标文件夹
    split_folder = labelme_folder + '_split'
    move_files(train_jpgs, split_folder + '/train')
    move_files(val_jpgs, split_folder + '/val')
    move_files(test_jpgs, split_folder + '/test')

    print(f"split pic and json to:\n"
          f"==>{split_folder}")
    return split_folder
