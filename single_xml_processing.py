from tools_zy_bak import yolo_need


# 从txt中获取单一类，重命名类序号（默认修改为0）
input_folder = r"D:\Zhiyuan\pic_data_20231229\__detect\JJN\00\txt\labels"  # ?
output_folder = r"D:\Zhiyuan\pic_data_20231229\__detect\JJN\00\xmls"
# 提取出需要的行，然后重命名类序号，保存
find_number = 1
replace_number = 0
# yolo_need.get_single_class_label_txt(input_folder, output_folder, find_number, replace_number=0)


# 从xml中获取单一类到新的文件夹中，可同时重命名类名（默认或为None时不修改）
input_folder = r"D:\Zhiyuan\pic_data_20231229\__detect\HUZ\detect_HUZ_20240520_1_xml"
output_up_folder = r"D:\Zhiyuan\pic_data_20231229\__detect\HUZ\detect_HUZ_20240520_1"
cs = {0: 'FJ', 1: 'person', 2: 'cone', 3: 'KTC', 4: 'QYC', 5: 'CSC', 6: 'PTC', 7: 'JYC', 8: 'KCM', 9: 'HCM', 10: 'LQ', 11: 'PCC', 12: 'workladder', 13: 'XLC', 14: 'PBC', 15: 'YDC', 16: 'other'}
# for key in cs:
#     find_object = cs[key]
#     output_folder = output_up_folder + '\\' + find_object
#     yolo_need.get_single_class_label_xml(input_folder, output_folder, find_object, rename=None)


# 替换xml文件中的类名
# ch = ['金属刀']
# pinyin = ['jinshudao']
# ch = ['手机', '碟形爆炸物', '粉末爆炸物', '矩形爆炸物', '陶瓷刀', '金属刀', '打火机', '液体', '子弹', '枪', '录音笔', '方形录音笔',
#       '小螺丝刀', '指甲刀', '注射器', '麻绳', '扳手', '烟盒', '纸币', '金属U盘', '塑料U盘', '细绳', '摄像头', 'fenmo', 'juxing', 'diexing']
# pinyin = ['shouji', 'diexingbaozhawu', 'fenmobaozhawu', 'juxingbaozhawu', 'taocidao', 'jinshudao', 'dahuoji', 'yeti', 'zidan', 'qiang', 'luyinbi', 'fangxingluyinbi',
#           'xiaoluosidao', 'zhijiadao', 'zhusheqi', 'masheng', 'banshou', 'yanhe', 'zhibi', 'jinshuUpan', 'suliaoUpan', 'xisheng', 'shexiangtou', 'fenmobaozhawu', 'juxingbaozhawu', 'diexingbaozhawu']
ch = ['fenmo', 'juxing', 'diexing']
pinyin = ['fenmobaozhawu', 'juxingbaozhawu', 'diexingbaozhawu']
xml_folder = r"D:\Zhiyuan\图片采集\data\外包采集图像\赵恩博 身高175 体重83_ok - 副本"
output_folder = r"D:\Zhiyuan\图片采集\data\外包采集图像\赵恩博 身高175 体重83_ok - 副本"
yolo_need.replace_labelname(xml_folder, old_names=ch, new_names=pinyin, output_folder=output_folder)

# 替换xml文件中的类名
xml_folder = r"D:\Zhiyuan\图片采集\data\外包采集图像\zhaoenbo\Annotations"
old_name = '粉末爆炸物'
new_name = 'fenmobaozhawu'
output_folder = r"D:\Zhiyuan\图片采集\data\外包采集图像\zhaoenbo\Annotations"
# yolo_need.replace_labelname(xml_folder, old_name, new_name, output_folder)


# 根据分类模型的结果，修改xml文件中的类名
xmls_folder = r'C:\Users\feeyo\Desktop\test\Annotations'
img_folder = r'C:\Users\feeyo\Desktop\test\images'
crop_output_folder = r'C:\Users\feeyo\Desktop\test\crop'
object_name = 'vest'
onnx_path = r"D:\Zhiyuan\video\record_all_finall\classify\crop\model\v2\best_noNM_softmax.onnx"
classes = {0: 'yellow', 1: 'orange', 2: 'blue'}
# yolo_need.change_xmls_labels_using_cls_onnx(xmls_folder, img_folder, object_name, onnx_path, classes, crop_output_folder, noNM=True)



# 根据一个xml文件查找对应的图片，有pool版本！！！（注意这是一个！！）
image_folder = r'C:\Users\feeyo\Desktop\FlightData_20230213\images'
xml_folder = r'C:\Users\feeyo\Desktop\FlightData_20230213\Annotations'
out_folder = xml_folder
# yolo_need.find_imgs_with_xml(image_folder, xml_folder, out_folder, image_ext='jpg')


# 合并xml文件
from_folder = r'D:\Zhiyuan\pic_data_20231229\__detect\JJN\00\xmls'
to_folder = r'D:\Zhiyuan\pic_data_20231229\__detect\JJN\00\Annotations'
# yolo_need.merge_xml(from_folder, to_folder)
