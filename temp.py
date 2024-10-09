import os

# pic_folder = r'D:\Zhiyuan\图片采集\2024-08-16'
# pics = os.listdir(pic_folder)
# for pic in pics:
#     # 重命名
#     if pic.endswith('.bmp'):
#         if len(pic.split('_')[0]) == 17:
#             new_pic = '_'.join(pic.split('_')[1:])
#             os.rename(pic_folder + '/' + pic, pic_folder + '/' + new_pic)
# file_path = r"D:\Zhiyuan\work_tools_py2\2024-08-20-13-42-39-186-down-A--img-rgb.data"


# from PIL import Image
# folder = r"D:\Zhiyuan\图片采集\data\外包采集图像\余晨奥 身高180 体重60kg\image\cc"
# for datai in os.listdir(folder):
#     if not datai.endswith(".data"):
#         continue
#     # file_path = r"D:\Zhiyuan\图片采集\data\演示案例\2024-08-27-17-18-16-250-up-B--img-rgb.data"
#     file_path = folder + '/' + datai
#     width = 512
#     height = 1024
#     # 打开文件
#     with open(file_path, 'rb') as file:
#         # 读取文件内容
#         binary_data = file.read()
#     # 确保数据长度与预期的图像大小匹配
#     expected_data_length = width * height * 3  # 3个颜色通道，每个像素3个字节
#     if len(binary_data) != expected_data_length:
#         raise ValueError(f"Data length mismatch. Expected {expected_data_length} bytes, got {len(binary_data)} bytes.")
#     # 将读取的数据转换为图像
#     image = Image.frombytes('RGB', (width, height), binary_data)
#     # 保存图像
#     save_path = file_path.replace(".data", ".png")
#     image.save(save_path)
#     # image.show()


from tools_zy import file_processing
import os

if __name__ == '__main__':
    
    def ff_(folder_folder):
        for folder in os.listdir(folder_folder):
            imgs_folder = os.path.join(folder_folder, folder)
            xml_out_folder = imgs_folder
            file_processing.find_xmls_with_imgs(imgs_folder, xml_folder, 'xml', xml_out_folder, 'move')

    def fff_(folder_folder_folder):
        for folder_folder in os.listdir(folder_folder_folder):
            xml_folder_folder = os.path.join(folder_folder_folder, folder_folder)
            for folder in os.listdir(xml_folder_folder):
                xml_folder = os.path.join(xml_folder_folder, folder)
                # 函数
                file_processing.replace_labelname(xml_folder, old_new=old_new)
    
    old_new = {'碟形爆炸物': 'diexingbaozhawu', 
               '陶瓷刀': 'taocidao', 
               '小螺丝刀': 'xiaoluosidao',
               '粉末爆炸物': 'fenmobaozhawu', 
               '指甲刀': 'zhijiadao', 
               '扳手': 'banshou', 
               '矩形爆炸物': 'juxingbaozhawu', 
               '摄像头': 'shexiangtou', 
               '枪': 'qiang', 
               '烟盒': 'yanhe', 
               '录音笔': 'luyinbi', 
               '打火机': 'dahuoji', 
               '细绳': 'xisheng', 
               '子弹': 'zidan', 
               'fenmo': 'fenmobaozhawu', 
               '注射器': 'zhusheqi', 
               '塑料U盘': 'suliaoUpan', 
               '金属U盘': 'jinshuUpan', 
               '金属刀': 'jinshudao', 
               '麻绳': 'masheng', 
               '方形录音笔': 'fangxingluyinbi', 
               'juxing': 'juxingbaozhawu', 
               '纸币': 'zhibi', 
               '液体': 'yeti'}
    fff_(r"D:\Zhiyuan\图片采集\data\外包采集图像\20240820_标注完成")
    
    # 根据图片文件查找xml
    # xml_folder = r"D:\Zhiyuan\图片采集\data\外包采集图像\20240820_标注完成"
    # ff_(r"D:\Zhiyuan\图片采集\data\外包采集图像\20240820_标注完成\夏季\左侧")
