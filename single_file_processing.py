from tools_zy import file_processing
import os

if __name__ == '__main__':

    # 将一个文件夹下的所有图片移动到新的文件夹（注意：两个输入不能有嵌套关系）,src_dir和dst_dir可以写一样的
    src_folder = r"D:\Zhiyuan\图片采集\data\外包采集图像\20240913"
    dst_folder = r"D:\Zhiyuan\图片采集\data\外包采集图像\可用归档\20240913"
    file_processing.move_images(src_folder, dst_folder + '/xml', file_format='.xml')
    file_processing.move_images(src_folder, dst_folder + '/rgb_data', file_format='.data')
    file_processing.move_images(src_folder, dst_folder + '/bmp', file_format='.bmp')

    # 替换xml文件中的类名
    # ch = ['fenmo', 'juxing', 'diexing']
    # pinyin = ['fenmobaozhawu', 'juxingbaozhawu', 'diexingbaozhawu']
    old_new = {'jinshuUpan': 'Upan'}
    xml_folder = r"D:\Zhiyuan\图片采集\data\外包采集图像\20240819_标注完成\右边\01蝶形爆炸物"
    # file_processing.replace_labelname(xml_folder, old_new=old_new)
    
    
    # 根据xml文件查找图片
    xml_folder = r"D:\Zhiyuan\图片采集\data\外包采集图像\ok\temp\Annotations"
    img_folder = r"D:\Zhiyuan\图片采集\data\外包采集图像\ok\temp\images"
    img_out_folder = r"D:\Zhiyuan\图片采集\data\外包采集图像\ok\temp\image"  # 没有会创建新的
    # for xml_name in os.listdir(xml_folder):
    #     xml_path = os.path.join(xml_folder, xml_name)
    #     file_processing.find_img_with_xml(xml_path, img_folder, 'bmp', img_out_folder, 'copy')
    
    
    # 根据图片文件查找xml
    imgs_folder = r"D:\Zhiyuan\图片采集\data\外包采集图像\20240819_标注完成\左侧\22摄像头"
    xml_folder = r"D:\Zhiyuan\图片采集\data\外包采集图像\20240819_标注完成"
    xml_out_folder = imgs_folder
    # file_processing.find_xmls_with_imgs(imgs_folder, xml_folder, 'xml', xml_out_folder, 'move')
    
    B_folder = r"D:\Zhiyuan\图片采集\data\外包采集图像\可用归档\20240820\bmp"
    A_folder = r"D:\Zhiyuan\图片采集\data\外包采集图像\可用归档\20240820\rgb_data"
    A_out_folder = r"D:\Zhiyuan\图片采集\data\外包采集图像\可用归档\20240820\rgb"
    # file_processing.find_As_with_Bs(B_folder, A_folder,'data', A_out_folder, method='move')