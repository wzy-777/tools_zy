from tools_zy_bak import yolo_need
import os
import multiprocessing as mp
import shutil

if __name__ == '__main__':
    # 设置启动方法为'spawn'
    mp.set_start_method('spawn')
    # 创建进程池，根据CPU数量创建对应数量的进程
    # pool = mp.Pool(mp.cpu_count())
    pool = mp.Pool(4)

    # object_list = ['tou', 'shouji', 'diexingbaozhawu', 'fenmobaozhawu', 'juxingbaozhawu', 'taocidao', 'jinshudao', 'dahuoji', 'yeti', 'zidan', 'qiang', 'luyinbi', 'fangxingluyinbi', 'xiaoluosidao', 'zhijiadao', 'zhusheqi', 'masheng', 'banshou', 'yanhe', 'zhibi', 'Upan', 'xisheng', 'shexiangtou']
    # object_dict = {index: folder_name for index, folder_name in enumerate(object_list)}
    # # 将txt转为xml文件
    # txtfolder = r"D:\Zhiyuan\pic_data_20231229\__detect\HUZ\detect_HUZ_20240520_1_txt\labels"
    # picfolder = r"D:\Zhiyuan\pic_data_20231229\__detect\HUZ\detect_HUZ_20240520_1"
    # dict = {0: 'FJ', 1: 'person', 2: 'cone', 3: 'KTC', 4: 'QYC', 5: 'CSC', 6: 'PTC', 7: 'JYC', 8: 'KCM', 9: 'HCM', 10: 'LQ', 11: 'PCC', 12: 'workladder', 13: 'XLC', 14: 'PBC', 15: 'YDC', 16: 'other'}  # 在class.txt里面
    # xmlfolder = r"D:\Zhiyuan\pic_data_20231229\__detect\HUZ\detect_HUZ_20240520_1_xml"  # 没有会创建新的
    # for filename in os.listdir(txtfolder):
    #     pool.apply_async(yolo_need.txt2xml, args=(os.path.join(txtfolder, filename), picfolder, dict, xmlfolder, "20240508"))


    # 根据xml文件查找图片
    image_folder = r"D:\Zhiyuan\图片采集\data\外包采集图像\ok\temp\images"
    xml_folder = r"D:\Zhiyuan\图片采集\data\外包采集图像\ok\temp\Annotations"
    # out_folder = r"C:\Users\feeyo\Desktop\FlightData_20230213\1HCM"  # 没有会创建新的
    out_folder = r"D:\Zhiyuan\图片采集\data\外包采集图像\ok\temp\image"
    # for filename in os.listdir(xml_folder):
    #     pool.apply_async(yolo_need.find_imgs_with_xml, args=(image_folder, filename, out_folder, 'bmp', 'move'))


    # 合并xml文件
    images_folder = r"D:\Zhiyuan\pic_data_20231229\__detect\JJN\20240506\03_04_10"
    xmls_folder_list = [
        r'D:\Zhiyuan\pic_data_20231229\__detect\JJN\20240506\cone',
        r'D:\Zhiyuan\pic_data_20231229\__detect\JJN\20240506\CSC',
        r'D:\Zhiyuan\pic_data_20231229\__detect\JJN\20240506\FJ',
        r'D:\Zhiyuan\pic_data_20231229\__detect\JJN\20240506\HCM',
        r'D:\Zhiyuan\pic_data_20231229\__detect\JJN\20240506\JYC',
        r'D:\Zhiyuan\pic_data_20231229\__detect\JJN\20240506\KCM',
        r'D:\Zhiyuan\pic_data_20231229\__detect\JJN\20240506\KTC',
        r'D:\Zhiyuan\pic_data_20231229\__detect\JJN\20240506\LQ',
        r'D:\Zhiyuan\pic_data_20231229\__detect\JJN\20240506\PBC',
        r'D:\Zhiyuan\pic_data_20231229\__detect\JJN\20240506\PCC',
        r'D:\Zhiyuan\pic_data_20231229\__detect\JJN\20240506\person',
        r'D:\Zhiyuan\pic_data_20231229\__detect\JJN\20240506\QYC',
        r'D:\Zhiyuan\pic_data_20231229\__detect\JJN\20240506\workladder',
        r'D:\Zhiyuan\pic_data_20231229\__detect\JJN\20240506\XLC',
        r'D:\Zhiyuan\pic_data_20231229\__detect\JJN\20240506\YDC',
    ]
    output_folder = r'D:\Zhiyuan\pic_data_20231229\__detect\JJN\20240506\Annotations'
    # for filename in os.listdir(images_folder):
    #     pool.apply_async(yolo_need.merge_multi_classes_label_xml,
    #                      args=(os.path.join(images_folder, filename), xmls_folder_list, output_folder, None))

    # 关闭并等待所有进程完成
    pool.close()
    pool.join()
