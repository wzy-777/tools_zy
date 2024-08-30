import os

from tools_wzy import file_processing
import multiprocessing as mp

if __name__ == '__main__':
    mp.set_start_method('spawn')  # 设置启动方法为'spawn'

    # 将一个文件夹下的所有图片移动到新的文件夹（注意：两个输入不能有嵌套关系）,src_dir和dst_dir可以写一样的
    src_folder = r"D:\Zhiyuan\图片采集\data\外包采集图像\ok\Annotations_bak"
    dst_folder = r"D:\Zhiyuan\图片采集\data\外包采集图像\ok\Annotations_bak"
    # file_processing.move_images(src_folder, dst_folder, file_format='.xml')  # file_format默认为('.png', '.jpg', '.jpeg', '.gif', '.bmp')


    # 将src_dir文件夹下所有图片分成每个文件夹下x张，src_dir和dst_dir可以写一样的
    src_dir = r"D:\Zhiyuan\pic_data_20231229\__detect\HUZ\20240520"
    dst_dir = r"D:\Zhiyuan\pic_data_20231229\__detect\HUZ\20240520"
    x = 500  # 每个目标文件夹中的图像文件数量
    # file_processing.split_pics_to_folders(src_dir, dst_dir, x)


    # 在folder2中找到与folder1中重复的图片，并删除folder2中重复的
    folder1 = r'D:\Zhiyuan\pic_data\nanyang_pd140_vest_crops\train\0'
    folder2_for_remove = r'D:\Zhiyuan\pic_data\nanyang_pd140_vest_crops\test\0'
    # file_processing.remove_same_pic(folder1, folder2_for_remove)


    # 删除文件夹中，长宽非常小的图片
    # 定义路径和最小像素大小
    folder_path = r"D:\Zhiyuan\models_result\pd561_202307_01-17_front_vest_detect_result\crops\person"
    min_width, min_height = 14, 16
    # file_processing.delete_small_pics(folder_path, min_width, min_height)


    # 图片去重，挑选出相似度小于阈值的，想要运行快，每个文件夹中图片应该少点
    pic_top_folder = r"D:\Zhiyuan\pic_data_20231229\__detect\HUZ\20240520"
    sim_threshold = 0.95
    # file_processing.filter_mp(pic_top_folder, sim_threshold, method='move', size=[1920, 1080], multi_process=9, dst_top_folder=None)
    # method为导出高相似度图片的方式, size为[宽, 高], [320, 180]


    # 从json文件中提取机位范围的多边形配置
    json_file = r"D:\Zhiyuan\pic_data\wingtip\140\s"
    json_folder = r"D:\Zhiyuan\pic_data\wingtip\561front"
    json_folder_folder = r"C:\Users\feeyo\Desktop\config"
    # file_processing.get_apron_poly_from_json(json_file)
    # file_processing.get_apron_polys_from_folder(json_folder)
    # file_processing.get_apron_polys_from_folder_foler(json_folder_folder)


    # 合并excel文件整体覆盖率
    folder_path = r'D:\Zhiyuan\node_coverage_excels\202308-202401'
    output_sheet = r'D:\Zhiyuan\node_coverage_excels\202308-202401_all.xlsx'
    # file_processing.read_and_merge_excel(folder_path, output_sheet, sheet_name='整体覆盖率')

    # 将文件夹下所有图片转为jpg格式
    folder_path = r'D:\Program Files\wxwork\WXWork\1688857814425476\Cache\File\2024-03\Downloads'
    # file_processing.convert_images_to_jpg(folder_path)

    # 从一个视频中提取帧
    video_path = r"D:\Zhiyuan\video\record\561_319_1246_20230701042908.mp4"
    save_dir = r"D:\Zhiyuan\video\record_frame"
    # file_processing.save_video_interval_frames(video_path, save_dir, interval=0.1)

    # 从多个视频中提取帧
    video_folder = r"F:\video\shop\T2-48004-国家地理4\2024-01-20"
    save_dir = r'F:\video\shop\T2-48004-国家地理4\2024-01-20_frame'
    # file_processing.save_videos_interval_frames(video_folder, save_dir, interval=3)


