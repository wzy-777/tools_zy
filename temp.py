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

from PIL import Image

folder = r"D:\Zhiyuan\图片采集\data\演示案例"
for datai in os.listdir(folder):
    if not datai.endswith(".data"):
        continue
    # file_path = r"D:\Zhiyuan\图片采集\data\演示案例\2024-08-27-17-18-16-250-up-B--img-rgb.data"
    file_path = folder + '/' + datai
    width = 512
    height = 1024
    # 打开文件
    with open(file_path, 'rb') as file:
        # 读取文件内容
        binary_data = file.read()
    # 确保数据长度与预期的图像大小匹配
    expected_data_length = width * height * 3  # 3个颜色通道，每个像素3个字节
    if len(binary_data) != expected_data_length:
        raise ValueError(f"Data length mismatch. Expected {expected_data_length} bytes, got {len(binary_data)} bytes.")
    # 将读取的数据转换为图像
    image = Image.frombytes('RGB', (width, height), binary_data)
    # 保存图像
    save_path = file_path.replace(".data", ".jpg")
    save_path = save_path.replace("2024-08-27", "rgb_2024-08-27")
    image.save(save_path)
    # image.show()




