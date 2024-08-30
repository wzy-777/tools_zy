import os
import shutil

import onnxruntime as ort
import numpy as np
from PIL import Image
from torchvision import transforms

model_path = r"C:\Users\Lenovo\Downloads\sp002det_202400827_512x256.onnx"
# pic = r"D:\Zhiyuan\pic_data_20231229\_raw_apron\Finall_20230107\561_res2\crops\test\Image53 (3).jpg"
pic_fold = r"D:\Zhiyuan\work_tools_py\files\2024-08-27"

for pic_name in os.listdir(pic_fold):
    pic = os.path.join(pic_fold, pic_name)
    # 定义预处理
    image_size = 224  # 根据您的模型需要设置
    test_transforms = transforms.Compose([
        transforms.Resize(size=(image_size, image_size)),
        transforms.ToTensor(),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 加载ONNX模型
    sess = ort.InferenceSession(model_path)

    # 加载图像并应用预处理
    img = Image.open(pic).convert("RGB")
    img_transformed = test_transforms(img)
    img_transformed = np.expand_dims(img_transformed.numpy(), 0)  # 添加批次维度

    # 进行推理
    input_name = sess.get_inputs()[0].name
    output = sess.run(None, {input_name: img_transformed})

    print(output)
    # 输出是softmax概率，获取最可能的类别
    predicted_class = np.argmax(output[0])

    print(f"Predicted class: {predicted_class}")
    # 创建对应类别的文件夹，将该文件放入对应类别的文件夹
    if not os.path.exists(os.path.join(pic_fold, str(predicted_class))):
        os.makedirs(os.path.join(pic_fold, str(predicted_class)))
    shutil.move(pic, os.path.join(pic_fold, str(predicted_class), os.path.basename(pic_name)))




