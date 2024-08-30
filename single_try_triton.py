import os
import numpy as np
import tritonclient.grpc as grpcclient
from PIL import Image


def triton_classify_pic(ip_port, )


input_name = 'input'
output_name = 'output'
event_name = "PCC"


def softmax(x):
    # 对输入数据进行指数函数的运算
    exp_x = np.exp(x)
    # 对每一行进行求和，并将结果扩展为与输入数据相同的形状
    sum_exp_x = np.sum(exp_x, axis=1, keepdims=True)
    # 对指数函数的结果进行除法运算，得到 softmax 的输出结果
    softmax_output = exp_x / sum_exp_x
    return softmax_output


triton_client = grpcclient.InferenceServerClient(
    url='192.168.102.118:8001',
    verbose=False,
    ssl=False,
    root_certificates=None,
    private_key=None,
    certificate_chain=None)

pic_dir = r"D:\Zhiyuan\pic_data_20231229\PCC\WMT\20231211\test\1"

count = 1
for root, directory, files in os.walk(pic_dir):
    for filename in files:  ## 遍历当前目录下的所有文件
        name, suf = os.path.splitext(filename)  ## 拆开为文件名+后缀
        if suf == ".jpg":  ## 如果是目标类型文件
            jpg_full_path_name = os.path.join(root, filename)  ## 拼接目录并保存
        else:
            continue
        image_data = Image.open(jpg_full_path_name)
        image_data = image_data.resize((224, 224))
        np_img = np.array(image_data)

        normalized_image = (np_img / 255.0 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        input_data = normalized_image.astype(np.float32)
        input_data = np.expand_dims(input_data, axis=0)
        input_data = np.transpose(input_data, (0, 3, 1, 2))
        inputs = [grpcclient.InferInput(input_name, [1, 3, 224, 224], "FP32")]
        inputs[0].set_data_from_numpy(input_data)
        outputs = [grpcclient.InferRequestedOutput(output_name)]
        results = triton_client.infer(model_name=event_name, inputs=inputs, outputs=outputs, timeout=2)
        predictions = results.as_numpy(output_name)
        predictions = softmax(predictions)[0]
        i = np.argmax(predictions)
        print(i,predictions[i])


