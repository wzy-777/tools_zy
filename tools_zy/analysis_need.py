import cv2 as cv
import cv2
import numpy as np
import tritonclient.grpc as grpcclient
from shapely.geometry import Polygon
from nb_log import get_logger
import os

fmtlogger = get_logger('analysisBusi', is_add_stream_handler=True, log_filename='analysisBusi.log',
                       formatter_template=9)



def get_object_rect(OneAnalyser, src_img, node_config_poly, model_name, object_name):
    # 现在的函数只能获取模型中的一个目标
    # 返回区域内所有object的位置列表
    np_img = cv.cvtColor(np.asarray(src_img), cv.COLOR_RGB2BGR)
    # Find the maximum side of the image
    if np_img.shape[1] > np_img.shape[0]:
        max_side = np_img.shape[1]
    else:
        max_side = np_img.shape[0]
    dst_img = np.zeros([max_side, max_side, 3], dtype=np.uint8)
    dst_img[0:np_img.shape[0], 0:np_img.shape[1]] = np_img.copy()
    x_factor = dst_img.shape[1] / 1280
    y_factor = dst_img.shape[0] / 1280
    input_img = cv.resize(dst_img, [1280, 1280])
    input_img = input_img / 255.
    inputs = []
    # inputs.append(grpcclient.InferInput('images', [1, 3, 640, 640], "FP32"))
    inputs.append(grpcclient.InferInput('images', [1, 3, 1280, 1280], "FP32"))
    input_img = input_img.astype(np.float32).transpose((2, 0, 1))
    input_img = np.expand_dims(input_img, axis=0)  # 增加维度

    inputs[0].set_data_from_numpy(input_img)
    outputs = []
    outputs.append(grpcclient.InferRequestedOutput('output'))
    try:
        results = OneAnalyser.triton_client.infer(model_name=model_name, inputs=inputs, outputs=outputs, timeout=4,
                                                    compression_algorithm=None)
        predictions = results.as_numpy("output")[0]
        # rows = 25200
        rows = 100800
        # dimensions = 6
        scoreThreshold = 0.2
        nmsThreshold = 0.25
        confThreshold = 0.25
        class_name = [object_name]
        confidence_list = []
        class_list = []
        rect_list = []

        for i in range(rows):
            res = predictions[i]
            confidence = res[4]
            if confidence > confThreshold:
                # print(confidence)
                max = -1
                max_index = -1
                for j in range(len(class_name)):
                    if res[5 + j] > max:
                        max = res[5 + j]
                        max_index = j
                confidence_list.append(confidence)
                class_list.append(class_name[max_index])
                x = res[0]
                y = res[1]
                w = res[2]
                h = res[3]
                left = int((x - w / 2) * x_factor)
                top = int((y - h / 2) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)
                rect_list.append([left, top, width, height])

        if len(rect_list) > 0:
            nms_result = cv.dnn.NMSBoxes(np.array(rect_list), np.array(confidence_list), scoreThreshold, nmsThreshold)
        else:
            return list()

        r_l = []
        # n_l = []
        for k in range(len(nms_result)):
            rect_id = nms_result[k]
            rect = rect_list[rect_id]
            rect_change = (rect[0], rect[1], rect[0] + rect[2], rect[1] + rect[3])
            shape_rect = Polygon(
                [(rect[0], rect[1]),
                 (rect[0] + rect[2], rect[1]),
                 (rect[0] + rect[2], rect[1] + rect[3]),
                 (rect[0], rect[1] + rect[3])])
            shape_ploy = Polygon(node_config_poly)
            if shape_rect.intersects(shape_ploy):
                fmtlogger.debug("finded %s in config region rect[%d,%d,%d,%d]" % (
                    object_name, rect_change[0], rect_change[1], rect_change[2], rect_change[3]))
                r_l.append(rect_change)
            else:
                fmtlogger.debug("finded %s not in config region rect[%d,%d,%d,%d]" % (
                    object_name, rect_change[0], rect_change[1], rect_change[2], rect_change[3]))
            # name = class_list[id]
            # n_l.append(name)
        return r_l
    except Exception as e:
        fmtlogger.error("post analysis request failed {}".format(e))
        return list()

def get_frame(video_path, gap_s, out_floder):
    if not os.path.exists(out_floder):
        os.makedirs(out_floder)
    # 打开 MP4 文件，并获取视频属性
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 逐帧读取视频，将每一帧保存到图片文件
    frame_id = 0
    gap_fps = gap_s * fps
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # 保存当前帧到文件
        if frame_id % (gap_fps) == 0:
            filename = out_floder + f'\\frame_{int(frame_id/fps)}.jpg'
            cv2.imwrite(filename, frame)
            # 输出当前帧号和时间戳
            print(filename)
        # 更新帧号
        frame_id += 1
    # 释放资源
    cap.release()


