import os

import cv2

def clip_video(input_path, output_path, start_time, end_time):
    # 使用 OpenCV 库加载源视频文件
    video_capture = cv2.VideoCapture(input_path)

    # 计算开始和结束帧数
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)

    # 跳过开始帧之前的所有帧
    for i in range(start_frame):
        success, frame = video_capture.read()

    # 初始化输出视频编写器
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'AVC1'), fps, (640, 480))

    # 循环读取和写入视频帧
    for i in range(start_frame, end_frame):
        success, frame = video_capture.read()
        if not success:
            break
        out.write(frame)

    # 释放视频捕获器和输出编写器
    video_capture.release()
    out.release()




if __name__ == "__main__":
    pass
    clip_video(r"D:\Zhiyuan\video\video_crop\_T2-48002.mp4", r"D:\Zhiyuan\video\video_crop", 1, 720)

