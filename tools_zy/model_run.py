import subprocess
import datetime
from PIL import Image
import torch
# from torch.utils.tensorboard import SummaryWriter
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets, models
# from efficientnet_pytorch import EfficientNet
import csv
# import matplotlib.pyplot as plt
import os
import shutil
import random
import sys
import copy
import cv2
import numpy as np
# from skimage.metrics import structural_similarity
import torch.onnx
import torch.nn.functional as F


# 定义一个新的模型，它在原始模型的输出上增加了一个Softmax层
class CustomModel(nn.Module):
    def __init__(self, original_model):
        super(CustomModel, self).__init__()
        self.original_model = original_model
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.original_model(x)
        x = self.softmax(x)
        # 添加额外的维度
        x = x.unsqueeze(-1).unsqueeze(-1)
        return x



def run_cmd(command):
    # 输出命令
    command_str = ' '.join(command)
    print(command_str)
    # 记录操作
    model_log_folder = './log'
    model_log_name = 'run_cmd_log.txt'
    optime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if not os.path.exists(model_log_folder):
        os.makedirs(model_log_folder)
    with open(model_log_folder + '/' + model_log_name, "a") as f:
        f.write(optime + '\t' + command_str + '\n')
    # 运行
    subprocess.call(command)


def filter(src_top_folder, similar):
    src_top_folder = os.path.normpath(src_top_folder)
    f_len = len(src_top_folder)
    dst_top_folder = src_top_folder + '_filtered'
    if os.path.exists(dst_top_folder):
        shutil.rmtree(dst_top_folder)

    N = 0
    n = 0
    for src_folder, dirs, files in os.walk(src_top_folder):
        dst_folder = os.path.join(dst_top_folder, src_folder[f_len + 1:])
        print('%s\t==>\t%s' % (src_folder, dst_folder))
        print('=' * 200)
        os.mkdir(dst_folder)
        print('selected  processed\tprocessing')
        print('-' * 200)
        files.sort()
        if len(files):
            while True:
                i = 0
                img1 = cv2.imread(os.path.join(src_folder, files[i]))
                if img1 is not None:
                    src_file = os.path.join(src_folder, files[i])
                    dst_file = os.path.join(dst_folder, files[i])
                    shutil.copy(src_file, dst_file)
                    # print(src_file, '\n', dst_file)
                    img1 = cv2.resize(img1, [500, 500])
                    break
                i += 1
            img1 = cv2.GaussianBlur(img1, (7, 7), 0)
        for file in files:
            src_file = os.path.join(src_folder, file)
            print('%6d  /  %6d\t%s' % (n, N, file), end='\r')
            img2 = cv2.imread(src_file)
            if img2 is not None:
                img2 = cv2.resize(img2, [500, 500])
                img2 = cv2.GaussianBlur(img2, (7, 7), 0)
                (ssim, diff) = structural_similarity(img1, img2, multichannel=True, full=True)
                if ssim < similar:
                    dst_file = os.path.join(dst_folder, file)
                    shutil.copy(src_file, dst_file)
                    n += 1
                    print('%6d  /  %6d\t%s' % (n, N, file))
                    img1 = img2
            N += 1

    print('=' * 200)
    print('%6d  /  %6d\tprocessed. All done.' % (n, N))

def check_sequential_folders(folder_path):
    folder_list = sorted(os.listdir(folder_path))
    count = 0
    for _, folder_name in enumerate(folder_list):
        expected_name = str(count)
        if folder_name == expected_name:
            count += 1
    return count

def split_classify_train_test(top_path, test_ratio=0.2, num_class=2):
    train_path = os.path.join(top_path, "train")
    test_path = os.path.join(top_path, "test")
    if os.path.exists(train_path):
        print("Error: %s exists, EXIT with nothing done" % (train_path))
        exit(-1)
    if os.path.exists(test_path):
        print("Error: %s exists, EXIT with nothing done" % (test_path))
        exit(-1)
    os.mkdir(test_path)
    os.mkdir(train_path)

    for i in range(num_class):
        count = 0
        class_name = str(i)
        original_img_path = os.path.join(top_path, class_name)
        if not os.path.exists(original_img_path):
            print("Error: %s does NOT exists, please check" % (original_img_path))
            continue
        test_img_path = os.path.join(test_path, class_name)
        os.mkdir(test_img_path)
        imgs = os.listdir(original_img_path)
        length = len(imgs)
        test_num = int(test_ratio * length)
        random.shuffle(imgs)
        for j in range(test_num):
            original_img_fullname = os.path.join(original_img_path, imgs[j])
            test_img_fullname = os.path.join(test_img_path, imgs[j])
            shutil.move(original_img_fullname, test_img_fullname)
            count += 1
            print("moved %d/%d from %d imgs:\t%s " % (count, test_num, length, original_img_fullname))
        shutil.move(original_img_path, top_path + "/train/" + class_name)


# from custom_dataset import CustomDataset
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, images_path, transform):
        self.images_path = images_path
        self.transform = transform

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, index):
        path = self.images_path[index]
        img = Image.open(path).convert('RGB')
        img = self.transform(img)
        return img, path


def classify_move(model_path, data_folder, image_size=224, batch_size=32, conf_threshold=0.90, num_workers=8, noNM=False):
    # 将训练模型转移到 GPU 上（如果可用）
    assert torch.cuda.is_available()
    device = torch.device("cuda:0")
    model = torch.load(model_path)
    model.to(device)

    # 定义数据预处理函数
    if noNM:
        test_transforms = transforms.Compose([
            transforms.Resize(size=(image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0, 0, 0], [0.229, 0.224, 0.225])
        ])
    else:
        test_transforms = transforms.Compose([
            transforms.Resize(size=(image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    # 创建分类图片放置的文件夹
    num_classes = model._fc.out_features
    for i in range(num_classes):
        os.makedirs(os.path.join(data_folder, str(i)), exist_ok=True)

    # 创建数据加载器
    test_images_path = [f.path for f in os.scandir(data_folder) if
                        f.is_file() and f.name.endswith(('.jpg', '.jpeg', '.png'))]
    test_dataset = CustomDataset(test_images_path, transform=test_transforms)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # 进行推理
    model.eval()
    with torch.no_grad():
        for inputs, paths in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            # print(f"probabilities: {probabilities}")
            _, predictions = torch.max(probabilities, 1)
            for i, (path, prediction) in enumerate(zip(paths, predictions.cpu().numpy())):
                print(i,prediction, probabilities[i, prediction])
                if probabilities[i, prediction] > conf_threshold:
                    shutil.move(path, os.path.join(data_folder, str(prediction), os.path.basename(path)))
    print("move over: " + data_folder)

def classify_check(model_path, data_folder, image_size=224, batch_size=32, conf_threshold=0.90, num_workers=8, noNM=False):
    # 将训练模型转移到 GPU 上（如果可用）
    assert torch.cuda.is_available()
    device = torch.device("cuda:0")
    model = torch.load(model_path)
    model.to(device)

    # 定义数据预处理函数
    if noNM:
        test_transforms = transforms.Compose([
            transforms.Resize(size=(image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0, 0, 0], [0.229, 0.224, 0.225])
        ])
    else:
        test_transforms = transforms.Compose([
            transforms.Resize(size=(image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def check_one_folder(folder_path, num_i, conf):
        # 检查一个文件夹(path\num)中是不是都是某一类(num_i)
        up_folder = os.path.dirname(folder_path)
        # 创建数据加载器
        images_path = [f.path for f in os.scandir(folder_path) if f.is_file() and f.name.endswith(('.jpg', '.jpeg', '.png'))]
        test_dataset = CustomDataset(images_path, transform=test_transforms)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        # 进行推理
        model.eval()
        with torch.no_grad():
            for inputs, paths in test_loader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                _, predictions = torch.max(probabilities, 1)
                for i, (path, pred_i) in enumerate(zip(paths, predictions.cpu().numpy())):
                    if pred_i != num_i:
                        print("find")
                        if probabilities[i, pred_i] > conf:
                            os.makedirs(os.path.join(up_folder, f"{str(num_i)}-{str(pred_i)}_-{str(conf)}"), exist_ok=True)
                            shutil.move(path, os.path.join(up_folder, f"{str(num_i)}-{str(pred_i)}_-{str(conf)}", os.path.basename(path)))
                        else:
                            os.makedirs(os.path.join(up_folder, f"{str(num_i)}-{str(pred_i)}_+{str(conf)}"), exist_ok=True)
                            shutil.move(path, os.path.join(up_folder, f"{str(num_i)}-{str(pred_i)}_+{str(conf)}", os.path.basename(path)))

    # 遍历文件夹
    num_classes = model._fc.out_features
    if os.path.exists(os.path.join(data_folder, 'train')):
        for train_test in ('train', 'test'):
            for num_i in range(num_classes):
                print(f"check {train_test}/{num_i} ...")
                sub_sub_folder = os.path.join(data_folder, train_test, str(num_i))
                check_one_folder(sub_sub_folder, num_i, conf_threshold)
    else:
        for num_i in range(num_classes):
            print(f"check {num_i} ...")
            sub_sub_folder = os.path.join(data_folder, str(num_i))
            check_one_folder(sub_sub_folder, num_i, conf_threshold)
    print("check over")


def plot_train_log(log_txt, out_img_name='metrics.png'):
    # 从 log.txt 文件中读取指标
    with open(log_txt, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # 跳过标题行
        epochs, train_losses, train_accs, test_losses, test_accs = [], [], [], [], []
        for row in reader:
            epoch, train_loss, train_acc, test_loss, test_acc = [float(item) for item in row]
            epochs.append(epoch)
            train_losses.append(train_loss)
            train_accs.append(train_acc)
            test_losses.append(test_loss)
            test_accs.append(test_acc)

    # 绘制图像
    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.plot(epochs, train_accs, color=color, label='Train Acc')
    ax1.plot(epochs, test_accs, '--', color=color, label='Test Acc')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # 共享 x 轴
    color = 'tab:blue'
    ax2.set_ylabel('Losses')
    ax2.plot(epochs, train_losses, color=color, label='Train Loss')
    ax2.plot(epochs, test_losses, '--', color=color, label='Test Loss')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('Training and Validation Metrics')
    fig.tight_layout()  # 调整布局以防止标签重叠
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    # 保存图像并不显示
    plt.savefig(os.path.join(os.path.dirname(log_txt), out_img_name), dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()


def create_data(data_folder, image_size=224, batch_size=16, num_workers=8, noNM=False):
    # 定义数据增强操作
    if noNM:
        train_transforms = transforms.Compose([
            transforms.RandomRotation(degrees=5),
            transforms.ColorJitter(0.1, 0.1, 0.1, 0.01),
            transforms.GaussianBlur(5, sigma=(0.1, 3.0)),  # 对图像进行高斯模糊处理
            transforms.Resize(size=(image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0, 0, 0], [0.229, 0.224, 0.225])
        ])
        test_transforms = transforms.Compose([
            transforms.Resize(size=(image_size, image_size)),
            # transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize([0, 0, 0], [0.229, 0.224, 0.225])
        ])
    else:
        train_transforms = transforms.Compose([
            transforms.RandomRotation(degrees=5),
            transforms.ColorJitter(0.2, 0.2, 0.1, 0.01),
            transforms.GaussianBlur(5, sigma=(0.1, 3.0)),  # 对图像进行高斯模糊处理
            transforms.Resize(size=(image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        test_transforms = transforms.Compose([
            transforms.Resize(size=(image_size, image_size)),
            # transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    # 加载数据集并进行数据增强处理
    train_dataset = datasets.ImageFolder(root=data_folder + '\\train', transform=train_transforms)
    test_dataset = datasets.ImageFolder(root=data_folder + '\\test', transform=test_transforms)

    # 创建数据加载器
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader


def classify_train(event_name, data_folder, model_save_folder, pre_model='efficientnet-b0', image_size=224,
                   epochs=100, batch_size=16, lr=2e-4, weight_decay=1e-5, log_txt: str = None, num_workers=8, max_no_better=20, noNM=False):
    num_classes = check_sequential_folders(data_folder + '/train')
    train_detail = f" vent_name: {event_name}\n num_classes: {num_classes}\n data_folder: {data_folder}\n model_save_folder: {model_save_folder}\n pre_model: {pre_model}\n image_size={image_size} epochs={epochs} batch_size={batch_size} lr={lr} weight_decay={weight_decay} log_txt={log_txt} num_workers={num_workers}"
    print(train_detail)
    print("=" * 100)
    if not os.path.exists(model_save_folder):
        os.makedirs(model_save_folder)
    # 将预训练模型转移到GPU上（如果CUDA不可用，退出）
    assert torch.cuda.is_available(), "cuda not found."
    device = torch.device("cuda:0")

    if pre_model == 'efficientnet-b0':
        # 加载预训练模型 EfficientNet-B0
        pretrained_model = EfficientNet.from_pretrained(pre_model)
        # 更改最后一层全连接层
        num_features = pretrained_model._fc.in_features
        pretrained_model._fc = nn.Linear(num_features, num_classes)
    else:
        pretrained_model = torch.load(pre_model)
    pretrained_model.to(device)
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(pretrained_model.parameters(), lr=lr, weight_decay=weight_decay)
    # 加载数据
    train_loader, test_loader = create_data(data_folder, image_size, batch_size=batch_size, num_workers=num_workers, noNM=noNM)
    # 进行训练和测试
    best_test_acc = 0.0
    # 创建log文件, 如果文件已经存在，则在文件名后加上序号，直到生成一个不存在的文件名
    if log_txt is not None:
        log_txt_path = os.path.join(model_save_folder, log_txt)
        if os.path.exists(log_txt_path):
            name, ext = os.path.splitext(log_txt)
            i = 1
            while True:
                log_txt_path = os.path.join(model_save_folder, f'{name}_{i}{ext}')
                if not os.path.exists(log_txt_path):
                    break
                i += 1
        with open(log_txt_path, 'a') as f:
            f.write(f"Epoch,Train Loss,Train Acc,Test Loss,Test Acc\n")
        # 记录训练参数
        with open(os.path.join(model_save_folder, f'train_parameter_log.txt'),'a') as f:
            current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            train_detail = f"\tvent_name:\t{event_name}\n\tnum_classes:\t{num_classes}\n\tdata_folder:\t{data_folder}\n\t" \
                           f"model_save_folder:\t{model_save_folder}\n\tpre_model:\t{pre_model}\n\timage_size={image_size}\n\t" \
                           f"epochs={epochs}\n\tbatch_size={batch_size}\n\tlr={lr}\n\tweight_decay={weight_decay}\n\tlog_txt={log_txt}\n\tnum_workers={num_workers}"
            f.write('=' * 80 + '\n')
            f.write("train time: " + current_time + "\n")
            f.write(train_detail + '\n')
            f.write('train log: ' + log_txt_path + '\n')
    not_best_count = 0
    for epoch in range(epochs):
        # -------- 训练模型
        pretrained_model.train()
        train_loss, train_acc = 0.0, 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = pretrained_model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
            _, predictions = torch.max(outputs.data, 1)
            train_acc += (predictions == labels).sum().item()
        train_loss /= len(train_loader.dataset)
        train_acc /= len(train_loader.dataset)
        # --------- 测试模型
        pretrained_model.eval()
        test_loss, test_acc = 0.0, 0.0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = pretrained_model(images)
                loss = criterion(outputs, labels)
                test_loss += loss.item() * images.size(0)
                _, predictions = torch.max(outputs.data, 1)
                test_acc += (predictions == labels).sum().item()
        test_loss /= len(test_loader.dataset)
        test_acc /= len(test_loader.dataset)
        # print('-')
        # 打印并保存模型
        if noNM:
            model_name = f"{event_name}_enb0_C{num_classes}_E{epoch + 1}_{train_loss:.5f}_{train_acc:.5f}_{test_loss:.5f}_{test_acc:.5f}_noNM.pt"
        else:
            model_name = f"{event_name}_enb0_C{num_classes}_E{epoch + 1}_{train_loss:.5f}_{train_acc:.5f}_{test_loss:.5f}_{test_acc:.5f}.pt"
        print(
            f"{event_name:6s} Epoch {epoch + 1}/{epochs} Train Loss: {train_loss:.5f}, Train Acc: {train_acc:.5f}, Test Loss: {test_loss:.5f}, Test Acc: {test_acc:.5f}")
        # 如果连续多个epoch都没有提升，则停止训练
        if test_acc > best_test_acc:
            not_best_count = 0
        else:
            not_best_count += 1
        if test_acc >= best_test_acc:
            best_test_acc = test_acc
            torch.save(pretrained_model, os.path.join(model_save_folder, model_name))
        # 保存日志
        if log_txt is not None:
            with open(log_txt_path, 'a') as f:
                f.write(f"{epoch + 1},{train_loss:.5f},{train_acc:.5f},{test_loss:.5f},{test_acc:.5f}\n")
            plot_train_log(log_txt_path)
        if not_best_count >= max_no_better:
            break
        # 如果测试精度达到100%，并且训练精度达到99%，则停止训练
        if test_acc >= 1.0 and train_acc >= 0.999:
            break
    return os.path.join(model_save_folder, model_name)


def classify_train_v2(event_name, data_folder, model_save_folder, pre_model='efficientnet-b3', image_size=224,
                   epochs=100, batch_size=16, lr=2e-4, weight_decay=1e-5, log_txt: str = None, num_workers=8):
    num_classes = check_sequential_folders(data_folder + '/train')
    train_detail = f" vent_name: {event_name}\n num_classes: {num_classes}\n data_folder: {data_folder}\n model_save_folder: {model_save_folder}\n pre_model: {pre_model}\n image_size={image_size} epochs={epochs} batch_size={batch_size} lr={lr} weight_decay={weight_decay} log_txt={log_txt} num_workers={num_workers}"
    print(train_detail)
    print("=" * 100)
    if not os.path.exists(model_save_folder):
        os.makedirs(model_save_folder)
    # 将预训练模型转移到GPU上（如果CUDA不可用，退出）
    assert torch.cuda.is_available(), "cuda not found."
    device = torch.device("cuda:0")

    if pre_model == 'efficientnet-b3':
        # 加载预训练模型 EfficientNet-B0
        pretrained_model = EfficientNet.from_pretrained(pre_model)
        # 更改最后一层全连接层
        num_features = pretrained_model._fc.in_features
        pretrained_model._fc = nn.Linear(num_features, num_classes)
    else:
        pretrained_model = torch.load(pre_model)
    pretrained_model.to(device)
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(pretrained_model.parameters(), lr=lr, weight_decay=weight_decay)
    # 加载数据
    train_loader, test_loader = create_data(data_folder, image_size, batch_size=batch_size, num_workers=num_workers)
    # 进行训练和测试
    best_test_acc = 0.0
    # 创建log文件, 如果文件已经存在，则在文件名后加上序号，直到生成一个不存在的文件名
    if log_txt is not None:
        log_txt_path = os.path.join(model_save_folder, log_txt)
        if os.path.exists(log_txt_path):
            name, ext = os.path.splitext(log_txt)
            i = 1
            while True:
                log_txt_path = os.path.join(model_save_folder, f'{name}_{i}{ext}')
                if not os.path.exists(log_txt_path):
                    break
                i += 1
        with open(log_txt_path, 'a') as f:
            f.write(f"Epoch,Train Loss,Train Acc,Test Loss,Test Acc\n")
        # 记录训练参数
        with open(os.path.join(model_save_folder, f'train_parameter_log.txt'),'a') as f:
            current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            train_detail = f"\tvent_name:\t{event_name}\n\tnum_classes:\t{num_classes}\n\tdata_folder:\t{data_folder}\n\t" \
                           f"model_save_folder:\t{model_save_folder}\n\tpre_model:\t{pre_model}\n\timage_size={image_size}\n\t" \
                           f"epochs={epochs}\n\tbatch_size={batch_size}\n\tlr={lr}\n\tweight_decay={weight_decay}\n\tlog_txt={log_txt}\n\tnum_workers={num_workers}"
            f.write('=' * 80 + '\n')
            f.write("train time: " + current_time + "\n")
            f.write(train_detail + '\n')
            f.write('train log: ' + log_txt_path + '\n')
    not_best_count = 0
    for epoch in range(epochs):
        # -------- 训练模型
        pretrained_model.train()
        train_loss, train_acc = 0.0, 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = pretrained_model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
            _, predictions = torch.max(outputs.data, 1)
            train_acc += (predictions == labels).sum().item()
        train_loss /= len(train_loader.dataset)
        train_acc /= len(train_loader.dataset)
        # --------- 测试模型
        pretrained_model.eval()
        test_loss, test_acc = 0.0, 0.0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = pretrained_model(images)
                loss = criterion(outputs, labels)
                test_loss += loss.item() * images.size(0)
                _, predictions = torch.max(outputs.data, 1)
                test_acc += (predictions == labels).sum().item()
        test_loss /= len(test_loader.dataset)
        test_acc /= len(test_loader.dataset)
        # print('-')
        # 打印并保存模型
        model_name = f"{event_name}_enb0_C{num_classes}_E{epoch + 1}_{train_loss:.5f}_{train_acc:.5f}_{test_loss:.5f}_{test_acc:.5f}.pt"
        print(
            f"{event_name:6s} Epoch {epoch + 1}/{epochs} Train Loss: {train_loss:.5f}, Train Acc: {train_acc:.5f}, Test Loss: {test_loss:.5f}, Test Acc: {test_acc:.5f}")
        # 如果连续多个epoch都没有提升，则停止训练
        if test_acc > best_test_acc:
            not_best_count = 0
        else:
            not_best_count += 1
        if test_acc >= best_test_acc:
            best_test_acc = test_acc
            torch.save(pretrained_model, os.path.join(model_save_folder, model_name))
        # 保存日志
        if log_txt is not None:
            with open(log_txt_path, 'a') as f:
                f.write(f"{epoch + 1},{train_loss:.5f},{train_acc:.5f},{test_loss:.5f},{test_acc:.5f}\n")
            plot_train_log(log_txt_path)
        if not_best_count >= 15:
            break
        # 如果测试精度达到100%，并且训练精度达到99%，则停止训练
        if test_acc >= 1.0 and train_acc >= 0.999:
            break
    return model_save_folder


def pt2onnx_for_classify(pt_path):
    assert pt_path[-3:] == ".pt"
    out_path = pt_path[:-3] + ".onnx"
    pretrained_model = torch.load(pt_path)
    input_tensor = torch.randn(1, 3, 224, 224, device='cuda')
    torch.onnx.export(pretrained_model, input_tensor, out_path, input_names=["input"], output_names=["output"])
    print("pt2onnx path: " + out_path)
    return out_path





def pt2onnx_sgie_for_classify(pt_path):
    assert pt_path[-3:] == ".pt", "Input file must be a .pt file"

    # 加载PyTorch模型
    pretrained_model = torch.load(pt_path, map_location='cuda')

    # 如果模型是nn.Module的子类，则可以直接修改，否则需要根据模型的具体实现进行调整
    if isinstance(pretrained_model, nn.Module):
        # 包装原始模型以修改输出维度
        class WrappedModel(nn.Module):
            def __init__(self, model):
                super(WrappedModel, self).__init__()
                self.model = model

            def forward(self, x):
                x = self.model(x)
                # 修改输出维度为 [n, 4, 1, 1]
                x = x.view(x.size(0), -1, 1, 1)
                return x

        # 封装模型
        wrapped_model = WrappedModel(pretrained_model)

        # 设置模型为评估模式
        wrapped_model.eval()

        # 创建一个示例输入张量
        input_tensor = torch.randn(1, 3, 224, 224, device='cuda')

        # 定义ONNX文件输出路径
        out_path = pt_path[:-3] + ".onnx"

        # 导出模型到ONNX
        torch.onnx.export(wrapped_model, input_tensor, out_path, input_names=["input_1"], output_names=["predictions/Softmax"],
                          dynamic_axes={'input_1': {0: 'batch_size'}, 'predictions/Softmax': {0: 'batch_size'}})

        print("ONNX model saved to: " + out_path)
        return out_path
    else:
        print("Model is not an instance of nn.Module. Custom modifications might be required.")
        return None

# 为模型添加Softmax层
def add_softmax(in_model_path, out_model_path=None):
    if out_model_path is None:
        out_model_path = in_model_path.split('.pt')[0] + '_softmax.pt'
    model = torch.load(in_model_path)
    model.eval()
    custom_model = CustomModel(model)
    torch.save(custom_model, out_model_path)
    print("out model: " + out_model_path)





# 示例调用
# pt2onnx_sgie_for_classify("path/to/your/model.pt")


if __name__ == '__main__':
    # 这是用于图片的分类，把图片放到对应结果的文件夹中
    model_path = r"F:\nanyang\reflect_classify_data_model\model\classify_nanyang_vest\vest_enb0_C4_E153_0.00224_0.99833_0.10103_0.98759.pt"
    data_folder = r"C:\Users\feeyo\Downloads\分类\分类\Reflect_Data\noconfidence"
    # classify_move(model_path, data_folder, image_size=224, batch_size=16, conf_threshold=0.90)

    # 也可以运行命令
    command_test = ["python", "--version"]
    run_cmd(command_test)

    # 数据文件夹data_folder(包含有test和train两个文件夹)
    event_name = "vest"
    num_classes = 4
    data_folder = r"C:\Users\feeyo\Desktop\temp\classify_nanyang_vest\vest",
    model_out_folder = r"F:\nanyang\reflect_classify_data_model\model\classify_nanyang_vest\vest"
    pre_model = r"F:\nanyang\reflect_classify_data_model\model\classify_nanyang_vest\vest_enb0_C4_E153_0.00224_0.99833_0.10103_0.98759.pt"
    # classify_train("vest", num_classes, data_folder, model_out_folder, pre_model=pre_model, epochs=200, batch_size=40)


    # pt2onnx_sgie_for_classify(r"D:\Zhiyuan\pic_data_20231229\_raw_apron\Finall_20230107\561_res2\crops\vest\vest_enb0_C4_E9_0.00552_1.00000_0.00072_1.00000.pt")