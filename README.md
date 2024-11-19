# tools_zy [![Version][version-badge]][version-link] ![MIT License][license-badge]


此工具包旨在给AI工作者提供一把趁手的工具，尽量简化非核心的工作。
其中包含深度学习模型过程中可能会用到的一些工具，比如数据整理、格式转换、划分数据集等。

因此也要注意，其中某些函数可能并没有提供过多自由度，如果需要，请自行修改。

### 安装

```bash
$ pip install tools-zy
```

### 使用

含有的功能示例如下：
```
import tools_zy as tz

# 复制、移动文件夹中以.bmp结尾的文件。（还支持指定文件名，支持递归操作）
tz.copy_file("/home/org_folder", "/home/new_folder", format=".bmp")
tz.move_file("/home/org_folder", "/home/new_folder", format=".bmp")

# 获取（复制、移动）文件夹中以.bmp结尾的一些随机文件。
tz.copy_some_random_files("/home/org_folder", "/home/new_folder", 1000, format='.bmp')

# 划分分类数据集
img_folder = r"/home/classify/rawData"
out_folder = r"/home/classify/splitData"
tz.split_classifid_images(img_folder, out_folder, (0.8, 0.2, 0), format=".bmp")
```




### License

[MIT](https://github.com/wzy-777/tools_zy/blob/main/LICENSE)


[version-badge]:   https://img.shields.io/badge/version-0.1-brightgreen.svg
[version-link]:    https://pypi.org/project/tools-zy/
[license-badge]:   https://img.shields.io/github/license/pythonml/douyin_image.svg