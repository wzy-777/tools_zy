from xml.dom.minidom import Document
import os


def txt2xml(txtpath, picfolder, dict, xmlfolder, element_folder="xxx_folder"):  # 读取txt路径，xml保存路径，数据集图片所在路径
    if not os.path.exists(xmlfolder):
        os.makedirs(xmlfolder)
    name = os.path.basename(txtpath)
    # 读取图片和txt列表
    img = cv2.imread(os.path.join(picfolder, name[0:-4] + ".jpg"))
    # print(os.path.join(picfolder, name[0:-4] + ".jpg"))
    Pheight, Pwidth, Pdepth = img.shape

    with open(txtpath, 'r') as f:
        txtList = f.readlines()
        xmlBuilder = Document()
        # 创建annotation标签
        annotation = xmlBuilder.createElement("annotation")
        xmlBuilder.appendChild(annotation)
        # folder标签
        folder = xmlBuilder.createElement("folder")
        folderContent = xmlBuilder.createTextNode(element_folder)
        folder.appendChild(folderContent)
        annotation.appendChild(folder)
        # filename标签
        filename = xmlBuilder.createElement("filename")
        filenameContent = xmlBuilder.createTextNode(name[0:-4] + ".jpg")
        filename.appendChild(filenameContent)
        annotation.appendChild(filename)
        # size标签
        size = xmlBuilder.createElement("size")
        # size子标签width
        width = xmlBuilder.createElement("width")
        widthContent = xmlBuilder.createTextNode(str(Pwidth))
        width.appendChild(widthContent)
        size.appendChild(width)
        # size子标签height
        height = xmlBuilder.createElement("height")
        heightContent = xmlBuilder.createTextNode(str(Pheight))
        height.appendChild(heightContent)
        size.appendChild(height)
        # size子标签depth
        depth = xmlBuilder.createElement("depth")
        depthContent = xmlBuilder.createTextNode(str(Pdepth))
        depth.appendChild(depthContent)
        size.appendChild(depth)
        annotation.appendChild(size)

        for i in txtList:
            oneline = i.strip().split(" ")

            object = xmlBuilder.createElement("object")
            picname = xmlBuilder.createElement("name")
            nameContent = xmlBuilder.createTextNode(dict[int(oneline[0])])
            picname.appendChild(nameContent)
            object.appendChild(picname)
            pose = xmlBuilder.createElement("pose")
            poseContent = xmlBuilder.createTextNode("Unspecified")
            pose.appendChild(poseContent)
            object.appendChild(pose)
            truncated = xmlBuilder.createElement("truncated")
            truncatedContent = xmlBuilder.createTextNode("0")
            truncated.appendChild(truncatedContent)
            object.appendChild(truncated)
            difficult = xmlBuilder.createElement("difficult")
            difficultContent = xmlBuilder.createTextNode("0")
            difficult.appendChild(difficultContent)
            object.appendChild(difficult)
            bndbox = xmlBuilder.createElement("bndbox")
            xmin = xmlBuilder.createElement("xmin")
            mathData = int(((float(oneline[1])) * Pwidth + 1) - (float(oneline[3])) * 0.5 * Pwidth)
            xminContent = xmlBuilder.createTextNode(str(mathData))
            xmin.appendChild(xminContent)
            bndbox.appendChild(xmin)
            ymin = xmlBuilder.createElement("ymin")
            mathData = int(((float(oneline[2])) * Pheight + 1) - (float(oneline[4])) * 0.5 * Pheight)
            yminContent = xmlBuilder.createTextNode(str(mathData))
            ymin.appendChild(yminContent)
            bndbox.appendChild(ymin)
            xmax = xmlBuilder.createElement("xmax")
            mathData = int(((float(oneline[1])) * Pwidth + 1) + (float(oneline[3])) * 0.5 * Pwidth)
            xmaxContent = xmlBuilder.createTextNode(str(mathData))
            xmax.appendChild(xmaxContent)
            bndbox.appendChild(xmax)
            ymax = xmlBuilder.createElement("ymax")
            mathData = int(((float(oneline[2])) * Pheight + 1) + (float(oneline[4])) * 0.5 * Pheight)
            ymaxContent = xmlBuilder.createTextNode(str(mathData))
            ymax.appendChild(ymaxContent)
            bndbox.appendChild(ymax)
            object.appendChild(bndbox)

            annotation.appendChild(object)

        f = open(os.path.join(xmlfolder, name[0:-4] + ".xml"), 'w')
        print(os.path.join(xmlfolder, name[0:-4] + ".xml"))
        xmlBuilder.writexml(f, indent='\t', newl='\n', addindent='\t', encoding='utf-8')
        f.close()