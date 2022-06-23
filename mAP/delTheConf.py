import numpy as np
import cv2
import torch
import os

label_path = './predict_label'
image_path = './test'


# 坐标转换，原始存储的是YOLOv5格式
# Convert nx4 boxes from [x, y, w, h] normalized to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right

folder = os.path.exists('fakelabel')
if not folder:
    os.makedirs('fakelabel')

folderlist = os.listdir(label_path)
for i in folderlist:
    label_path_new = os.path.join(label_path, i)
    with open(label_path_new, 'r') as f:
        lb = np.array([x.split() for x in f.read().strip().splitlines()], dtype=np.float32)  # labels
        print(lb)
    read_label = label_path_new.replace(".txt", ".jpg")
    read_label_path = read_label.replace('predict_label', 'test')

    # 绘图
    for _, x in enumerate(lb):
        class_label = int(x[0])  # class
        with open('fakelabel/' + i, 'a') as fw:  # 这里需要把confidence放到第二位
            fw.write(str(int(x[0])) + ' ' + str(x[1]) + ' ' + str(x[2]) + ' ' + str(x[3]) + ' ' + str(x[4]) + '\n')
    '''cv2.imshow('show', img)
    cv2.waitKey(0)  # 按键结束
    cv2.destroyAllWindows()'''
