# 基于YOLOv5的手势识别控制程序

## 摘要

随着信息技术的快速发展，计算机广泛进入人们的生活。在商务、游戏、教育、医疗等众多领域，都有不同的人机交互需求。研究更符合人们生活习惯的人机交互技术变得更加有意义。

本系统是人物动作检测系统中的具体应用。目标是使用传统摄像头和计算机实现通过人的手势控制演示文稿放映，主要研究内容分为融合模块注意力机制的YOLOv5模型的调整与搭建、手势识别数据集的建立、交互与控制逻辑的设计实现以及UI交互界面的设计实现。

本系统技术层面是以基于视觉的深度学习方法为核心，使用传统的二维RGB摄像头采集人体手部信息，利用手势检测识别网络模型识别手部位置坐标、大小与类型信息。控制部分利用当前时刻与前一时刻采集到的手势信息综合判断解读语义，然后根据不同的手势语义，使用PyKeyboard发出相应控制指令,例如操作鼠标指针、PPT翻页、图片缩放等功能。

本系统UI交互界面清晰易于控制，核心网络在引入模块注意力机制后，对手部的检测率及分类准确度有较小提升，网络参数少量增加，识别速率小幅降低。手势控制部分的目标功能成功率较好，可以满足使用需求。

> 关键词：手势识别；目标检测；深度学习；YOLO；人机交互；模块注意力

## 运行环境

| 名称                 | 版本      | 名称                      | 版本        |
| -------------------- | --------- | ------------------------- | ----------- |
| absl-py              | 1.0.0     | altgraph                  | 0.17.2      |
| auto-py-to-exe       | 2.19.0    | automium                  | 0.2.6       |
| beautifulsoup4       | 4.11.1    | PyGetWindow               | 0.0.9       |
| bottle               | 0.12.19   | pyinstaller               | 4.10        |
| bottle-websocket     | 0.2.9     | pyinstaller-hooks-contrib | 2022.0      |
| bs4                  | 0.0.1     | PyMsgBox                  | 1.0.9       |
| cachetools           | 4.2.4     | pyparsing                 | 3.0.4       |
| certifi              | 2021.5.30 | pyperclip                 | 1.8.2       |
| cffi                 | 1.15.0    | pypiwin32                 | 223         |
| charset-normalizer   | 2.0.12    | PyQt5                     | 5.15.4      |
| click                | 7.1.2     | pyqt5-plugins             | 5.15.4.2.2  |
| colorama             | 0.4.4     | PyQt5-Qt5                 | 5.15.2      |
| cors                 | 1.0.1     | PyQt5-sip                 | 12.9.1      |
| cycler               | 0.11.0    | pyqt5-tools               | 5.15.4.3.2  |
| Cython               | 0.29.28   | PyRect                    | 0.2.0       |
| dataclasses          | 0.8       | PyScreeze                 | 0.1.28      |
| dlib                 | 19.8.1    | PySide2                   | 5.15.2.1    |
| Eel                  | 0.12.4    | PySocks                   | 1.7.1       |
| filelock             | 3.4.1     | python-dateutil           | 2.8.2       |
| Flask                | 2.0.3     | python-dotenv             | 0.20.0      |
| Flask-Cors           | 3.0.10    | pytweening                | 1.0.4       |
| future               | 0.18.2    | pytz                      | 2021.3      |
| gevent               | 21.12.0   | PyUserInput               | 0.1.10      |
| gevent-websocket     | 0.10.1    | pywin32                   | 303         |
| google-auth          | 2.6.2     | pywin32-ctypes            | 0.2.0       |
| google-auth-oauthlib | 0.4.6     | PyYAML                    | 5.2         |
| greenlet             | 1.1.2     | qt5-applications          | 5.15.2.2.2  |
| grpcio               | 1.45.0    | qt5-tools                 | 5.15.2.1.2  |
| idna                 | 3.3       | requests                  | 2.27.1      |
| importlib-metadata   | 4.8.3     | requests-file             | 1.5.1       |
| imutils              | 0.5.4     | requests-oauthlib         | 1.3.1       |
| itsdangerous         | 2.0.1     | rsa                       | 4.8         |
| Jinja2               | 3.0.3     | scipy                     | 1.5.2       |
| kiwisolver           | 1.3.1     | seaborn                   | 0.7.1       |
| legacy               | 0.1.6     | setuptools                | 58.0.4      |
| Markdown             | 3.3.6     | shiboken2                 | 5.15.2.1    |
| MarkupSafe           | 2.0.1     | six                       | 1.16.0      |
| matplotlib           | 3.3.4     | soupsieve                 | 2.3.2.post1 |
| mkl-fft              | 1.3.0     | tensorboard               | 2.8.0       |
| mkl-random           | 1.1.1     | tensorboard-data-server   | 0.6.1       |
| mkl-service          | 2.3.0     | tensorboard-plugin-wit    | 1.8.1       |
| MouseInfo            | 0.1.3     | tldextract                | 3.1.2       |
| numpy                | 1.19.2    | torch                     | 1.10.2      |
| oauthlib             | 3.2.0     | torchaudio                | 0.10.2      |
| olefile              | 0.46      | torchvision               | 0.11.3      |
| onnx                 | 1.11.0    | tornado                   | 6.1         |
| pandas               | 0.20.3    | tqdm                      | 4.63.0      |
| pefile               | 2021.9.3  | typing_extensions         | 4.1.1       |
| Pillow               | 8.3.1     | urllib3                   | 1.26.9      |
| pip                  | 21.2.2    | utils                     | 1.0.1       |
| place                | 0.5.5     | Werkzeug                  | 2.0.3       |
| protobuf             | 3.19.4    | wheel                     | 0.37.1      |
| pyasn1               | 0.4.8     | whichcraft                | 0.6.1       |
| pyasn1-modules       | 0.2.8     | wincertstore              | 0.2         |
| PyAutoGUI            | 0.9.53    | zipp                      | 3.6.0       |
| pycocotools-windows  | 2.0       | zope.event                | 4.5.0       |
| pycparser            | 2.21      | zope.interface            | 5.4.0       |

## 关于自采数据集

> 本自采数据集还有过多肖像等私人信息，不便于分享，十分抱歉

## 网络整体结构

![YOLOv5+CBAM](https://raw.githubusercontent.com/FeOAr/Gesture-recognition-and-control/main/ImgforReadme/backbone%E7%BD%91%E7%BB%9C.drawio.png)

> 备注：common.py中添加了三种模块注意力机制，可以根据需求在yolov5m_CBAM.yaml中修改。

## 一些补充

### mAP文件夹

> 该文件夹内容源自CSDN，暂时忘了出处

### 修改网络成功后应该出现的模型信息

> ./yolov5-master/runs/detect/model.pdf

### 已训练模型

> ./yolov5-master/UI/weights

### B站视频

[B站视频](https://www.bilibili.com/video/BV1c44y1u7ex?spm_id_from=333.999.0.0)

### 其他

注意在pycharm中配置模型路径