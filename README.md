# YOLO-MIF: Improved YOLOv8 with Multi-Information Fusion for Object Detection in Gray-Scale Images


## 简介
本文针对灰度图像中目标检测的挑战，提出了一种增强型目标检测网络YOLO-MIF，该网络整合了多种多信息融合策略，以改进YOLOv8网络。文章首先介绍了一种技术，用于创建伪多通道灰度图像，增加网络的通道信息，并减轻潜在的图像噪声和虚焦模糊问题。随后，采用网络结构重新参数化技术，提升网络的检测性能而不增加推断时间。另外，引入了一种新颖的解耦式检测头，增强了模型在处理灰度图像时的表现力。文章还对该算法在两个开源灰度图像检测数据集（NEU-DET和FLIR-ADAS）上进行了评估。结果表明，在相同速度下，该算法在平衡检测效率和有效性方面优于YOLOv8 2.1％，优于Faster R-CNN 4.8％，取得了更好的性能表现。
![YOLO-MIF原理图如下：](PaperImages/YOLO-MIF.png)


## 论文贡献
1. YOLO-MIF ：针对灰度图像设计的目标检测网络
2. 新的重参数化模块：WDBB, RepC2f
3. Rep3C Head
4. GIS：针对灰度图像的输入策略

## 论文链接(待更新)
- [YOLO-MIF: Improved YOLOv8 with Multi-Information Fusion for Object Detection in Gray-Scale Images]

[//]: # (- &#40;https://www.sciencedirect.com/science/article/pii/S0957417423008576&#41;)

## 安装
<details open>
<summary>Install</summary>

Pip install the ultralytics package including all [requirements](https://github.com/ultralytics/ultralytics/blob/main/requirements.txt) in a [**Python>=3.7**](https://www.python.org/) environment with [**PyTorch>=1.7**](https://pytorch.org/get-started/locally/).

```bash
pip install -r requirements.txt
```

</details>


<details open>
<summary>Usage</summary>

1. NEU-DET 
```bash
python train_NEU-DET-RepDC.py 

```

2. FLIR-ADAS
```bash
python train_FLIR_ADAS-16-RepDCHead.py

```

</details>

<details open>
<summary> 论文与代码对应地方 </summary>

1. RIR=True   +   SimOTMBBS =  GIS
- SimOTM 效果更好，但是会降低速度，本文采用的SimOTMBBS几乎不会降低速度，后续读者有需要的话SimOTM将会单独开源在arXiv，不再投递期刊，原论文和细节见：https://www.researchgate.net/publication/372944004_Otm-Fusion_An_Image_Preprocessing_Method_for_Object_Detection_in_Grayscale_Image
- Function.cpp 为CUDA代码和C++（CPU）实现
- ultralytics/yolo/data/base.py  （代码位于此文件）
- train_NEU-DET-RepDC.py  （调用代码位于此文件）
```python
parser.add_argument('--use_rir', action='store_true', default=False, help='RIR: random_interpolation_resize ')
parser.add_argument('--use_simotm', type=str, choices=['Gray2BGR', 'SimOTM', 'SimOTMBBS','Gray'], default='SimOTMBBS', help='simotm')
```
- GIS 
![GIS简图如下：](PaperImages/GIS.png)

2. 重参数模块 

- ultralytics/nn/modules/rep_block.py
```python
['DiverseBranchBlock','DeepACBlockDBB','WideDiverseBranchBlock','DeepDiverseBranchBlock','ACBlockDBB','ACBlock']
# WideDiverseBranchBlock 对应论文中 WideDiverseBranchBlock(WDBB),其余模块待做实验验证，需要自取
```
- WDBB 
![WDBB简图如下：](PaperImages/WDBB.png)
- DeepDBB(试验和原理并未在文中说明)
![DeepDBB简图如下：](PaperImages/DeepDBB.png)


- ultralytics/nn/modules/block.py
```python
'C2f_ACDBB', 'C2f_DeepACDBB', 'C2f_DeepDBB', 'C2f_DeepACDBBMix', 'C2f_DBB', 'C2f_ACNET', 'C2f_WDBB'

# 代码中的 C2f_WDBB 对应论文中的  RepC2f ，C2f_DeepDBB及其细节说明将用于下一篇论文，如有使用，请引用github链接或者本论文，其余模块需要自取
```


- ultralytics/nn/modules/head.py
```python
'Detect', 'Segment', 'Pose', 'Classify', 'RTDETRDecoder','DetectDBB','DetectACDBB','DetectAC','DetectDeepDBB',\
          'DetectDeepACDBB' , 'Detect_Efficient','DetectSingleDBB','Detect2AC2DBB',\
          'Detect2DBB2AC','Detect2DBBAC','Detect2ACDBB','Detect_Efficient3DBB','Detect_Efficient3DBBR'

# 代码中的 Detect_Efficient3DBB 对应论文中的Rep3C Head ，部分模块已验证有效果，但是并未加入论文中，其余模块待做实验验证，需要自取
```
- Rep3CHead 
![Rep3CHead简图如下：](PaperImages/Rep3CHead.png)

</details>
  


## 中文解读链接
- [YOLO-MIF中文解读](中文解读链接) [TODO: 如有需要，会在后面编写并更新]

## 视频教程链接
- [YOLO-MIF 视频解读和二次创新方案]() [TODO: 文字版PPT详细教程]

## 二次创新点梳理和代码实现（TODO）
- [二次创新方案]() [PPT教程的最后一页提供了部分二次创新方案，TODO: 如有需要，会在后面编写并更新代码]


## 引用格式
- 待更新论文

[//]: # (如果项目和文章对您有所帮助，请引用以下论文：)

[//]: # (@article{WAN2023120355,)

[//]: # (title = {Random Interpolation Resize: A free image data augmentation method for object detection in industry},)

[//]: # (journal = {Expert Systems with Applications},)

[//]: # (volume = {228},)

[//]: # (pages = {120355},)

[//]: # (year = {2023},)

[//]: # (issn = {0957-4174},)

[//]: # (doi = {https://doi.org/10.1016/j.eswa.2023.120355},)

[//]: # (url = {https://www.sciencedirect.com/science/article/pii/S0957417423008576},)

[//]: # (author = {Dahang Wan and Rongsheng Lu and Ting Xu and Siyuan Shen and Xianli Lang and Zhijie Ren},)

[//]: # (keywords = {Data augmentation, Image interpolation, Object detection, Industry applications, Deep learning algorithm, Convolutional neural network},)

[//]: # (})

[//]: # ()
[//]: # (例如：)

[//]: # ()
[//]: # (Dahang Wan, Rongsheng Lu, Ting Xu, Siyuan Shen, Xianli Lang, Zhijie Ren,)

[//]: # (Random Interpolation Resize: A free image data augmentation method for object detection in industry,)

[//]: # (Expert Systems with Applications,)

[//]: # (Volume 228,)

[//]: # (2023,)

[//]: # (120355,)

[//]: # (ISSN 0957-4174,)

[//]: # (https://doi.org/10.1016/j.eswa.2023.120355.)
## 参考链接
- [整体框架使用代码：YOLOv8](https://github.com/ultralytics/ultralytics)
- [重参数化参考丁霄汉代码：DiverseBranchBlock](https://github.com/DingXiaoH/DiverseBranchBlock)
- [部分模块参考魔鬼面具 开源主页代码](https://github.com/z1069614715/objectdetection_script)
- [YOLOv7](https://github.com/WongKinYiu/yolov7)
- [Albumentations 数据增强库](https://github.com/albumentations-team/albumentations)
- 重参数化验证部分代码参考 手写AI 的重参数化课程
## 结尾
感谢您对本项目的关注和支持。作者尽力提供最好的质量和服务，但仍然有很多需要改进之处。如果您发现任何问题或有任何建议，请告诉我。
另外，本项目目前由我个人维护，难免存在疏漏和错误。如果您发现了任何问题，欢迎提出建议和意见。

## 其他开源项目
其余开源项目陆续在整理发布，后续请查看作者主页进行下载
[主页](https://github.com/wandahangFY)

## 相关问题解答
1. README.md 文件添加 （已完成）  
2. 详细教程 （TODO）
3. 项目环境配置（整个项目是YOLOv8  2023-11-29当日版本，配置参考README-YOLOv8.md文件和requirements.txt）
4. 文件夹对应说明（与YOLOv8保持一致，未改变超参数）（TODO：详细说明 ）
5. 二次创新点梳理和代码实现（TODO）
6. 论文作图：
   - 原理图，网络结构图，流程图：PPT （根据个人选择，也可以使用Visio，亿图，AI等）
   - 实验对比：Orgin（matlab,python,R,Excel都可以）