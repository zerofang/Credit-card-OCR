# Credit-card-OCR
Failed case of Credit card OCR with OpenCV and Python
Python 3.6.2 + OpenCV3.3.0
参加某比赛的练手作品，比赛已于10日截止提交，故上传献丑。
本项目在干扰较少的理想情况下有较大可能能够完成识别，总体效果较差，以后有机会可能会优化。

## 运行环境
### (一)	软件和扩展包版本
Python-3.6.2（32bit） 
OpenCV-3.2.0 numpy-1.13.1 
Iminuit-1.2 
 
### (二)	具体import
1.	main.py 

```
import cv2 
import numpy as np 
import argparse 
from transform import four_point_transform 
from ocr_template_match import match 
```

 
2.	ocr_template_match.py 

```
from imutils import contours 
import numpy as np 
import imutils 
import cv2 
```

3.	transform.py 

```
import numpy as np 
import cv2 
```



## 项目概述
### 功能描述
该项目工具用于识别**卡号凸起镀金属的银行卡**卡号前8位，输入银行卡图片，输出卡号及经矫正和标注卡号的银行卡图片。
### 项目结构框架 

待补
### 使用简介
输入命令： 

```
python main.py  –i cards/xx.jpg -r reference.jpg
```

其中”xx.jpg”指欲识别卡片图片名称 

## 算法设计及具体实现 
### (一)	卡片图像预处理 
1.	解析参数，读取图像。 
2.	利用OpenCV的GaussianBlur函数对图像进行模糊处理 
3.	利用Canny函数进行边缘检测 
4.	利用dilate函数进行膨胀处理 
5.	利用HoughLines函数进行直线检测，得到包含图像中所有符合条件直线上的极坐标list 
6.	利用极坐标计算出直角坐标，然后利用自己编写的直线过滤算法过滤多余直线，留下通过卡片四条边的直线。算法具体原理为： 
取所有确定直线的两点中纵坐标之和最小者为银行卡的确定上边界直线的点；
纵坐标之和最大者为银行卡的确定下边界直线的点；
横坐标之和最小者为银行卡的确定左边界直线的点；
横坐标之和最大者为银行卡的确定右边界直线的点。 
7.	利用由确定直线的两点坐标求两直线交点坐标算法计算出卡片四个角的坐标 
### (二)	图像矫正 
1.	将四角坐标list传入transform.py中的four_point_transform函数 
2.	利用自定义函数order_points将list中的点的坐标顺序统一为左上，右上，右下，左下。具体算法原理为： 
左上点坐标的横纵坐标之和最小；右下点坐标的横纵坐标之和最大；右上点横纵坐标之差最小；坐下点横纵坐标之差最大。 
3.	将 list 中的坐标提取出来，并计算图像宽度（取左上-右上点横坐标之差和左下-右下点横坐标之差的较大者），计算图像高度（取左上
-左下点横坐标之差和右上-右下点横坐标之差的较大者） 
4.	由得到的坐标，长宽等值计算出透视矩阵，利用矩阵变换将图像矫正并切割掉卡片背景部分，返回矫正后图像。 
### (三)	卡号识别 
1.	载入参考字符图片，进行灰度，二值化处理，然后分割数符，排序，调整为统一大小 
2.	载入矫正后图像，进行调整大小，灰度化处理，tophat形态学处理，二值化，morphologyEx闭运算等处理，过滤矩形轮廓，得到卡号前八位的两个矩形轮廓 
3.	对两个矩形轮廓内的字符依次进行识别，并将识别结果画到图像中矩形框的上方，记录识别出的结果数字 
4.	显示结果图像和卡号数字串 


## 算法分析及经验总结 
1. 由于本人能力较弱，不熟悉OpenCV的函数及其用法，以及OpenCV一些算法的局限性，在某些极端情况（倾斜角度过大，反光，卡片背景复杂，卡号暗淡）下效果不尽如人意。 
2. 由于样本数据集过小，无法使用深度学习来训练可靠的模型，实际应用中可以利用深度学习提高极端情况下的识别率。 
3. 不同于学术理论，应用于工业实际的项目需要考虑的意外和极端情况非常多，因此想要做出一个能应用于实际的，健壮的，通用的工具，需要多方面的知识和缜密的思维，难度较大，但对于推动行业进步也十分重要。 
