# import the necessary packages
import cv2
import numpy as np
import argparse
from transform import four_point_transform
from ocr_template_match import match

# 利用确定一条直线的两点坐标（x0,y0),(x1,y1)和确定另一条直线的两点坐标(x2,y2),(x3,y3)求两直线的交点坐标
def point(x0,y0,x1,y1,x2,y2,x3,y3):
    a = y1-y0
    b = x1*y0-x0*y1
    c = x1-x0
    d = y3-y2
    e = x3*y2-x2*y3
    f = x3-x2
    y = float(a*e-b*d)/(a*f-c*d)
    x = float(y*c-b)/a
    pt = (int(x),int(y))
    return pt

# 参数解析
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
ap.add_argument("-r", "--reference", required=True,
	help="path to reference OCR-A image")
args = vars(ap.parse_args())

# 读取图像
image = cv2.imread(args["image"])

# 高斯模糊处理
image1 = cv2.GaussianBlur(image, (7 , 7), 0)

# 边缘检测
edges = cv2.Canny(image1, 50, 150, apertureSize=3)

# 膨胀处理
closed = cv2.dilate(edges, None, iterations = 2)

# 霍夫直线检测
lines = cv2.HoughLines(edges,1,np.pi/180,100)
lines1 = lines[:,0,:]
outline = [[100000,100000,100000,100000],[0,0,0,0],[100000,100000,100000,100000],[0,0,0,0]]
coords = []

# 直线过滤
for rho,theta in lines1[:]:
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))

    # 取所有确定直线的两点中纵坐标之和最小者为银行卡的确定上边界直线的点
    # 纵坐标之和最大者为银行卡的确定下边界直线的点
    # 横坐标之和最小者为银行卡的确定左边界直线的点
    # 横坐标之和最大者为银行卡的确定右边界直线的点
    if(y1>0 and y2>0 and y1+y2 < outline[0][1] + outline[0][3]):
        outline[0] = [x1, y1, x2, y2]
    if(y1>0 and y2>0 and y1+y2 > outline[1][1] + outline[1][3]):
        outline[1] = [x1, y1, x2, y2]
    if(x1>0 and x2>0 and x1+x2 < outline[2][0] + outline[2][2]):
        outline[2] = [x1, y1, x2, y2]
    if(x1>0 and x2>0 and x1+x2 > outline[3][0] + outline[3][2]):
        outline[3] = [x1, y1, x2, y2]

# 传入上一步循环获得的确定四条边直线的点，获得两两相交的交点
coords.append((point(outline[0][0],outline[0][1],outline[0][2],outline[0][3],outline[2][0],outline[2][1],outline[2][2],outline[2][3])))
coords.append((point(outline[0][0],outline[0][1],outline[0][2],outline[0][3],outline[3][0],outline[3][1],outline[3][2],outline[3][3])))
coords.append((point(outline[1][0],outline[1][1],outline[1][2],outline[1][3],outline[3][0],outline[3][1],outline[3][2],outline[3][3])))
coords.append((point(outline[1][0],outline[1][1],outline[1][2],outline[1][3],outline[2][0],outline[2][1],outline[2][2],outline[2][3])))

# 传入交点，进行图像矫正
warped = four_point_transform(image,coords)

# 显示矫正结果
cv2.imshow("Original", image)
cv2.imshow("Warped", warped)
cv2.waitKey(0)

# 读取参考数字图像
ref = cv2.imread(args["reference"])

# 进行匹配
match(ref,warped)
