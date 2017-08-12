# import the necessary packages
import numpy as np
import cv2

def order_points(pts):
	# 生成坐标的list，调整顺序为左上，右上，右下，左下
	rect = np.zeros((4, 2), dtype = "float32")

	# 左上点坐标的横纵坐标之和最小
	# 右下点坐标的横纵坐标之和最大
	s = np.sum(pts,axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]

	# 右上点横纵坐标之差最小
	# 坐下点横纵坐标之差最大
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]

	# 返回处理完的坐标list
	return rect

def four_point_transform(image, pts):

	# 统一坐标顺序并将其分开提取出来
	rect = order_points(pts)
	(tl, tr, br, bl) = rect

	# 计算图像宽度，取左上-右上点横坐标之差和左下-右下点横坐标之差的较大者
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))

	# 计算图像高度，取左上-左下点横坐标之差和右上-右下点横坐标之差的较大者
	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))

	# 利用已得长宽和坐标得到图像的鸟瞰图数组
	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")

	# 计算得到透视变换的矩阵并应用
	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

	# 返回矫正完毕的图像
	return warped