# import the necessary packages
from imutils import contours
import numpy as np
import imutils
import cv2

def match(ref,image):

	# 对参考OCR字体图片进行灰度化和二值化处理，以使数字呈现黑底白字状态
	ref = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
	ref = cv2.threshold(ref, 10, 255, cv2.THRESH_BINARY_INV)[1]

	# 勾勒出数字的外轮廓
	# 按数字按从左到右的顺序生成一个映射字典
	refCnts = cv2.findContours(ref.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
	refCnts = refCnts[0] if imutils.is_cv2() else refCnts[1]
	refCnts = contours.sort_contours(refCnts, method="left-to-right")[0]
	digits = {}

	# 对参考数字的轮廓进行循环处理
	for (i, c) in enumerate(refCnts):
		# 计算并提取出数字的边界包围框大小
		# 并将其统一为固定的大小
		(x, y, w, h) = cv2.boundingRect(c)
		roi = ref[y:y + h, x:x + w]
		roi = cv2.resize(roi, (47, 81))

		# 更新数字字典，并将数字图像对应值映射到ROI
		digits[i] = roi

	# 生成一个矩形和正方形
	# 构造kernel
	rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (12, 4))
	sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

	# 对输入图像进行调整大小，灰度化处理
	image = imutils.resize(image, width=400)

	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	# 对图像应用tophat形态学处理来找到暗背景上的比较亮的部分，即卡号部分
	tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, rectKernel)

	# 计算tophat图像的Scharr gradient，然后将其缩放到0到255范围
	gradX = cv2.Sobel(tophat, ddepth=cv2.CV_32F, dx=1, dy=0,ksize=-1)
	gradX = np.absolute(gradX)
	(minVal, maxVal) = (np.min(gradX), np.max(gradX))
	gradX = (255 * ((gradX - minVal) / (maxVal - minVal)))
	gradX = gradX.astype("uint8")

	# 利用morphologyEx闭运算补全卡号数字之间的空隙，然后用对图像进行二值化处理
	gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)
	thresh = cv2.threshold(gradX, 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

	# 进一步补全卡号数字之间的空隙
	thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sqKernel)

	# 找到二值化后的图像中白色部分的矩形轮廓，生成卡号数字的location list
	cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
	cnts = cnts[0] if imutils.is_cv2() else cnts[1]
	locs = []

	# 对数字轮廓进行循环处理
	for (i, c) in enumerate(cnts):
		# 利用卡号轮廓的外包矩形的四角坐标得到纵横比例
		(x, y, w, h) = cv2.boundingRect(c)
		ar = w / float(h)
		# 由于信用卡上的卡号单组数字的大小基本固定，因此可以利用纵横比例得到卡号位置对轮廓进行裁剪
		if ar > 2.5 and ar < 4:
			# 轮廓可根据最大/最小的宽度/高度进一步裁剪
			if (w > 50 and w < 70) and (h > 10 and h < 25):
				# 将卡号的位置加入location list
				locs.append((x, y, w, h))

	# 按数字从左到右的顺序生成已分类数字的list
	locs = sorted(locs, key=lambda x:x[0])
	output = []

	# 循环处理两组数字
	for (i, (gX, gY, gW, gH)) in enumerate(locs):
		# 生成组内数字的list
		groupOutput = []

		# 从灰度化图像中提取四个数字的组ROI，然后将其从背景中截取出来
		group = gray[gY - 3:gY + gH + 3, gX - 3:gX + gW + 3]
		group = cv2.threshold(group, 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

		# 检测组中的每个数字的轮廓，从左到右排序
		digitCnts = cv2.findContours(group.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
		digitCnts = digitCnts[0] if imutils.is_cv2() else digitCnts[1]
		digitCnts = contours.sort_contours(digitCnts,method="left-to-right")[0]

		# 循环处理数字轮廓
		for c in digitCnts:

			# 计算出单个数字的轮廓的外包矩形，提取出数字，进行缩放处理
			# 使之与参考OCR字体相同
			(x, y, w, h) = cv2.boundingRect(c)
			roi = group[y:y + h, x:x + w]
			roi = cv2.resize(roi, (47, 81))

			# 生成匹配分数的list
			scores = []

			# 循环处理参考数字的对应值和数字ROI
			for (digit, digitROI) in digits.items():
				# 利于基于相关的模板匹配得到并更新得分list
				result = cv2.matchTemplate(roi, digitROI,cv2.TM_CCOEFF)
				(_, score, _, _) = cv2.minMaxLoc(result)
				scores.append(score)

			# 取匹配得分最高的参考数字图像的对应值作为该数字的匹配结果
			groupOutput.append(str(np.argmax(scores)))

		# 在数字组的上方标出数字对应的匹配结果
		cv2.rectangle(image, (gX - 3, gY - 3),(gX + gW + 3, gY + gH + 3), (0, 0, 255), 2)
		cv2.putText(image, "".join(groupOutput), (gX, gY - 15),cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

		# 更新输出数字的list
		output.extend(groupOutput)

	# 显示匹配结果图像，打印匹配得到的数字
	print("Credit Card #: {}".format("".join(output)))
	cv2.imshow("result", image)
	cv2.waitKey(0)