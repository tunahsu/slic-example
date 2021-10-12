from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.measure import regionprops
from skimage.util import img_as_float
from skimage import io
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2
import os


# 設置參數
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True, help="Path to the image set")
ap.add_argument("-o", "--output", required=True, help="Path to the result set")
ap.add_argument("-s", "--segments", required=False, default=100, help="Number of segments")
args = vars(ap.parse_args())

# 取得圖片輸入位置
inputPath = os.path.join(os.path.dirname(__file__), args["input"])
files = os.listdir(inputPath)

# 設置圖片輸出位置
outPath = os.path.join(os.path.dirname(__file__), args["output"])
if not os.path.isdir(outPath):
	os.mkdir(outPath)

for idx1, f in enumerate(files):
	subPath = os.path.join(outPath, '%d' % (idx1+1))
	os.mkdir(subPath)

	# SLIC分割、繪製分割線
	image = io.imread(os.path.join(inputPath, f))
	segments = slic(img_as_float(image), n_segments=int(args["segments"]), compactness=10, sigma=5)
	marked = mark_boundaries(img_as_float(image), segments)

	# 顯示、輸出已分割圖片
	fig = plt.figure("Segments%d" % (idx1))
	ax = fig.add_subplot(1, 1, 1)
	ax.imshow(marked)
	plt.axis("off")
	# plt.show()
	plt.imsave(os.path.join(subPath, 'result.jpg'), marked)
	cv2.imwrite(os.path.join(subPath, 'origin.jpg'), cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

	# 輸出每個單一分割
	for idx2, region in enumerate(regionprops(segments)):
		# 建立分割對應的遮罩
		print("export segment %d: %d" % (idx1+1, idx2+1), end='\r')
		# mask = np.zeros(image.shape[:2], dtype = "uint8")
		# mask[segments == segVal] = 255

		# 顯示遮罩區域
		# cv2.imshow("Mask", mask)
		# cv2.imshow("Applied", cv2.bitwise_and(image, image, mask = mask))
		# cv2.waitKey(0)

		# 儲存遮罩區域
		# output = cv2.bitwise_and(image, image, mask = mask)
		# output[segments != segVal] = 255

		min_row, min_col, max_row, max_col = region.bbox

		output = image[min_row:max_row, min_col:max_col]
		filename = os.path.join(subPath, '%d.jpg' % (idx2+1))
		cv2.imwrite(filename, cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
