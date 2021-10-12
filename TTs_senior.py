import matplotlib.pyplot as plt
import numpy as np
import cv2

from skimage.data import astronaut
from skimage.color import rgb2gray
from skimage.filters import sobel
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float

import matplotlib.image as mpimg
import os


from PIL import Image



def ImgCut(OutPath, ImgName, cut_w, cut_h):
    if not os.path.isdir(OutPath):            
        os.makedirs(OutPath)    
    
    # 取得圖片:長 寬 RGB
    img = cv2.imread(ImgName)
    h, w, band = img.shape 
    print(w, h, band)

    # 裁切區域的 x 與 y 座標（左上角） 初始化
    x_point = 0
    y_point = 0
    
    # 裁切圖片
    i = 1
    for y in range(0, h, cut_h):
        y_point = y
        for x in range(0, w, cut_w):
            x_point = x
            cut_photo = img[y_point: y_point + cut_h, x_point: x_point + cut_w]
            cv2.imwrite(OutPath + str(i) + '.jpg', cut_photo)
            i += 1
    
def ResizeImg(Read_Img, ImgOutPath, scale):
    if not os.path.isdir(ImgOutPath):            
        os.makedirs(ImgOutPath)    
       
    files = os.listdir(Read_Img)
    for f in files:        
        SP_img = cv2.imread(Read_Img + f)
        h, w, band = SP_img.shape
        SP_img = cv2.resize(SP_img, ( int(w*scale), int(h*scale)))
        # 寫入圖檔
        cv2.imwrite(ImgOutPath + f, SP_img)

def Img_Mapping(Read_Img, ImgOutPath): #橫向
    if not os.path.isdir(ImgOutPath):            
        os.makedirs(ImgOutPath)    
        
    #橫向 ----
    files = os.listdir(Read_Img)
    files.sort()
    files.sort(key = lambda x: int(x[:-4]))
    i = 1
    col = 1
    for f in files:  
        image = cv2.imread(Read_Img + f)
        if (i == 1 ):
            oneT = image
        else:
            oneT = np.concatenate((oneT, image), axis = 1)
        if(i == 8):
            cv2.imwrite(ImgOutPath +str(col) +".jpg",oneT)
            col+=1
            i = 1
        else:
            i += 1
    #縱向 |
    files = os.listdir(ImgOutPath)
    files.sort()
    files.sort(key = lambda x: int(x[:-4]))
    i = 1
    col = 1
    for f in files:  
        image = cv2.imread(ImgOutPath + f)
        if (i == 1 ):
            oneT = image
            os.remove(ImgOutPath + f)
        else:
            oneT = np.concatenate((oneT, image))
            os.remove(ImgOutPath + f)
        if(i == 7):
            
            cv2.imwrite(ImgOutPath +str(col) +".jpg",oneT)
            col+=1
            i = 1
        else:
            i += 1

def Compact_watershed(ImgN):    
    Img = cv2.imread(ImgN)
    h, w, band = Img.shape #取得圖片 :長 寬 RGB

    print(w, h, band)
    gradient = sobel(rgb2gray(Img))
    segments_watershed = watershed(gradient, markers=250, compactness=0.001)

    Img = mark_boundaries(Img, segments_watershed)
    
    return Img
    
def SLIC(img):
    segments_slic = slic(img, n_segments=250, compactness=10, sigma=1)
    slic = mark_boundaries(img, segments_slic)
    
    return slic
    
    
outpath = "./split_20190118_0.95cm/"
 
# 裁切區域的長度與寬度
cut_w = 2000
cut_h = 2000
    
ImgCut(outpath, "20190118_0.95cm.jpg", cut_w, cut_h)

ResizeF = "./Resize/"    
scale = 0.03 #縮3%
# ResizeImg(outpath, ResizeF, scale)

# Img_Mapping(ResizeF, "./Out/") #橫向

'''
Compact_watershed = Compact_watershed("./Out/1.jpg")

plt.figure()
plt.axis("off")
plt.imshow( Compact_watershed)
plt.title('Compact watershed')
plt.show()

_SLIC = SLIC("./Out/1.jpg")

plt.figure()
plt.axis("off")
plt.imshow( _SLIC)
plt.title('SLIC')
plt.show()
'''