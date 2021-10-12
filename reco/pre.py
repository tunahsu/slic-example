import os
import cv2
import csv
import argparse


# 定義命令列參數
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True, help="Path to the input  set")
ap.add_argument("-o", "--output", required=True, help="Path to the output set")
args = vars(ap.parse_args())

# 設置資料集位置
base_dir = os.getcwd()
dataset_dir = os.path.join(base_dir, args['output'])
if not os.path.isdir(dataset_dir):
    os.mkdir(dataset_dir)

folders = os.listdir(os.path.join(base_dir, args["input"]))

for folder in folders:
    # 設置輸出位置
    result_dir = os.path.join(base_dir, args["output"], folder)
    if not os.path.isdir(result_dir):
        os.mkdir(result_dir)

    # 讀取圖檔
    img_path = os.path.join(base_dir, args['input'], folder)
    files = os.listdir(img_path)
    count = 1
    
    for file in files:
        img = cv2.imread(os.path.join(img_path, file))

        # 圖案前處理
        img_resize = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)

        # 輸出撲克牌圖案
        filename = os.path.join(result_dir, '%d' % (count) + '.jpg')
        cv2.imwrite(filename, img_resize)
        count += 1

