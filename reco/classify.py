import os
import argparse
import imutils
import pickle
import cv2
import numpy as np

from keras.models import Sequential
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from keras.layers.core import Dense
from keras.applications.vgg16 import VGG16

def show(frame):
    try:
        frame = cv2.resize(frame, (224, 224))
        frame = frame.astype('float') / 255.0
        frame = img_to_array(frame)
        frame = np.expand_dims(frame, axis=0)

        # 表情分類
        proba = model.predict(frame)[0]
        idx = np.argmax(proba)
        label = lb.classes_[idx]

        # 框臉並標示分類結果
        print(label, proba)
    except Exception as e:
        print(e)
    return frame

# 定義參數
ap = argparse.ArgumentParser()
ap.add_argument('-m', '--model', required=True, help='path to trained model model')
ap.add_argument('-l', '--labelbin', required=True, help='path to label binarizer')
ap.add_argument('-i', '--image', required=False, help='path to input image')
ap.add_argument('-v', '--video', required=False, help='path to input video')
args = vars(ap.parse_args())

# 載入模型及標籤
lb = pickle.loads(open(args['labelbin'], 'rb').read())
model = Sequential()
for layer in VGG16().layers[:-1]:
    model.add(layer)

for layer in model.layers:
    layer.trainable = False

model.add(Dense(len(lb.classes_), activation='softmax'))  

model.load_weights(args['model'])


if args['image']:
    frame = cv2.imread(args['image'])
    frame = show(frame)

    # 顯示結果並儲存
    # cv2.imshow('Emotion Recognition', frame)
    # cv2.imwrite('result.jpg', frame)
    
    # 釋放資源
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
else:
    if args['video']:
        cap = cv2.VideoCapture(args['video'])
    else:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 500)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 500)

    # 設定輸出影片格式
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('result.avi', fourcc, 20.0, (width, height))

    while (cap.isOpened()):
        ret, frame = cap.read()
        frame = show(frame)

        # 顯示結果
        cv2.imshow('Emotion Recognition', frame)
        out.write(frame)

        if cv2.waitKey(10) == 27:
            break

    # 釋放資源
    out.release()
    cap.release()
    cv2.destroyAllWindows()
