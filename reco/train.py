import os
import argparse
import random

import numpy as np
import pickle
import cv2
import matplotlib
import matplotlib.pyplot as plt

from imutils import paths

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense, Flatten
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator, img_to_array
from keras.layers.normalization import BatchNormalization
from keras.metrics import categorical_crossentropy
from keras.layers.convolutional import *
from keras.applications.vgg16 import VGG16

from network import SmallerVGGNet

matplotlib.use('Agg')



# 定義參數
ap = argparse.ArgumentParser()
ap.add_argument('-d', '--dataset', required=True, help='path to input dataset')
ap.add_argument('-m', '--model', required=True, help='path to output model')
ap.add_argument('-l', '--labelbin', required=True, help='path to output label binarizer')
ap.add_argument('-p', '--plot', type=str, default='plot.png', help='path to output accuracy/loss plot')
args = vars(ap.parse_args())

EPOCHS = 100
LR = 1e-4
BS = 32
IMAGE_DIMS = (224, 224, 3)

data = []
labels = []

# 隨機抓取圖片路徑
print('[INFO] loading images...')
imagePaths = sorted(list(paths.list_images(args['dataset'])))
random.seed(8787)
random.shuffle(imagePaths)


for imagePath in imagePaths:
    # 獲取圖片
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (IMAGE_DIMS[1], IMAGE_DIMS[0]))
    image = img_to_array(image)
    data.append(image)

    # 獲取標籤
    label = imagePath.split(os.path.sep)[-2]
    labels.append(label)

# 圖片正規化
data = np.array(data, dtype='float') / 255.0
labels = np.array(labels)

# # 標籤二值化
lb = LabelBinarizer()
labels = lb.fit_transform(labels)

# 將資料分為 80% 訓練, 20% 測試
trainX, testX, trainY, testY = train_test_split(data, labels, test_size=0.2, random_state=8787)

# 資料增強
aug = ImageDataGenerator(
    rotation_range=30, width_shift_range=0.2,
    height_shift_range=0.2, shear_range=0.2, 
    zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')

# 模型初始化
# print('[INFO] compiling model...')
# model = SmallerVGGNet.build(width=IMAGE_DIMS[1], height=IMAGE_DIMS[0], depth=IMAGE_DIMS[2], classes=len(lb.classes_))
# opt = Adam(lr=LR, decay=LR / EPOCHS)
# model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

# # 開始訓練
# H = model.fit_generator(
#     aug.flow(trainX, trainY, batch_size=BS),
#     validation_data=(testX, testY),
#     steps_per_epoch=len(trainX) // BS,
#     epochs=EPOCHS, verbose=1)

# use VGG16
vgg16_model = VGG16()
model = Sequential()

for layer in vgg16_model.layers[:-1]:
    model.add(layer)

for layer in model.layers:
    layer.trainable = False


model.add(Dense(len(lb.classes_), activation='softmax'))
opt = Adam(lr=LR)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

H = model.fit_generator(
    aug.flow(trainX, trainY, batch_size=BS),
    validation_data=(testX, testY),
    steps_per_epoch=len(trainX) // BS,
    epochs=EPOCHS,
    verbose=2
)


# 保存模型
print('[INFO] serializing network...')
model.save(args['model'])


# 保存標籤
f = open(args['labelbin'], 'wb')
f.write(pickle.dumps(lb))
f.close()

# 繪製 Acc 及 Loss
plt.style.use('ggplot')
plt.figure()
N = EPOCHS
plt.plot(np.arange(0, N), H.history['loss'], label='train_loss')
plt.plot(np.arange(0, N), H.history['val_loss'], label='val_loss')
plt.plot(np.arange(0, N), H.history['accuracy'], label='train_acc')
plt.plot(np.arange(0, N), H.history['val_accuracy'], label='val_acc')
plt.title('Training Loss and Accuracy')
plt.xlabel('Epoch #')
plt.ylabel('Loss/Accuracy')
plt.legend(loc='upper left')
plt.savefig(args['plot'])
