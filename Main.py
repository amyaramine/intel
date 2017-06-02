# -*- coding : utf-8 -*-
import pandas as pd
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from keras.utils import np_utils
from numpy.random import permutation
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, Convolution3D, MaxPooling3D
from sklearn.utils import shuffle

# 2448 X 3268
img_rows = 224
img_cols = 224
color_type = 1
nb_conv = 3
nb_pool = 2
nb_classes = 3
batch_size = 10
nb_epoch = 5
split = 0.3
mean_pixel = [103.939, 116.779, 123.68]


# img = cv2.imread( '../IntelMobileODTCervicalCancerScreening/train/Type_1/0.jpg')
# resized = cv2.resize(img, (img_rows, img_cols))
#
#
# plt.imshow(img)
# plt.savefig('../Intel & MobileODT Cervical Cancer Screening/TailleOriginal')
# plt.show()
#
# plt.imshow(resized)
# plt.savefig('../Intel & MobileODT Cervical Cancer Screening/Taillemodif')
# plt.show()
#
# img = cv2.imread( '../Intel & MobileODT Cervical Cancer Screening/train/Type_1/0.jpg', 0)
# resized = cv2.resize(img, (img_rows, img_cols))
#
# plt.imshow(img)
# plt.savefig('../Intel & MobileODT Cervical Cancer Screening/TailleOriginalNG')
# plt.show()
#
# plt.imshow(resized)
# plt.savefig('../Intel & MobileODT Cervical Cancer Screening/TaillemodifNG')
# plt.show()
# exit()

#Load images
def get_im(path, img_rows=224, img_cols=224, color_type=1):
    if color_type == 1:
        img = cv2.imread(path, 0)
    elif color_type == 3:
        img = cv2.imread(path)


    #Reduce size
    resized = cv2.resize(img, (img_rows, img_cols))

    return  resized


def save_model(model, index, cross=''):
    json_string = model.to_json()
    if not os.path.isdir('cache'):
        os.mkdir('cache')
    json_name = 'architecture' + str(index) + cross + '.json'
    weight_name = 'model_weights' + str(index) + cross + '.h5'
    open(os.path.join('cache', json_name), 'w').write(json_string)
    model.save_weights(os.path.join('cache', weight_name), overwrite=True)


path = '../IntelMobileODTCervicalCancerScreening/train/'
typeCervical = os.listdir(path)
print typeCervical

X_train = []
Y_train = []
i = 0
for typeNumber in typeCervical:
    # j = 0
    print typeNumber
    imagesC = os.listdir(path+typeNumber)

    for img in imagesC:
        image = get_im(path+typeNumber+'/'+img,img_rows, img_cols, color_type)
        X_train.append(image)
        Y_train.append(i)
        # j += 1
        # if j == 10:
        #     break

    i += 1

print "Data Loaded"

X_train = np.array(X_train, dtype=np.uint8)
if color_type == 1:
    X_train = X_train.reshape(X_train.shape[0], color_type,
                                    img_rows, img_cols)
else:
    X_train = X_train.transpose((0, 3, 1, 2))

# for c in range(3):
#     X_train[:, c, :, :] = X_train[:, c, :, :] - mean_pixel[c]

# X_train = X_train.reshape(X_train.shape[0], 1 ,color_type,
#                                      img_rows, img_cols)

X_train = X_train.astype('float32')
X_train = X_train / 255



Y_train = np.array(Y_train, dtype=np.uint8)
Y_train = np_utils.to_categorical(Y_train, 3)

perm = permutation(len(X_train))
X_train = X_train[perm]
Y_train = Y_train[perm]

print('Train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print len(Y_train)

def CNN_Model2D(img_rows, img_cols, color_type=1):
  model = Sequential()
  model.add(Convolution2D(16, nb_conv, nb_conv,
                        border_mode='same',
                        input_shape=(color_type, img_rows, img_cols)))
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
  model.add(Dropout(0.25))

  model.add(Convolution2D(32, nb_conv, nb_conv))
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
  model.add(Dropout(0.25))

  model.add(Convolution2D(64, nb_conv, nb_conv))
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
  model.add(Dropout(0.25))

  model.add(Convolution2D(128, nb_conv, nb_conv))
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
  model.add(Dropout(0.25))

  model.add(Flatten())
  model.add(Dense(128))
  model.add(Activation('relu'))
  model.add(Dropout(0.5))
  model.add(Dense(nb_classes))
  model.add(Activation('sigmoid'))

  model.summary()
  model.compile(loss='categorical_crossentropy',
              optimizer='adamax',
              metrics=['accuracy'])
  print "model compiled"
  return model

def CNN_Model4Couches(img_rows, img_cols, color_type=1):
  model = Sequential()
  model.add(Convolution3D(16, nb_conv, nb_conv, nb_conv,
                        border_mode='same',
                        input_shape=(color_type, 1, img_rows, img_cols)))
                         # input_shape = input_shape))
  model.add(Activation('relu'))
  # model.add(MaxPooling3D(pool_size=(nb_pool, nb_pool, nb_pool)))
  model.add(Dropout(0.25))

  model.add(Convolution3D(32, nb_conv, nb_conv, nb_conv, border_mode='same'))
  model.add(Activation('relu'))
  # model.add(MaxPooling3D(pool_size=(1, nb_pool, nb_pool)))
  model.add(Dropout(0.25))

  # model.add(Convolution3D(64, nb_conv, nb_conv, nb_conv, border_mode='same'))
  # model.add(Activation('relu'))
  # model.add(MaxPooling3D(pool_size=(1, nb_pool, nb_pool)))
  # model.add(Dropout(0.25))
  #
  # model.add(Convolution3D(128, nb_conv, nb_conv, nb_conv, border_mode='same'))
  # model.add(Activation('relu'))
  # model.add(MaxPooling3D(pool_size=(1, nb_pool, nb_pool)))
  # model.add(Dropout(0.25))

  model.add(Flatten())
  model.add(Dense(128))
  model.add(Activation('relu'))
  model.add(Dropout(0.5))
  model.add(Dense(nb_classes))
  model.add(Activation('softmax'))

  model.summary()
  model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])
  print "model compiled"
  return model

model  = CNN_Model2D(img_rows, img_cols, 1)
model.fit(X_train, Y_train, batch_size=batch_size,
              nb_epoch=nb_epoch,
              verbose=1,
              validation_split=split, shuffle=True)

score = model.evaluate(X_train, Y_train, verbose=0)
print('Test score for slices:', score[0])
print('Test accuracy for slices:', score[1])
save_model(model, '', str(1))

del X_train
del Y_train

print "Start test"
path = '../IntelMobileODTCervicalCancerScreening/test/'
X_test = []
liste = os.listdir(path)
liste = sorted(liste, key=len)
for element in liste:
    img = get_im(path+element, img_rows, img_cols, color_type)
    X_test.append(img)

X_test = np.array(X_test, dtype=np.uint8)

if color_type == 1:
    X_test = X_test.reshape(X_test.shape[0], color_type,
                                    img_rows, img_cols)
else:
    X_test = X_test.transpose((0, 3, 1, 2))

# for c in range(3):
#     X_test[:, c, :, :] = X_test[:, c, :, :] - mean_pixel[c]
X_test = X_test / 255
# X_test = X_test.reshape(X_test.shape[0],3, color_type,
#                                     img_rows, img_cols)
X_test = X_test.astype('float32')
prob_result = model.predict(X_test)

subbmision = open("submission.csv", "w")
subbmision.write('image_name,Type_1,Type_2,Type_3')

i = 0
for element in prob_result:
    subbmision.write('\n'+str(i)+','+str(round(element[0],2))+','+str(round(element[1],2))+','+str(round(element[2],2)))
    i += 1

subbmision.close()
