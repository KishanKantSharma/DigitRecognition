# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 22:20:19 2020

@author: Karan
"""


import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D,MaxPooling2D,Flatten,SpatialDropout2D,Dropout
import keras.backend as k
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from keras.preprocessing import image

(X_train,y_train),(X_test,y_test)=keras.datasets.mnist.load_data()

image_size=(28,28)
if k.image_data_format()=='channels_last':
  X_train=X_train.reshape(X_train.shape[0],image_size[0],image_size[1],1) #Number of channels is 1 as the images are black and white
  X_test=X_test.reshape(X_test.shape[0],image_size[0],image_size[1],1)
  input_shape=(image_size[0],image_size[1],1)
else:
  X_train=X_train.reshape(X_train.shape[0],1,image_size[0],image_size[1]) #Number of channels is 1 as the images are black and white
  X_test=X_test.reshape(X_test.shape[0],1,image_size[0],image_size[1])
  input_shape=(1,image_size[0],image_size[1])

print('Training set shape: ',X_train.shape,'\nTest set shape: ',X_test.shape,'\nInput Shape: ',input_shape)

#Dividing all pixel values by 255 to scale them to the range [0,1]
X_train=X_train.astype(float)

X_test=X_test.astype(float)

X_train/=255
X_test/=255

#Converting target variable (y) to cateorical
y_train=to_categorical(y_train,num_classes=10)
y_test=to_categorical(y_test,num_classes=10)

#Building the CNN
model=Sequential()
model.add(Conv2D(64,(3,3),padding='same',activation='relu',input_shape=input_shape))
model.add(MaxPooling2D(2,2))
model.add(Conv2D(32,(3,3),padding='same',activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(Conv2D(32,(3,3),padding='valid',activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(Flatten())
model.add(Dropout(0.3))
model.add(Dense(100,activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(10,activation='softmax'))

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

model_save=model.fit(X_train,y_train,batch_size=16,epochs=32,validation_data=(X_test,y_test))

model.save('mymodel.h5',model_save)



#Test
from keras.models import load_model
mymodel=load_model('mymodel.h5')
img=image.load_img('52.png',target_size=(28,28),color_mode='grayscale')

print(img)

img=image.img_to_array(img)
img=img/255

print(img.shape)

for y in range(28):
  for z in range(28):
    if img[y][z][0]==1:
      img[y][z][0]=0
    else:
      img[y][z][0]= 1-img[y][z][0]

img=np.expand_dims(img,axis=0)


mymodel.predict_classes(img)



