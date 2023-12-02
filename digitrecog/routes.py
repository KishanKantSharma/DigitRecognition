from digitrecog import app
from flask import render_template,flash, request

IMAGE_FOLDER='static/'
app.config['UPLOAD_FOLDER']=IMAGE_FOLDER

import numpy as np
import matplotlib.pyplot as plt
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D,MaxPooling2D,Flatten,SpatialDropout2D,Dropout
import tensorflow.keras.backend as k
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import os
mymodel=load_model('mymodel.h5')

@app.route('/',methods=['GET','POST'])
def home():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])
def predict():
    #if request.method=='POST':
    f=request.files['file']
    f.save(os.path.join(app.config['UPLOAD_FOLDER'], f.filename))
    full_filename = os.path.join(app.config['UPLOAD_FOLDER'],f.filename)
    img=image.load_img(full_filename,target_size=(28,28),color_mode='grayscale')
    img=image.img_to_array(img)
    img=img/255
    for y in range(28):
        for z in range(28):
            if img[y][z][0]==1:
                img[y][z][0]=0
            else:
                img[y][z][0]= 1-img[y][z][0]

    img=np.expand_dims(img,axis=0)
    prediction=str(mymodel.predict_classes(img)[0])
    return render_template('predict.html',img=str(full_filename),prediction=prediction)
