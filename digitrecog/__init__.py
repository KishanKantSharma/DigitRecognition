import flask
from flask import Flask
from digitconfig import Config

app=Flask(__name__)
app.config.from_object  (Config)

import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D,MaxPooling2D,Flatten,SpatialDropout2D,Dropout
import keras.backend as k
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from keras.preprocessing import image
from keras.models import load_model
import os

from digitrecog import routes
