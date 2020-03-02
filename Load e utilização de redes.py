#Autor: Guilherme Salomão Agostini         


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model

import os
import numpy as np
import matplotlib.pyplot as plt
import cv2


model = load_model('Potesujosoulimpos')#Load da rede neural JA treinada, neste endereço

IMG_HEIGHT = 500
IMG_WIDTH = 500

img = cv2.imread('teste.jpeg')#Qual a imagem ao qual deseja  passar pela rede?
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)#A rede neural foi treinada com imagens em RGB
img = cv2.resize(img,(500,500))#A rede neural foi treinada utilizando 500x500x3

plt.imshow(img)
plt.show()

img = img/255.0#A rede neural foi treinada em um intervalo de 0-1
img = np.expand_dims(img, axis=0) #Este passo coloca mais uma dimensão -> necessário para a equivalencia (pois no keras existe mais uma dmensao ligada ao batch size)
pred = model.predict(img) #Esta diretiva utiliza o modelo para feedforward
print(pred) #Qual o resultado (binario)