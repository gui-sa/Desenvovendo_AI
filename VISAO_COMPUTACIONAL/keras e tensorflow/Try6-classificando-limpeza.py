# coding: iso-8859-1 -*-
#Autor: Guilherme Salomão Agostini 
#Email: guime.sa9@gmail.com
#Referecias importantes:

#O objetivo � tentar classificar a limpeza com o modelo Xception ja implementado, em um dataset preprocessado por mascaras.

#%% Prepare paths of input images and target segmentation masks
import os
import cv2 as cv
import numpy as np
import tqdm
from tensorflow import keras
import tensorflow as tf
import numpy as np
from Desktop.Desenvovendo_AI.VISAO_COMPUTACIONAL.libs import extracting_data_from_video #Lib para mexer com videos 
import sklearn

#%%
# jobs = 2 # it means number of cores

# config = tf.ConfigProto(intra_op_parallelism_threads=jobs,
#                           inter_op_parallelism_threads=jobs,
#                           allow_soft_placement=True,
#                           device_count={'CPU': jobs})
# session = tf.Session(config=config)

#%% ==============================================Criando o dataset

img_size = (1000, 1000)

path_img = "/home/salomao/Desktop/insulators-dataset/jpg"
path_ann = "/home/salomao/Desktop/insulators-dataset/tiff"

list_img = os.listdir(path_img)
list_img.sort()
list_img = sorted(list_img,key=len)
list_ann = os.listdir(path_ann)
list_ann.sort()
list_ann = sorted(list_ann,key=len)
ask = "q"
images =  []
for img_idx,ann_idx in zip(list_img,list_ann):
    path_img_idx = os.path.join(path_img, img_idx)
    path_ann_idx = os.path.join(path_ann, ann_idx)
    img  =  cv.imread(path_img_idx)
    ann =   cv.imread(path_ann_idx, cv.IMREAD_GRAYSCALE)
    _, ann = cv.threshold(ann,1,1,cv.THRESH_BINARY)
    img = cv.resize(img,img_size,interpolation = cv.INTER_AREA)/255.0
    ann = cv.resize(ann,img_size,interpolation = cv.INTER_AREA)  
    
    ann = np.expand_dims(ann, axis=-1)
    img = np.multiply(img,ann) 
    images.append(img)
    if ask==113:
        continue    
    cv.imshow("preview",img)
    ask = cv.waitKey()
    cv.destroyAllWindows()

    
images = np.array(images)
train_0 = images[0:63]
train_1 = images[64:]
images = None

#%%============== Splitando o dataset:

val_1 = train_1[int(0.8*len(train_1)):len(train_1)]#20% do dataset do train_1 vai para validacao
val_0 = train_0[int(0.8*len(train_0)):len(train_0)]#20% do dataset do train_1 vai para validacao
train_1 = train_1[0:int(0.8*len(train_1))]#80% do dataset do train_1 vai para treinamento
train_0 = train_0[0:int(0.8*len(train_0))]#80% do dataset do train_1 vai para treinamento



#%%============  Usando o dataset
train_data,label_array_train  = extracting_data_from_video.shufle_balance(train_1,train_0)#Trecho relacionado ao balanceamento do dataset
val_data,label_val = extracting_data_from_video.shufle_balance(val_1,val_0)

train_1  =None
train_0  =None
val_1 = None
val_0 = None

train_data,label_array_train = sklearn.utils.shuffle(train_data,label_array_train )#misturo o data de label 1
train_data,label_array_train = sklearn.utils.shuffle(train_data,label_array_train )#misturo o data de label 1

#%% Parametros de treinamentos =======================================================================================
    
batch_size = 5 #Este parametro define o paralelismo que a sua rede é treinada... Quanto maior, mais rapido
epochs = 100


#%% Data augmentation

datagen_args = dict(    
    rotation_range=180,
    width_shift_range=0.6,
    height_shift_range=0.6,
    zoom_range=0.5,
    horizontal_flip=True,
    vertical_flip=True
    )

img_datagen = keras.preprocessing.image.ImageDataGenerator(**datagen_args)

batch_size = 5

img_generator = img_datagen.flow(train_data ,label_array_train,  batch_size=batch_size)

#%% model
from tensorflow import keras
from tensorflow.keras.applications.nasnet import NASNetMobile,NASNetLarge

import tensorflow as tf


# model  = keras.applications.NASNetMobile(
#     input_shape=img_size + (3,),
#     include_top=True,
#     weights=None, #randomize de weights
#     input_tensor=None,
#     pooling=None,
#     classes=1,
# )


model = NASNetLarge(
    input_shape=img_size + (3,),
    include_top=None,
    weights=None, #randomize de weights
    input_tensor=None,
    pooling=None,
    classes=1,
)



model.summary()


#%% compilando

import tensorflow as tf

model.compile(optimizer= 'adam',  loss='binary_crossentropy', metrics= ['accuracy'])


model_checkpoint_callback = [
    tf.keras.callbacks.ModelCheckpoint(
    filepath="/home/salomao/Desktop/CNN_try6_cleanlisses.h5",
    save_weights_only=False,
    monitor= "val_accuracy",
    mode='max',
    save_best_only=True)]


#%% ================    Treinamento

#history = model.fit(train_data, label_array_train, validation_data = (val_data,label_val), batch_size=batch_size, epochs=epochs, verbose=1)
history = model.fit(img_generator, steps_per_epoch = int(len(train_data)/batch_size), validation_data = (val_data,label_val), epochs=epochs, verbose=1, callbacks=model_checkpoint_callback )


#%%====================================grafico de treinamento
import matplotlib.pyplot as plt
plt.figure()
plt.plot( np.add(list(range(epochs)),1), history.history["val_accuracy"], lw = 2 , color ="b", label = "Validacão")
plt.plot( np.add(list(range(epochs)),1), history.history["accuracy"], lw = 2 , color ="k", label = "Treinamento")
plt.legend()
plt.title("Aprendizado da rede conforme o tempo - acuracia")
plt.xlabel("Epocas")
plt.ylabel("Taxa de acuracia")
plt.grid(True)
plt.show()
plt.figure()
plt.plot(np.add(list(range(epochs)),1), history.history["loss"], lw = 2 , color ="r", label = "Treinamento")
plt.plot(np.add(list(range(epochs)),1), history.history["val_loss"], lw = 2 , color ="y", label = "Validacao")
plt.legend()
plt.title("Aprendizado da rede conforme o tempo - erro")
plt.xlabel("Epocas")
plt.ylabel("Erro da funcao crossentropy")
plt.grid(True)
plt.show()




