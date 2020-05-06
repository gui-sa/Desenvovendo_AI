#O objetivo é criar um programa que é treinado em cima de um data base de pratos sujos/pratos limpos
#O usuario escolhe entre treinar a rede e utilizar a rede ja treinada
#OBVIO, que ainda nao vai possuir interface grafica.
#De inicio vou usar o data set ja montado : pardais!!


#A ideia da tentiva dois é, paralelizar  alguns filtros, em tese, aumentando a percepçao à alguns elementos.


import tensorflow as tf
import keras
import numpy as np
import cv2 as cv
import sklearn 
import matplotlib.pyplot as plt
from Desktop.Desenvovendo_AI.VISAO_COMPUTACIONAL.libs import extracting_data_from_video

train_data = extracting_data_from_video.Load_data('/home/salomao/Desktop/relogio1_x')
data = extracting_data_from_video.Load_data('/home/salomao/Desktop/relogio2_x')
train_data = np.append(train_data,data, axis = 0)
data = extracting_data_from_video.Load_data('/home/salomao/Desktop/relogio3_x')
train_data = np.append(train_data,data, axis = 0)

label_array_train = extracting_data_from_video.Load_data('/home/salomao/Desktop/relogio1_y')
data = extracting_data_from_video.Load_data('/home/salomao/Desktop/relogio2_y')
label_array_train = np.append(label_array_train,data, axis = 0)
data = extracting_data_from_video.Load_data('/home/salomao/Desktop/relogio3_y')
label_array_train = np.append(label_array_train,data, axis = 0)

train_data,label_array_train = sklearn.utils.shuffle(train_data,label_array_train, random_state=0)#Os vetores devem estar no formato numpy.

ones = np.ones((len(label_array_train),1))

label_array_train = np.append(label_array_train,ones, axis = 1)
train_data = train_data/255


val_data = train_data[0:50]
train_data = train_data[50:len(train_data)]

label_val = label_array_train[0:50]
label_array_train = label_array_train[50:len(label_array_train)]

# Parametros de treinamentos =======================================================================================

batch_size = 5 #Este parametro define o paralelismo que a sua rede é treinada... Quanto maior, mais rapido
epochs = 20#Quantos conjuntos de batchs se repetem
batch_p_step = 11#Quantos batchs formam uma epoca

#Optmizer config=======================================================================================================
keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
keras.optimizers.SGD(learning_rate=0.001, momentum=0.0, nesterov=False)
# Modelo ===========================================================================================================

#Input da rede 
layer_in = keras.layers.Input(shape= (500,500,3))
#Linha de acao disturtiva 1
conv1_l1  = keras.layers.Conv2D(6, kernel_size=2, padding='same',activation= 'relu')(layer_in)
pool1_l1 = keras.layers.AveragePooling2D(pool_size=(2, 2))(conv1_l1)
conv2_l1  = keras.layers.Conv2D(16, kernel_size=2, padding='same', activation= 'relu')(pool1_l1)
pool2_l1 = keras.layers.AveragePooling2D(pool_size=(15, 15))(conv2_l1)
flat_l1 = keras.layers.Flatten()(pool2_l1) 
#Linha de acao disturtiva 2
conv1_l2 = keras.layers.Conv2D(32, kernel_size=3, padding='same', activation= 'relu')(layer_in)
pool1_l2  = keras.layers.MaxPool2D(pool_size=(7,7))(conv1_l2)
conv2_l2 = keras.layers.Conv2D(512, kernel_size=3, padding='same', activation= 'relu')(pool1_l2)
pool2_l2  = keras.layers.MaxPool2D(pool_size=(15,15))(conv2_l2)
flat_l2 = keras.layers.Flatten()(pool2_l2)
#Linha de acao disturtiva 3
conv1_l3  = keras.layers.Conv2D(8, kernel_size=3, padding='same', activation= 'relu')(layer_in)
pool1_l3  = keras.layers.MaxPool2D()(conv1_l3)
conv2_l3  = keras.layers.Conv2D(16, kernel_size=3, padding='same', activation= 'relu')(pool1_l3)
pool2_l3  = keras.layers.MaxPool2D()(conv2_l3)
conv3_l3  = keras.layers.Conv2D(32, kernel_size=3, padding='same', activation= 'relu')(pool2_l3)
pool3_l3  = keras.layers.MaxPool2D()(conv3_l3)
conv4_l3  = keras.layers.Conv2D(64, kernel_size=3, padding='same', activation= 'relu')(pool3_l3)
pool4_l3  = keras.layers.MaxPool2D()(conv4_l3)
conv5_l3  = keras.layers.Conv2D(128, kernel_size=3, padding='same', activation= 'relu')(pool4_l3)
pool5_l3  = keras.layers.MaxPool2D()(conv5_l3)
conv6_l3  = keras.layers.Conv2D(256, kernel_size=3, padding='same', activation= 'relu')(pool5_l3)
pool6_l3  = keras.layers.MaxPool2D()(conv6_l3)
flat_l3 = keras.layers.Flatten()(pool6_l3) 
#Point of connection
concatenate1 = keras.layers.concatenate([flat_l1, flat_l2, flat_l3])
#Rede fully connected1
dense1 = keras.layers.Dense(600,activation='relu')(concatenate1)
dense2 = keras.layers.Dense(200,activation='relu')(dense1)
dense3 = keras.layers.Dense(100,activation='relu')(dense2)
layer_out = keras.layers.Dense(1,activation='sigmoid')(dense3)
dense_box = keras.layers.Dense(4, activation= 'relu')(dense3)
concatenate2 = keras.layers.concatenate([ dense_box, layer_out])


model = keras.models.Model(inputs=layer_in,outputs=concatenate2)   
                         
model.summary()



#custom metrics:===================================================================

#+++++++++++++++++++++++++++++++++++++++++++++++++================================


model.compile(optimizer= 'adam',  loss='mae', metrics= ['accuracy'])

history = model.fit(train_data, label_array_train, batch_size=batch_size,epochs=epochs, verbose=1, validation_data=(val_data,label_val))

#Avaliando o modelo:

plt.plot(history.history['loss'][5:])
plt.plot(history.history['val_loss'][5:])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()



model.save('/home/salomao/Desktop/detector_rel.h5')





