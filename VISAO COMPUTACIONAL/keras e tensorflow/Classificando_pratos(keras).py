#O objetivo é criar um programa que é treinado em cima de um data base de pratos sujos/pratos limpos
#O usuario escolhe entre treinar a rede e utilizar a rede ja treinada
#OBVIO, que ainda nao vai possuir interface grafica.
#De inicio vou usar o data set ja montado : pardais!!

import tensorflow as tf
import keras
import numpy as np
import sklearn 
import cv2 as cv
import os
import matplotlib.pyplot as plt
#Preparando os dados de treinamento =============================================================================================

train_path = '/home/salomao/Desktop/Detector_images/Pardal_Treinamento'#Dretorio de treinamento
labels = os.listdir(train_path)#As legendas das imagens.
print('\nTraining Data:\n')
print('\nO valor 0 se refere ' + str(labels[0]))
print('\nO valor 1 se refere ' + str(labels[1]) + '\n')

img_label_1 = os.path.join(train_path,str(labels[0]))#Entro no diretorio da primeira lista
ls = os.listdir(img_label_1)#Criando uma lista com o que tem dentro da pasta, para podermos acessar de forma facil

label_1_array = []#Criando uma lista vazia
for i in ls:#Pego o nome de um arquivo da lista ls 
    pat = os.path.join(img_label_1,str(i))#Dou join nele
    img = cv.imread(pat)#leio ele como imagem
    img = cv.resize(img,(500,500))#Dou rezise - padronizo as imagens
    label_1_array.append(img)#Adiciona na lista
label_1_array = np.array(label_1_array)#Deposis que tudo foi adicionado, passo a lista para um array

#Preciso criar um array contendo o label respectivo... ele deve conter o shape do numero de imagens

label_array_from_label_1_array = np.zeros((len(label_1_array)))
label_array_train = label_array_from_label_1_array
train_data = label_1_array

#Juntando os dados dp outro

img_label_2 = os.path.join(train_path,str(labels[1]))#Entro no diretorio da primeira lista
ls = os.listdir(img_label_2)#Criando uma lista com o que tem dentro da pasta, para podermos acessar de forma facil

label_1_array = []#Criando uma lista vazia
for i in ls:#Pego o nome de um arquivo da lista ls 
    pat = os.path.join(img_label_2,str(i))#Dou join nele
    img = cv.imread(pat)#leio ele como imagem
    img = cv.resize(img,(500,500))#Dou rezise - padronizo as imagens
    label_1_array.append(img)#Adiciona na lista
label_1_array = np.array(label_1_array)#Deposis que tudo foi adicionado, passo a lista para um array

#Preciso criar um array contendo o label respectivo... ele deve conter o shape do numero de imagens

label_array_from_label_1_array = np.ones((len(label_1_array)))
    
train_data=np.append(train_data,label_1_array,axis=0)#Juntando tudo das imagens em um unico numpy. Este é o input
label_array_train= np.append(label_array_train,label_array_from_label_1_array,axis=0)#Juntando tudo das imagens em um unico numpy. Este é o target_true



#Preparando os dados de validacao ===============================================================================================


val_path = '/home/salomao/Desktop/Detector_images/Pardal_Validacao'#Dretorio de treinamento
labels = os.listdir(val_path)#As legendas das imagens.
print('\nValidation data:\n')
print('\nO valor 0 se refere ' + str(labels[0]))
print('\nO valor 1 se refere ' + str(labels[1]) + '\n')

img_label_1 = os.path.join(val_path,str(labels[0]))#Entro no diretorio da primeira lista
ls = os.listdir(img_label_1)#Criando uma lista com o que tem dentro da pasta, para podermos acessar de forma facil

label_1_array = []#Criando uma lista vazia
for i in ls:#Pego o nome de um arquivo da lista ls 
    pat = os.path.join(img_label_1,str(i))#Dou join nele
    img = cv.imread(pat)#leio ele como imagem
    img = cv.resize(img,(500,500))#Dou rezise - padronizo as imagens
    label_1_array.append(img)#Adiciona na lista
label_1_array = np.array(label_1_array)#Deposis que tudo foi adicionado, passo a lista para um array

#Preciso criar um array contendo o label respectivo... ele deve conter o shape do numero de imagens

label_array_from_label_1_array = np.zeros((len(label_1_array)))
label_array_val = label_array_from_label_1_array
val_data = label_1_array

#Juntando os dados dp outro

img_label_2 = os.path.join(val_path,str(labels[1]))#Entro no diretorio da primeira lista
ls = os.listdir(img_label_2)#Criando uma lista com o que tem dentro da pasta, para podermos acessar de forma facil

label_1_array = []#Criando uma lista vazia
for i in ls:#Pego o nome de um arquivo da lista ls 
    pat = os.path.join(img_label_2,str(i))#Dou join nele
    img = cv.imread(pat)#leio ele como imagem
    img = cv.resize(img,(500,500))#Dou rezise - padronizo as imagens
    label_1_array.append(img)#Adiciona na lista
label_1_array = np.array(label_1_array)#Deposis que tudo foi adicionado, passo a lista para um array

#Preciso criar um array contendo o label respectivo... ele deve conter o shape do numero de imagens

label_array_from_label_1_array = np.ones((len(label_1_array)))
    
val_data=np.append(val_data,label_1_array,axis=0)#Juntando tudo das imagens em um unico numpy. Este é o input
label_array_val= np.append(label_array_val,label_array_from_label_1_array,axis=0)#Juntando tudo das imagens em um unico numpy. Este é o target_true



print ('\nAVISO: se os valores dos nomes treinamento de validacao nao coincidirem, os dados nao serao validos.\n')

#Misturando data ====================================================================================================

train_data,label_array_train = sklearn.utils.shuffle(train_data,label_array_train, random_state=0)#Os vetores devem estar no formato numpy.

val_data,label_array_val = sklearn.utils.shuffle(val_data,label_array_val, random_state=0)#Os vetores devem estar no formato numpy.


# Parametros de treinamentos =======================================================================================

batch_size = 5 #Este parametro define o paralelismo que a sua rede é treinada... Quanto maior, mais rapido
epochs = 50#Quantos conjuntos de batchs se repetem
batch_p_step = 11#Quantos batchs formam uma epoca

#Adam config=======================================================================================================
keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)

# Modelo ===========================================================================================================

model = keras.models.Sequential()#Criando o objeto sequencial
model.add(keras.layers.Conv2D(16,(3,3),padding='valid',activation= 'relu',input_shape = (500,500,3)))
model.add(keras.layers.MaxPool2D(pool_size=(2,2)))
model.add(keras.layers.Conv2D(32,(3,3),padding='valid',activation= 'relu'))
model.add(keras.layers.MaxPool2D(pool_size=(5,5)))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(100, activation='relu'))
model.add(keras.layers.Dense(1, activation='sigmoid'))

model.summary()

model.compile(optimizer= 'adam',  loss='binary_crossentropy', metrics= ['accuracy'])

history = model.fit(train_data,label_array_train,batch_size=batch_size,epochs=epochs, verbose=1, validation_data=(val_data,label_array_val))

#Avaliando o modelo:

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
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



model.save('/home/salomao/Desktop/modelo_conv_pratos.h5')











