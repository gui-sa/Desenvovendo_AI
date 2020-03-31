import tensorflow as tf
import keras
import matplotlib.pyplot as plt
import numpy as np

file_mnist=keras.datasets.mnist.load_data()#load do mnist dataset -O arquivo mnist da pra gente duas tuplas: a primeira tupla com 60000 exemplares tanto as imagens quanto o target e a segundo tupla com 10000 exemplares da mesma forma
file_train,file_val = file_mnist#Captamos as tuplas, dentro destas tuplas estao os dados disposto em arrays numpy
input_train,target_train = file_train#Captamos a matrix numpy que contem as imagens e o target respectivo - o target nao esta one-hot-enconded
input_val,target_val = file_val#Captamos a matrix numpy que contem as imagens e o target respectivo - o target nao esta one-hot-enconded

#Como o input é uma imagem em cinza 0-255, temos que passar todas para normalizaçao 0-1

input_train =input_train/255
input_val =input_val/255
input_train=input_train.astype('float32')#mudando a precisao de 64->32
input_val=input_val.astype('float32')#mudando a precisao de 64->32

#O input da imagem deve ser batch,widthxheight (flattening image):

input_train = input_train.reshape(60000,784)# 28x28 = 784
input_val = input_val.reshape(10000,784)

#Hotencoding the data target into classes. Classes is a away to go!

target_train = keras.utils.np_utils.to_categorical(target_train,10)
target_val = keras.utils.np_utils.to_categorical(target_val,10)


#Parametros de rede:

batch = 13#Tamanho do batch
epochs = 20#Numero de epcoas



#Criando arquitetura:

model = keras.models.Sequential()#Criando objeto que cria um modelo sequencial

model.add(keras.layers.Dense(10, input_dim = 784, activation = 'softmax'))


model.compile(optimizer = 'sgd', loss = 'categorical_crossentropy', metrics=['accuracy'])#Stochastic Gradient Decent, crossentropy

model.summary()#Plotando o resumo do modelo com o numero de parametros

history = model.fit(input_train, target_train, batch_size = batch, epochs = epochs , verbose = 1 , validation_data = (input_val,target_val)) #Treinando o modelo  com esses endereços... Verbose = 0 : silencioso, Verbose = 1 :barra de progresso, Verbose = 2 : Barra de progresso em cada epoca


#Avaliando o modelo:

score = model.evaluate(input_val, target_val, verbose=0)#Avaliando o modelo para o database de validaçao silenciosamenteo score retornando tera dois floats o primeiro é o erro e o segundo a accuracy
print('Test loss', score[0])
print('Test accuracy', score[1])

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

model.save('/home/salomao/Desktop/saved.h5')
