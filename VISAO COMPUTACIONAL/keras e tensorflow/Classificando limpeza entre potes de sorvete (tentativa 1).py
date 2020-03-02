# Autor: Guilherme Salomão Agostini

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model

import os
import numpy as np
import matplotlib.pyplot as plt


# ============================================Configurando o caminho para a pasta de imagens ========================

dir_train = os.path.join('/home','salomao','Desktop','Desenvolvimento de algoritmos de Inteligencia artificial','Visao computacional para fiscalizar o status de limpeza de isoladores de rede de alta tensao','limpeza_potes_de_sorvete','Treinamento')#Criando o caminho para a pasta à qual contem os dados. 
dir_val = os.path.join('/home','salomao','Desktop','Desenvolvimento de algoritmos de Inteligencia artificial','Visao computacional para fiscalizar o status de limpeza de isoladores de rede de alta tensao','limpeza_potes_de_sorvete','Validacao')

train_potes_sujos = os.path.join(dir_train,'potes_sujos')#Criando o caminho para a pasta à qual contem os dados.
train_potes_limpos = os.path.join(dir_train,'potes_limpos')#Criando o caminho para a pasta à qual contem os dados.

val_potes_sujos = os.path.join(dir_val,'potes_sujos')#Criando o caminho para a pasta à qual contem os dados.
val_potes_limpos = os.path.join(dir_val,'potes_limpos')#Criando o caminho para a pasta à qual contem os dados.

num_train_potes_sujos = len(os.listdir(train_potes_sujos))#Numero de arquivos dentro do diretorio'dir_potes_sujos'
num_train_potes_limpos= len(os.listdir(train_potes_limpos))#Numero de arquivos dentro do diretorio'dir_potes_limpos'

num_val_potes_sujos = len(os.listdir(val_potes_sujos))#Numero de arquivos dentro do diretorio'dir_potes_sujos'
num_val_potes_limpos= len(os.listdir(val_potes_limpos))#Numero de arquivos dentro do diretorio'dir_potes_limpos'

num_tot_val = num_val_potes_sujos + num_val_potes_limpos#O total de arquivos que a pasta possui
num_tot_train = num_train_potes_sujos + num_train_potes_limpos#O total de arquivos que a pasta possui

num_tot = num_tot_train + num_tot_val

#============================================Configurando variaveis de treinamento ==================================

batch_size =10#Cada epoca usara 128 imagens 
epochs = 15#Quantas passadas é relizada
IMG_HEIGHT = 500#Altura em pixel da imagem

IMG_WIDTH = 500#Comprimento em pixel da imagem


#============================================Preprocessamento dos dados =============================================

train_image_generator = ImageDataGenerator(rescale = 1./255) # reescala os valores em float de 0 - 1


train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,  #Do objeto train_image_generator, criar um fluxo de  matrizes de batch tamanho batch size
                                                           directory=dir_train, #No diretorio: 
                                                           shuffle=True, #devo randomizar?sim!
                                                           target_size=(IMG_HEIGHT, IMG_WIDTH),#Imagens de tamanho...
                                                           class_mode='binary')


val_image_generator = ImageDataGenerator(rescale = 1./255) # reescala os valores em float de 0 - 1


val_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,  #Do objeto train_image_generator, criar um fluxo de  matrizes de batch tamanho batch size
                                                           directory=dir_val, #No diretorio: 
                                                           shuffle=True, #devo randomizar?sim!
                                                           target_size=(IMG_HEIGHT, IMG_WIDTH),#Imagens de tamanho...
                                                           class_mode='binary')


#============================================ Visualizar imagens =============================================

sample_training_images, _ = next(train_data_gen)#The next function returns a batch from the dataset. The return value of next function is in form of (x_train, y_train) where x_train is training features and y_train, its labels

for img in sample_training_images:
    plt.imshow(img)
    plt.show()

#==========================================Criando o modelo =========================================================

model = Sequential([
    Conv2D(16, 3, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH ,3)),
    MaxPooling2D(),
    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(1, activation='sigmoid')
])


model.compile(optimizer='adam',#ESTA FUNCAO CONFIGURA O MODELO (OBJETO) para um possível treinamento
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()

history = model.fit_generator(#Esta funçao treina sua rede neural.
    train_data_gen,
    steps_per_epoch=num_tot_train,
    epochs=epochs,
    validation_data=val_data_gen,
    validation_steps=num_tot_val 
)


#=========================================== Salvando modelo ===============================================

model.save('Potesujosoulimpos')


#============================================================ Plotando  resultado ============================

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()


#===============================================================================================================










