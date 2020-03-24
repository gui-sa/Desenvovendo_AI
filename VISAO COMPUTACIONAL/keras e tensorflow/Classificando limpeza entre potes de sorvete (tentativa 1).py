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

dir_train = os.path.join('/home','salomao','Desktop','Detector_images','Pardal_Treinamento')#Criando o caminho para a pasta à qual contem os dados. 
dir_val = os.path.join('/home','salomao','Desktop','Detector_images','Pardal_Validacao')#Criando o caminho para a pasta de validacao

train_pmacho = os.path.join(dir_train,'pardal macho')#Criando o caminho para a pasta à qual contem os dados.
train_pfemea = os.path.join(dir_train,'pardal femea')#Criando o caminho para a pasta à qual contem os dados.

val_pmacho = os.path.join(dir_val,'pardal macho')#Criando o caminho para a pasta à qual contem os dados.
val_pfemea = os.path.join(dir_val,'pardal femea')#Criando o caminho para a pasta à qual contem os dados.

num_train_pmacho = len(os.listdir(train_pmacho))#Numero de arquivos dentro do diretorio'dir_potes_sujos'
num_train_pfemea= len(os.listdir(train_pfemea))#Numero de arquivos dentro do diretorio'dir_potes_limpos'

num_val_pmacho = len(os.listdir(val_pmacho))#Numero de arquivos dentro do diretorio'dir_potes_sujos'
num_val_pfemea= len(os.listdir(val_pfemea))#Numero de arquivos dentro do diretorio'dir_potes_limpos'

num_tot_val = num_val_pmacho + num_val_pfemea#O total de arquivos que a pasta possui
num_tot_train = num_train_pmacho + num_train_pfemea#O total de arquivos que a pasta possui

num_tot = num_tot_train + num_tot_val

#============================================Configurando variaveis de treinamento ==================================

batch_size =10#Cada epoca usara 128 imagens 
epochs = 10#Quantas passadas é relizada
IMG_HEIGHT = 720#Altura em pixel da imagem

IMG_WIDTH = 720#Comprimento em pixel da imagem


#============================================Preprocessamento dos dados =============================================

train_image_generator = ImageDataGenerator(rescale = 1./255, horizontal_flip=True, rotation_range=45) # reescala os valores em float de 0 - 1


train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,  #Do objeto train_image_generator, criar um fluxo de  matrizes de batch tamanho batch size
                                                           directory=dir_train, #No diretorio: 
                                                           shuffle=True, #devo randomizar?sim!
                                                           target_size=(IMG_HEIGHT, IMG_WIDTH),#Imagens de tamanho...
                                                           class_mode='categorical')


val_image_generator = ImageDataGenerator(rescale = 1./255) # reescala os valores em float de 0 - 1


val_data_gen = val_image_generator.flow_from_directory(batch_size=batch_size,  #Do objeto train_image_generator, criar um fluxo de  matrizes de batch tamanho batch size
                                                           directory=dir_val, #No diretorio: 
                                                           shuffle=True, #devo randomizar?sim!
                                                           target_size=(IMG_HEIGHT, IMG_WIDTH),#Imagens de tamanho...
                                                           class_mode='categorical')


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
    Dropout(0.2),#20% dos neuronios nao serao ativadas
    Flatten(),
    Dense(512, activation='relu'),
    Dense(2)
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

model.save('pardal')


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










