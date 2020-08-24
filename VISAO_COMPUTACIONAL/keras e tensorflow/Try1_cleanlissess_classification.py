#



import tensorflow as tf
import keras
import numpy as np
import cv2 as cv
import sklearn#Lib para misturar as paradinhas
from Desktop.Desenvovendo_AI.VISAO_COMPUTACIONAL.libs import extracting_data_from_video #Lib para mexer com videos 
from Desktop.Desenvovendo_AI.VISAO_COMPUTACIONAL.libs import feedback_humanizado   #Feedback humanizado testa o modelo para um splitted video





#==============================================Preparando o dataset

video_cap = '/home/salomao/Desktop/Validacao.mp4'   #video usado no feedback humanizado


train_1 = extracting_data_from_video.capturing_frames_appended([('/home/salomao/Desktop/Object_background_contant.mp4', 1),('/home/salomao/Desktop/Objeto1.mp4', 20),('/home/salomao/Desktop/Objeto2.mp4', 20)],DEBUG=0)
train_0 = extracting_data_from_video.capturing_frames_appended([('/home/salomao/Desktop/Ambient_background_contant.mp4', 1),('/home/salomao/Desktop/Ambient1.mp4', 20),('/home/salomao/Desktop/Ambient2.mp4', 20)],DEBUG=0)

val_1 = train_1[int(0.8*len(train_1)):len(train_1)]#20% do dataset do train_1 vai para validação
val_0 = train_0[int(0.8*len(train_0)):len(train_0)]#20% do dataset do train_1 vai para validação
train_1 = train_1[0:int(0.8*len(train_1))]#80% do dataset do train_1 vai para treinamento
train_0 = train_0[0:int(0.8*len(train_0))]#80% do dataset do train_1 vai para treinamento

train_data,label_array_train  = extracting_data_from_video.shufle_balance(train_1,train_0)#Trecho relacionado ao balanceamento do dataset
val_data,label_val = extracting_data_from_video.shufle_balance(val_1,val_0)



## Parametros de treinamentos =======================================================================================
    
batch_size = 5 #Este parametro define o paralelismo que a sua rede é treinada... Quanto maior, mais rapido
epochs = 1


# Modelo ===========================================================================================================

    #Input da rede 
layer_in = keras.layers.Input(shape= (500,500,3))

     

conv1 = keras.layers.Conv2D(32, kernel_size=3, padding='valid', activation= 'relu')(layer_in)
pool1  = keras.layers.MaxPool2D(pool_size=(3,3))(conv1)

conv4 = keras.layers.Conv2D(32, kernel_size=3, padding='valid', activation= 'relu')(pool1)
pool5  = keras.layers.MaxPool2D(pool_size=(3,3))(conv4)
conv5 = keras.layers.Conv2D(32, kernel_size=3, padding='valid', activation= 'relu')(pool5)
pool5  = keras.layers.MaxPool2D(pool_size=(3,3))(conv5)

flat2 = keras.layers.Flatten()(pool5)


dense1 = keras.layers.Dense(200,activation='relu',kernel_regularizer=keras.regularizers.l2(0.01))(flat2)
layer_out = keras.layers.Dense(1,activation='sigmoid')(dense1)


model = keras.models.Model(inputs=layer_in,outputs=layer_out)   
                         
model.summary()

model.compile(optimizer= 'adam',  loss='binary_crossentropy', metrics= ['accuracy'])









    #Daqui pra frente ja é o treinamento
for i in range(epochs):
    
    history = model.fit(train_data, label_array_train, validation_data = (val_data,label_val), batch_size=batch_size, epochs=1, verbose=1)
    train_data,label_array_train  = extracting_data_from_video.shufle_balance(train_1,train_0)#Trecho relacionado ao balanceamento do dataset

model.save('/home/salomao/Desktop/detector_rel.h5')
#===============================================================================================================
ask='\0'
ask = input('feedback? (y/n)')

if ask=='y'  :
    feedback_humanizado.feedback_splitted_by_video( model , video_cap, 0.5)
    
