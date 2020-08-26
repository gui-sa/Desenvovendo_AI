#



import tensorflow as tf
import keras
import numpy as np
import cv2 as cv
import sklearn#Lib para misturar as paradinhas
import matplotlib.pyplot as plt
from Desktop.Desenvovendo_AI.VISAO_COMPUTACIONAL.libs import extracting_data_from_video #Lib para mexer com videos 
from Desktop.Desenvovendo_AI.VISAO_COMPUTACIONAL.libs import feedback_humanizado   #Feedback humanizado testa o modelo para um splitted video





#%% ==============================================Preparando o dataset

video_cap = '/home/salomao/Desktop/Validacao.mp4'   #video usado no feedback humanizado


train_1 = extracting_data_from_video.capturing_frames_appended([('/home/salomao/Desktop/Object_background_contant.mp4', 1),('/home/salomao/Desktop/Objeto1.mp4', 20),('/home/salomao/Desktop/Objeto2.mp4', 20)],DEBUG=0)
train_0 = extracting_data_from_video.capturing_frames_appended([('/home/salomao/Desktop/Ambient_background_contant.mp4', 1),('/home/salomao/Desktop/Ambient1.mp4', 1 ),('/home/salomao/Desktop/Ambient2.mp4', 1)],DEBUG=0)

val_1 = train_1[int(0.8*len(train_1)):len(train_1)]#20% do dataset do train_1 vai para validação
val_0 = train_0[int(0.8*len(train_0)):len(train_0)]#20% do dataset do train_1 vai para validação
train_1 = train_1[0:int(0.8*len(train_1))]#80% do dataset do train_1 vai para treinamento
train_0 = train_0[0:int(0.8*len(train_0))]#80% do dataset do train_1 vai para treinamento
temp= extracting_data_from_video.data_augment(train_1, command = "constant_color_scaler_shufle", show=False)
train_1 = np.append(train_1,temp[0:int(len(temp)/2)], axis=0)

train_data,label_array_train  = extracting_data_from_video.shufle_balance(train_1,train_0)#Trecho relacionado ao balanceamento do dataset
val_data,label_val = extracting_data_from_video.shufle_balance(val_1,val_0)


datagen = keras.preprocessing.image.ImageDataGenerator(
        rotation_range=45,
        zoom_range=0.5,
        horizontal_flip=True)

datagen.fit(train_data)
#%% Parametros de treinamentos =======================================================================================
    
batch_size = 5 #Este parametro define o paralelismo que a sua rede é treinada... Quanto maior, mais rapido
epochs = 20
val_acc = 0

#%% Modelo ===========================================================================================================

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

historico_erro = [10] #Para salvar os valores de acuracia da validaçao
historico_val = [0] #Para salvar os valores de acuracia da validaçao
historico = [0]#Para salvar os valores de acuracia de treinamento
historico_epocas = list(range(epochs+1))#Salvar uma lista de epocas
for i in range(epochs):
    
    #history = model.fit(train_data, label_array_train, validation_data = (val_data,label_val), batch_size=batch_size, epochs=1, verbose=1)
    history = model.fit(datagen.flow(train_data, label_array_train, batch_size=batch_size), validation_data = (val_data,label_val), epochs=1, verbose=1)
    
    train_data,label_array_train  = extracting_data_from_video.shufle_balance(train_1,train_0)#Trecho relacionado ao balanceamento do dataset
    historico_val.append(history.history['val_accuracy'][-1])
    historico.append(history.history['accuracy'][-1])
    historico_erro.append(history.history['loss'][-1])
    if history.history['val_accuracy'][-1] > val_acc:
        model.save('/home/salomao/Desktop/detector_rel.h5')
        val_acc = history.history['val_accuracy'][-1]
        print("\n\nModelo salvo!\n\n")


#%%====================================grafico de treinamento

plt.figure()
plt.plot(historico_epocas, historico_val, lw = 2 , color ="b", label = "Validacão")
plt.plot(historico_epocas, historico, lw = 2 , color ="k", label = "Treinamento")
plt.legend()
plt.title("Aprendizado da rede conforme o tempo - acuracia")
plt.xlabel("Epocas")
plt.ylabel("Taxa de acurácia")
plt.show()
plt.figure()
plt.plot(historico_epocas, historico_erro, lw = 2 , color ="r", label = "Treinamento")
plt.legend()
plt.title("Aprendizado da rede conforme o tempo - erro")
plt.xlabel("Epocas")
plt.ylabel("Erro da funcao crossentropy")
plt.show()
#%%===============================================================================================================
model = keras.models.load_model('/home/salomao/Desktop/detector_rel.h5')
ask='\0'
ask = input('feedback? (y/n)')

if ask=='y'  :
    feedback_humanizado.feedback_splitted_by_video( model , video_cap, 0.5)
    
