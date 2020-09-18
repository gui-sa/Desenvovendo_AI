#



import tensorflow as tf
import keras
import numpy as np
import cv2 as cv
import sklearn#Lib para misturar as paradinhas
import matplotlib.pyplot as plt
import os
from Desktop.Desenvovendo_AI.VISAO_COMPUTACIONAL.libs import extracting_data_from_video #Lib para mexer com videos 
from Desktop.Desenvovendo_AI.VISAO_COMPUTACIONAL.libs import feedback_humanizado   #Feedback humanizado testa o modelo para um splitted video





#%% ==============================================Criando o dataset

video_cap = 0   #video usado no feedback humanizado

diretorios_1 = ["/home/salomao/Desktop/pratos/branco/Cropped-limpo"]#Lista de pastas, onde as fotos de etiqueta 1 estarão
diretorios_0 = ["/home/salomao/Desktop/pratos/cenario/void"]#Lista de pastas onde as fotos de etiqueta zero estarão

train_1 = []
for pastas in diretorios_1:
    conteudo = os.listdir(pastas)
    for datas in conteudo:
        datas = os.path.join(pastas, datas)
        train_1.append(cv.resize(cv.imread(datas),(500,500)))     
train_1 = np.array(train_1)



train_0 = []
for pastas in diretorios_0:
    conteudo = os.listdir(pastas)
    for datas in conteudo:
        datas = os.path.join(pastas, datas)
        train_0.append(cv.resize(cv.imread(datas),(500,500)))      
train_0 = np.array(train_0)



temp = extracting_data_from_video.capturing_frames_appended([("/home/salomao/Desktop/pratos/branco/branco-limpo-manha-cenario1.mp4",30),("/home/salomao/Desktop/pratos/branco/branco-limpo-tarde-cenario1.mp4", 30),("/home/salomao/Desktop/pratos/branco/branco-limpo-noite-cenario1.mp4", 30),("/home/salomao/Desktop/pratos/branco/branco-sujo1-noite-cenario1.mp4", 30),("/home/salomao/Desktop/pratos/branco/branco-sujo1-manha-cenario1.mp4", 30),("/home/salomao/Desktop/pratos/branco/branco-sujo1-tarde-cenario1.mp4", 30)],DEBUG=0)
train_1 = np.append(train_1,temp,axis=0)

temp = extracting_data_from_video.capturing_frames_appended([("/home/salomao/Desktop/pratos/cenario/cenario1-manha-maos.mp4", 30),("/home/salomao/Desktop/pratos/cenario/cenario1-manha.mp4", 30 ),("/home/salomao/Desktop/pratos/cenario/cenario1-noite-maos.mp4", 30),("/home/salomao/Desktop/pratos/cenario/cenario1-noite-maos2.mp4", 30),("/home/salomao/Desktop/pratos/cenario/cenario1-tarde.mp4", 30)],DEBUG=0)
train_0 = np.append(train_0,temp,axis=0)


#%%============== Splitando o dataset:

val_1 = train_1[int(0.8*len(train_1)):len(train_1)]#20% do dataset do train_1 vai para validação
val_0 = train_0[int(0.8*len(train_0)):len(train_0)]#20% do dataset do train_1 vai para validação
train_1 = train_1[0:int(0.8*len(train_1))]#80% do dataset do train_1 vai para treinamento
train_0 = train_0[0:int(0.8*len(train_0))]#80% do dataset do train_1 vai para treinamento

#%%============  Usando o dataset
train_data,label_array_train  = extracting_data_from_video.shufle_balance(train_1,train_0)#Trecho relacionado ao balanceamento do dataset
val_data,label_val = extracting_data_from_video.shufle_balance(val_1,val_0)

datagen = keras.preprocessing.image.ImageDataGenerator(
        rotation_range=45,
        horizontal_flip=True)


#%% Parametros de treinamentos =======================================================================================
    
batch_size = 5 #Este parametro define o paralelismo que a sua rede é treinada... Quanto maior, mais rapido
epochs = 50


#%% Modelo ===========================================================================================================

    #Input da rede 
layer_in = keras.layers.Input(shape= (500,500,3))

     

conv1 = keras.layers.Conv2D(8, kernel_size=3, padding='valid', activation= 'relu')(layer_in)
pool1  = keras.layers.MaxPool2D(pool_size=(3,3))(conv1)

conv4 = keras.layers.Conv2D(16, kernel_size=3, padding='valid', activation= 'relu')(pool1)
pool5  = keras.layers.MaxPool2D(pool_size=(3,3))(conv4)
conv5 = keras.layers.Conv2D(32, kernel_size=3, padding='valid', activation= 'relu')(pool5)

pool5  = keras.layers.MaxPool2D(pool_size=(10,10))(conv5)

flat2 = keras.layers.Flatten()(pool5)


dense1 = keras.layers.Dense(50,activation='relu',kernel_regularizer=keras.regularizers.l2(0.001))(flat2)
dense2 = keras.layers.Dense(40,activation='relu',kernel_regularizer=keras.regularizers.l2(0.001))(dense1)
layer_out = keras.layers.Dense(1,activation='sigmoid')(dense2)


model = keras.models.Model(inputs=layer_in,outputs=layer_out)   
                         
model.summary()

model.compile(optimizer= 'adam',  loss='binary_crossentropy', metrics= ['accuracy'])


model_checkpoint_callback = [
    tf.keras.callbacks.ModelCheckpoint(
    filepath="/home/salomao/Desktop/CNN_try1_cleanlisses.h5",
    save_weights_only=False,
    monitor= "val_accuracy",
    mode='max',
    save_best_only=True)]





#%% ================    Daqui pra frente ja é o treinamento

#history = model.fit(train_data, label_array_train, validation_data = (val_data,label_val), batch_size=batch_size, epochs=epochs, verbose=1)
history = model.fit(datagen.flow(train_data, label_array_train, batch_size=batch_size), validation_data = (val_data,label_val), epochs=epochs, verbose=1, callbacks=model_checkpoint_callback )



#%%====================================grafico de treinamento

plt.figure()
plt.plot( np.add(list(range(epochs)),1), history.history["val_accuracy"], lw = 2 , color ="b", label = "Validacão")
plt.plot( np.add(list(range(epochs)),1), history.history["accuracy"], lw = 2 , color ="k", label = "Treinamento")
plt.legend()
plt.title("Aprendizado da rede conforme o tempo - acuracia")
plt.xlabel("Epocas")
plt.ylabel("Taxa de acurácia")
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

#%%===============================================================================================================

ask='\0'
ask = input('feedback? (y/n)')

if ask=='y'  :
    model2 = keras.models.load_model("/home/salomao/Desktop/CNN_try1_cleanlisses.h5")
    feedback_humanizado.feedback_splitted_by_video( model2 , video_cap, 0.5)
    
