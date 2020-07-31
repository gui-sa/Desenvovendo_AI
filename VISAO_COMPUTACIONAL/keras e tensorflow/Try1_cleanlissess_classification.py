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

train_1 = extracting_data_from_video.capturing_frames('/home/salomao/Desktop/Object_background_contant.mp4', 1)#lendo um video com uma divisoria de 10
temp = extracting_data_from_video.capturing_frames('/home/salomao/Desktop/Objeto2.mp4', 10)#lendo um video com uma divisoria de 10
train_1 = np.append(train_1,temp,axis=0)#Somando os dois videos

train_0 = extracting_data_from_video.capturing_frames('/home/salomao/Desktop/Ambient_background_contant.mp4', 1)#lendo o video todo
temp = extracting_data_from_video.capturing_frames('/home/salomao/Desktop/Ambient1.mp4', 20)#lendo o video todo
train_0 = np.append(train_0,temp,axis=0)#Somando os dois videos
temp = extracting_data_from_video.capturing_frames('/home/salomao/Desktop/Ambient2.mp4', 20)#lendo o video todo
train_0 = np.append(train_0,temp,axis=0)#Somando os dois videos






#Trecho relacionado ao balanceamento do dataset
if (len(train_1)>len(train_0)):#Se o label 1 for maior que o label 0
    train_1 = sklearn.utils.shuffle(train_1, random_state=0)#misturo o data de label 1
    train_data = train_1[0:len(train_0)]#Pego a mesma quantidade do label zero e coloco em uma nova variavel
    label_1 = np.ones(len(train_data))#Crio um vetor dizendo que tem um certa quantidade de label 1, o mesmo tamanho do data 
    label_0 = np.zeros(len(train_0))#Crio um vetor dizendo que tem um certa quantidade de label 0, o mesmo tamanho do data 
    label_array_train = np.append(label_1,label_0,axis=0)#Somo os vetores de label 1 e depois zero
    train_data = np.append(train_data,train_0,axis=0)#Somo os vetores de data 1 e depois zero
    train_data,label_array_train = sklearn.utils.shuffle(train_data,label_array_train, random_state=0)#Misturo o data juntamente com o label, sem deslinkar a posicao
else:#Se o label 0 for maior que o label 1
    train_0 = sklearn.utils.shuffle(train_0, random_state=0)#misturo o data de label 0
    train_data = train_0[0:len(train_1)]#Pego a mesma quantidade do label um e coloco em uma nova variavel
    label_1 = np.ones(len(train_1))#Crio um vetor dizendo que tem um certa quantidade de label 1, o mesmo tamanho do data 
    label_0 = np.zeros(len(train_data))#Crio um vetor dizendo que tem um certa quantidade de label 0, o mesmo tamanho do data 
    label_array_train = np.append(label_1,label_0,axis=0)#Somo os vetores de label 1 e depois zero
    train_data = np.append(train_1,train_data,axis=0)#Somo os vetores de data 1 e depois zero
    train_data,label_array_train = sklearn.utils.shuffle(train_data,label_array_train, random_state=0)#Misturo o data juntamente com o label, sem deslinkar a posicao   


## Parametros de treinamentos =======================================================================================
    
batch_size = 5 #Este parametro define o paralelismo que a sua rede é treinada... Quanto maior, mais rapido
epochs = 5#Quantos conjuntos de batchs se repetem




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
    history = model.fit(train_data, label_array_train, batch_size=batch_size, epochs=1, verbose=1)
    if (len(train_1)>len(train_0)):
        train_1 = sklearn.utils.shuffle(train_1, random_state=0)#Os vetores devem estar no formato numpy.
        train_data = train_1[0:len(train_0)]
        label_1 = np.ones(len(train_data))
        label_0 = np.zeros(len(train_0))
        label_array_train = np.append(label_1,label_0,axis=0)
        train_data = np.append(train_data,train_0,axis=0)
        train_data,label_array_train = sklearn.utils.shuffle(train_data,label_array_train, random_state=0)#Os vetores devem estar no formato numpy.
    else:
        train_0 = sklearn.utils.shuffle(train_0, random_state=0)#Os vetores devem estar no formato numpy.
        train_data = train_0[0:len(train_1)]
        label_1 = np.ones(len(train_1))
        label_0 = np.zeros(len(train_data))
        label_array_train = np.append(label_1,label_0,axis=0)
        train_data = np.append(train_1,train_data,axis=0)
        train_data,label_array_train = sklearn.utils.shuffle(train_data,label_array_train, random_state=0)#Os vetores devem estar no formato numpy.    


model.save('/home/salomao/Desktop/detector_rel.h5')
#===============================================================================================================
ask='\0'
ask = input('feedback? (y/n)')

if ask=='y'  :
    feedback_humanizado.feedback_splitted_by_video( model , video_cap, 0.5)
    
