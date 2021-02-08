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
import time

ask = input('O que deseja  fazer? (Se voce desejar treinar a rede, digite ( p )// Se voce desejar detectar um objeto em imagem digite ( o )// Qualquer outra letra resultara em exit:   ')


if ask=='p':
    
    # Arrumando o dataset =============================================================================================
    #Temos que organizar um dataset x no train_data, um data y no label_array_train
    #Para validação temos o val_data e label_val
    
    train_data = extracting_data_from_video.Load_data('/home/salomao/Desktop/data_x')
    label_array_train = extracting_data_from_video.Load_data('/home/salomao/Desktop/data_y')
    
    train_data,label_array_train = sklearn.utils.shuffle(train_data,label_array_train, random_state=0)#Os vetores devem estar no formato numpy.
    
    train_data,label_array_train = sklearn.utils.shuffle(train_data,label_array_train, random_state=0)#Vou fazer duas vezes
    
    
    # Parametros de treinamentos =======================================================================================
    
    batch_size = 5 #Este parametro define o paralelismo que a sua rede é treinada... Quanto maior, mais rapido
    epochs = 5#Quantos conjuntos de batchs se repetem
    batch_p_step = 11#Quantos batchs formam uma epoca
    
    #Optmizer config=======================================================================================================
    keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
    
    # Modelo ===========================================================================================================
    
    #Input da rede 
    layer_in = keras.layers.Input(shape= (500,500,3))
    
     
    #Linha de acao disturtiva 2
    conv1_l2 = keras.layers.Conv2D(32, kernel_size=3, padding='same', activation= 'relu')(layer_in)
    pool1_l2  = keras.layers.MaxPool2D(pool_size=(7,7))(conv1_l2)
    conv2_l2 = keras.layers.Conv2D(128, kernel_size=3, padding='same', activation= 'relu')(pool1_l2)
    pool2_l2  = keras.layers.MaxPool2D(pool_size=(7,7))(conv2_l2)
    flat_l2 = keras.layers.Flatten()(pool2_l2)
    #Linha de acao disturtiva 3
#    conv1_l3  = keras.layers.Conv2D(8, kernel_size=3, padding='same', activation= 'relu')(layer_in)
#    pool1_l3  = keras.layers.MaxPool2D()(conv1_l3)
#    conv2_l3  = keras.layers.Conv2D(16, kernel_size=3, padding='same', activation= 'relu')(pool1_l3)
#    pool2_l3  = keras.layers.MaxPool2D()(conv2_l3)
#    conv3_l3  = keras.layers.Conv2D(32, kernel_size=3, padding='same', activation= 'relu')(pool2_l3)
#    pool3_l3  = keras.layers.MaxPool2D()(conv3_l3)
#    conv4_l3  = keras.layers.Conv2D(64, kernel_size=3, padding='same', activation= 'relu')(pool3_l3)
#    pool4_l3  = keras.layers.MaxPool2D()(conv4_l3)
#    conv5_l3  = keras.layers.Conv2D(128, kernel_size=3, padding='same', activation= 'relu')(pool4_l3)
#    pool5_l3  = keras.layers.MaxPool2D()(conv5_l3)
#    conv6_l3  = keras.layers.Conv2D(256, kernel_size=3, padding='same', activation= 'relu')(pool5_l3)
#    pool6_l3  = keras.layers.MaxPool2D()(conv6_l3)
#    flat_l3 = keras.layers.Flatten()(pool6_l3) 
    #Point of connection
#    concatenate1 = keras.layers.concatenate([flat_l2, flat_l3])
    #Rede fully connected1
    dense1 = keras.layers.Dense(100,activation='relu')(flat_l2)
    layer_out = keras.layers.Dense(1,activation='sigmoid')(dense1)
    
    
    model = keras.models.Model(inputs=layer_in,outputs=layer_out)   
                             
    model.summary()
    
    
    
    
    model.compile(optimizer= 'adam',  loss='binary_crossentropy', metrics= ['accuracy'])
    
    history = model.fit(train_data, label_array_train, batch_size=batch_size,epochs=epochs, verbose=1)
    
    #Avaliando o modelo =======================================================================================================
    
    plt.plot(history.history['loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
    
    plt.plot(history.history['accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
    
    #Salvando o modelo ===========================================================================================================
    
    model.save('/home/salomao/Desktop/detector_rel.h5')









else:
    if ask=='o':
        diretorio_model = '/home/salomao/Desktop/detector_rel.h5'#diretorio para o modelo
        model = keras.models.load_model(diretorio_model)#loadando um modelo
        start_time = time.time()#Para captar o tempo===============================================================
        img = cv.imread('/home/salomao/Desktop/Teste1.jpeg')#Carrego a imagem
        split = extracting_data_from_video.spliting_image(img, shape=(200,200))#Splito ela nas divisorias e mudo elas para o mesmo input
        pred = model.predict(split)#faço a prediçao com base no modelo
        print("--- %s seconds ---" % (time.time() - start_time))#printo o tempo para fazer tudo isso 
        cv.imshow("cropped",split[2] )
        cv.waitKey()
        cv.destroyAllWindows()




