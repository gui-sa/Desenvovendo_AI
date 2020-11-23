#Autor: Guilherme Salomão Agostini 
#Email: guime.sa9@gmail.com
#Referecias importantes:

#O objetivo � tentar classificar a limpeza com o modelo Xception ja implementado, em um dataset preprocessado por mascaras.

#%% Prepare paths of input images and target segmentation masks
import os
import cv2 as cv
import numpy as np
import tqdm
from tensorflow import keras
import numpy as np
from Desktop.Desenvovendo_AI.VISAO_COMPUTACIONAL.libs import extracting_data_from_video #Lib para mexer com videos 
from Desktop.Desenvovendo_AI.VISAO_COMPUTACIONAL.libs import feedback_humanizado   #Feedback humanizado testa o modelo para um splitted video



#%% ==============================================Criando o dataset

video_cap = 0   #video usado no feedback humanizado

diretorios_1 = ["/home/salomao/Desktop/Cadeias_sujas"]#Lista de pastas, onde as fotos de etiqueta 1 estarão
diretorios_0 = ["/home/salomao/Desktop/Cadeia_limpa"]#Lista de pastas onde as fotos de etiqueta zero estarão


train_1 = []
for pastas in diretorios_1:
    conteudo = os.listdir(pastas)
    for datas in conteudo:
        datas = os.path.join(pastas, datas)
        train_1.append(cv.resize(cv.imread(datas),(800,800)))     
train_1 = np.array(train_1)



train_0 = []
for pastas in diretorios_0:
    conteudo = os.listdir(pastas)
    for datas in conteudo:
        datas = os.path.join(pastas, datas)
        train_0.append(cv.resize(cv.imread(datas),(800,800)))      
train_0 = np.array(train_0)


#%%                       Semantic segmentation

from tensorflow import keras
import numpy as np

model_seg = keras.models.load_model("/home/salomao/Desktop/insulators.h5")


temp = []
for img in tqdm.tqdm(train_1):
    img = cv.resize(img,(160,160))/255.0
    temp.append(img)
temp = np.array(temp)

pred = model_seg.predict(temp)

for i in tqdm.tqdm(range(len(train_1))):
    res = pred[i]
    res = np.argmax(res,-1)
    res = np.float32(res)
    size =(800,800)
    res = cv.resize(res, size, interpolation = cv.INTER_AREA)
    res = np.round(res)
    res = np.expand_dims(res, axis=-1)
    img = train_1[i]
    train_1[i] = np.multiply(img/255.0,res)
    cv.imshow("Segmented Controlled",train_1[i])
    ask = cv.waitKey()
    cv.destroyAllWindows()
    if ask==113:
        break




temp = []
for img in tqdm.tqdm(train_0):
    img = cv.resize(img,(160,160))/255.0
    temp.append(img)
temp = np.array(temp)

pred = model_seg.predict(temp)

for i in tqdm.tqdm(range(len(train_0))):
    res = pred[i]
    res = np.argmax(res,-1)
    res = np.float32(res)
    size =(800,800)
    res = cv.resize(res, size, interpolation = cv.INTER_AREA)
    res = np.round(res)
    res = np.expand_dims(res, axis=-1)
    img = train_0[i]
    train_0[i] = np.multiply(img/255.0,res)
    cv.imshow("Segmented Controlled",train_0[i])
    ask = cv.waitKey()
    cv.destroyAllWindows()
    if ask==113:
        break

model_seg = None
#%%============== Splitando o dataset:

val_1 = train_1[int(0.8*len(train_1)):len(train_1)]#20% do dataset do train_1 vai para validação
val_0 = train_0[int(0.8*len(train_0)):len(train_0)]#20% do dataset do train_1 vai para validação
train_1 = train_1[0:int(0.8*len(train_1))]#80% do dataset do train_1 vai para treinamento
train_0 = train_0[0:int(0.8*len(train_0))]#80% do dataset do train_1 vai para treinamento



#%%============  Usando o dataset
train_data,label_array_train  = extracting_data_from_video.shufle_balance(train_1,train_0)#Trecho relacionado ao balanceamento do dataset
val_data,label_val = extracting_data_from_video.shufle_balance(val_1,val_0)




#%% Parametros de treinamentos =======================================================================================
    
batch_size = 5 #Este parametro define o paralelismo que a sua rede é treinada... Quanto maior, mais rapido
epochs = 10




#%% model
from tensorflow.keras import layers
from tensorflow import keras
from keras_unet.models import custom_unet

num_classes = 2

def get_model(img_size, num_classes):
    inputs = keras.Input(shape=img_size + (3,)) #(img_size + (3,)) = (160, 160, 3)

    ### [First half of the network: downsampling inputs] ###

    # Entry block
    x = layers.Conv2D(32, 3, strides=2, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    # Blocks 1, 2, 3 are identical apart from the feature depth.
    for filters in [64, 128]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(filters, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.1)(x)
    # x = keras.layers.Dense(1000,activation='relu',kernel_regularizer=keras.regularizers.l2(0.001))(x)
    outputs = keras.layers.Dense(1, activation='sigmoid',kernel_regularizer=keras.regularizers.l2(0.001))(x)
    # Define the model
    model = keras.Model(inputs, outputs)
    return model



# Free up RAM in case the model definition cells were run multiple times
keras.backend.clear_session()

# Build model
model = get_model((800,800), num_classes)
model.summary()


#%% compilando

import tensorflow as tf

model.compile(optimizer= 'adam',  loss='binary_crossentropy', metrics= ['accuracy'])


model_checkpoint_callback = [
    tf.keras.callbacks.ModelCheckpoint(
    filepath="/home/salomao/Desktop/CNN_try1_cleanlisses.h5",
    save_weights_only=False,
    monitor= "val_accuracy",
    mode='max',
    save_best_only=True)]


#%% ================    Treinamento

#history = model.fit(train_data, label_array_train, validation_data = (val_data,label_val), batch_size=batch_size, epochs=epochs, verbose=1)
history = model.fit(train_data, label_array_train, batch_size=batch_size, validation_data = (val_data,label_val), epochs=epochs, verbose=1, callbacks=model_checkpoint_callback )


#%%====================================grafico de treinamento
import matplotlib.pyplot as plt
plt.figure()
plt.plot( np.add(list(range(epochs)),1), history.history["val_accuracy"], lw = 2 , color ="b", label = "Validacão")
plt.plot( np.add(list(range(epochs)),1), history.history["accuracy"], lw = 2 , color ="k", label = "Treinamento")
plt.legend()
plt.title("Aprendizado da rede conforme o tempo - acuracia")
plt.xlabel("Epocas")
plt.ylabel("Taxa de acuracia")
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




