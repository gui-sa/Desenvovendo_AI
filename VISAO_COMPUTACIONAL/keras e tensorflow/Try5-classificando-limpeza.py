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
#from Desktop.Desenvovendo_AI.VISAO_COMPUTACIONAL.libs import extracting_data_from_video #Lib para mexer com videos 
import sklearn


#%% ==============================================Criando o dataset

img_size = (800, 800)

path_img = "/home/salomao/Desktop/insulators-dataset/jpg"
path_ann = "/home/salomao/Desktop/insulators-dataset/tiff"

list_img = os.listdir(path_img)
list_img.sort()
list_img = sorted(list_img,key=len)
list_ann = os.listdir(path_ann)
list_ann.sort()
list_ann = sorted(list_ann,key=len)
ask = "q"
images =  []
for img_idx,ann_idx in zip(list_img,list_ann):
    path_img_idx = os.path.join(path_img, img_idx)
    path_ann_idx = os.path.join(path_ann, ann_idx)
    img  =  cv.imread(path_img_idx)
    ann =   cv.imread(path_ann_idx, cv.IMREAD_GRAYSCALE)
    _, ann = cv.threshold(ann,1,1,cv.THRESH_BINARY)
    img = cv.resize(img,img_size,interpolation = cv.INTER_AREA)/255.0
    ann = cv.resize(ann,img_size,interpolation = cv.INTER_AREA)  
    
    ann = np.expand_dims(ann, axis=-1)
    img = np.multiply(img,ann) 
    images.append(img)
    if ask==113:
        continue    
    cv.imshow("preview",img)
    ask = cv.waitKey()
    cv.destroyAllWindows()

    
images = np.array(images)
train_0 = images[0:63]
train_1 = images[64:]
images = None

#%%============== Splitando o dataset:

val_1 = train_1[int(0.8*len(train_1)):len(train_1)]#20% do dataset do train_1 vai para validacao
val_0 = train_0[int(0.8*len(train_0)):len(train_0)]#20% do dataset do train_1 vai para validacao
train_1 = train_1[0:int(0.8*len(train_1))]#80% do dataset do train_1 vai para treinamento
train_0 = train_0[0:int(0.8*len(train_0))]#80% do dataset do train_1 vai para treinamento



#%%============  Usando o dataset
train_data,label_array_train  = extracting_data_from_video.shufle_balance(train_1,train_0)#Trecho relacionado ao balanceamento do dataset
val_data,label_val = extracting_data_from_video.shufle_balance(val_1,val_0)

train_1  =None
train_0  =None
val_1 = None
val_0 = None

train_data,label_array_train = sklearn.utils.shuffle(train_data,label_array_train )#misturo o data de label 1
train_data,label_array_train = sklearn.utils.shuffle(train_data,label_array_train )#misturo o data de label 1

#%% Parametros de treinamentos =======================================================================================
    
batch_size = 5 #Este parametro define o paralelismo que a sua rede é treinada... Quanto maior, mais rapido
epochs = 100


#%% Data augmentation

datagen_args = dict(    
    rotation_range=180,
    width_shift_range=0.6,
    height_shift_range=0.6,
    zoom_range=0.5,
    horizontal_flip=True,
    vertical_flip=True
    )

img_datagen = keras.preprocessing.image.ImageDataGenerator(**datagen_args)

batch_size = 5

img_generator = img_datagen.flow(train_data ,label_array_train,  batch_size=batch_size)

#%% model
from tensorflow.keras import layers
from tensorflow import keras


num_classes = 1

def get_model(img_size, num_classes):
    inputs = keras.Input(shape=img_size + (3,)) #(img_size + (3,)) = (160, 160, 3)

    ### [First half of the network: downsampling inputs] ###

    # Entry block
    x = layers.Conv2D(32, 3, strides=2, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    # Blocks 1, 2, 3 are identical apart from the feature depth.
    for filters in [64, 128, 256, 512]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)
        x = keras.layers.Dropout(0.3)(x)
        # Project residual
        residual = layers.Conv2D(filters, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual
    x = keras.layers.GlobalAveragePooling2D()(x)
    
    # x = keras.layers.Dense(1000,activation='relu',kernel_regularizer=keras.regularizers.l2(0.001))(x)
    outputs = keras.layers.Dense(num_classes, activation='sigmoid',kernel_regularizer=keras.regularizers.l2(0.001))(x)
    # Define the model
    model = keras.Model(inputs, outputs)
    return model



# Free up RAM in case the model definition cells were run multiple times
keras.backend.clear_session()

# Build model
model = get_model(img_size, num_classes)
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
history = model.fit(img_generator, steps_per_epoch = int(len(train_data)/batch_size), validation_data = (val_data,label_val), epochs=epochs, verbose=1, callbacks=model_checkpoint_callback )


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




