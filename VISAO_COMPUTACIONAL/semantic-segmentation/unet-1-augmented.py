#Autor: Guilherme SalomÃ£o Agostini 
#Email: guime.sa9@gmail.com
#Referecias importantes:
#    https://pypi.org/project/keras-unet/#Customizable-U-Net
#    https://digitalslidearchive.github.io/HistomicsTK/examples/annotations_to_semantic_segmentation_masks.html
#    https://keras.io/api/preprocessing/image/
#    https://github.com/tensorflow/tensorflow/issues/32357
#%% dataset

#%% Prepare paths of input images and target segmentation masks
import os
import cv2 as cv
import numpy as np

img_size = (160, 160)

path_img = "/home/salomao/Desktop/insulators-dataset/jpg"
path_ann = "/home/salomao/Desktop/insulators-dataset/tiff"

list_img = os.listdir(path_img)
list_img.sort()
list_img = sorted(list_img,key=len)
list_ann = os.listdir(path_ann)
list_ann.sort()
list_ann = sorted(list_ann,key=len)

images =  []
annotation= []
for img_idx,ann_idx in zip(list_img,list_ann):
    path_img_idx = os.path.join(path_img, img_idx)
    path_ann_idx = os.path.join(path_ann, ann_idx)
    img  =  cv.imread(path_img_idx)
    ann =   cv.imread(path_ann_idx, cv.IMREAD_GRAYSCALE)
    _, ann = cv.threshold(ann,1,1,cv.THRESH_BINARY)
    img = cv.resize(img,img_size,interpolation = cv.INTER_AREA)/255.0
    ann = cv.resize(ann,img_size,interpolation = cv.INTER_AREA)  
    
    images.append(img)
    annotation.append(ann)
    
    # img = np.multiply(img,ann) 
    # cv.imshow("preview",img)
    # cv.waitKey()
    # cv.destroyAllWindows()
    
images = np.array(images)
annotation = np.array(annotation)
annotation = np.reshape(annotation,(len(annotation),img_size[0],img_size[1], -1))
#%% Dividindo dataset em treinamento e validação
import sklearn

images,annotation = sklearn.utils.shuffle(images,annotation)#misturo o data de label 1

train_data = images[0:int(len(images)*0.8)]
val_data = images[int(len(images)*0.8):]

label_data = annotation[0:int(len(annotation)*0.8)]
val_label_data = annotation[int(len(annotation)*0.8):]

images = None
annotation = None


#%% Inicio augmentation
import numpy as np
import keras

datagen_args = dict(    
    rotation_range=180,
    width_shift_range=0.3,
    height_shift_range=0.3,
    zoom_range=0.5,
    horizontal_flip=True,
    vertical_flip=True
    )

img_datagen = keras.preprocessing.image.ImageDataGenerator(**datagen_args)
mask_datagen = keras.preprocessing.image.ImageDataGenerator(**datagen_args)

seed = 1


# img_datagen.fit(train_data, augment=True,seed = seed)
# mask_datagen.fit(label_data, augment=True, seed = seed )


batch_size = 5
img_generator = img_datagen.flow(train_data , batch_size=batch_size, seed=seed)
mask_generator = mask_datagen.flow(label_data , batch_size=batch_size, seed=seed)

#%% Augmentation
#O objetivo é augmentar o data até aproximadamente o dobro do tamanho... Nao coloquei o datagen junto com o fit, pq por algum motivo ele nao esta aceitando.
import tqdm

teste_num = int ((len(train_data)/batch_size))
for i in tqdm.tqdm(range(teste_num)):
    img_batch = img_generator[i]
    ann_batch = mask_generator[i]
    train_data = np.append(train_data,img_batch,axis=0)
    label_data = np.append(label_data,ann_batch,axis=0)
    # for idx in range(batch_size):
    #     img = img_batch[idx]
    #     ann = ann_batch[idx]
    #     teste = np.multiply(img,ann)
    #     cv.imshow("Imagem Original",img)
    #     cv.waitKey()
    #     cv.destroyAllWindows()
    #     cv.imshow("preview",teste)
    #     cv.waitKey()
    #     cv.destroyAllWindows()       
        
train_data,label_data = sklearn.utils.shuffle(train_data,label_data )#misturo o data de label 1
train_data,label_data = sklearn.utils.shuffle(train_data,label_data )#misturo o data de label 1
#%% model
from keras import layers
num_classes = 2
from keras_unet.models import custom_unet

keras.backend.clear_session()

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

        # Project residual
        residual = layers.Conv2D(filters, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    ### [Second half of the network: upsampling inputs] ###

    for filters in [512, 256, 128, 64, 32]:
        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.UpSampling2D(2)(x)

        # Project residual
        residual = layers.UpSampling2D(2)(previous_block_activation)
        residual = layers.Conv2D(filters, 1, padding="same")(residual)
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    # Add a per-pixel classification layer
    outputs = layers.Conv2D(num_classes, 3, activation="softmax", padding="same")(x)

    # Define the model
    model = keras.Model(inputs, outputs)
    return model


# Free up RAM in case the model definition cells were run multiple times
#keras.backend.clear_session()
model = custom_unet(
    input_shape=img_size +  (3,),
    use_batch_norm=True,
    num_classes=2,
    filters=32,
    dropout=0.2,
    output_activation='softmax')
# Build model
# model = get_model(img_size, num_classes)
model.summary()


#%% Train the model

# Configure the model for training.
# We use the "sparse" version of categorical_crossentropy
# because our target data is integers.
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")

callbacks = [
    keras.callbacks.ModelCheckpoint("insulators-augmented.h5", save_best_only=True)
]

# Train the model, doing validation at the end of each epoch.
epochs = 50

history = model.fit(train_data, label_data ,batch_size= batch_size , epochs=epochs, validation_data= (val_data,val_label_data), callbacks=callbacks)


#%%plots
import matplotlib.pyplot as plt


plt.figure()
plt.plot(np.add(list(range(epochs)),1), history.history["loss"], lw = 2 , color ="r", label = "Treinamento")
plt.plot(np.add(list(range(epochs)),1), history.history["val_loss"], lw = 2 , color ="y", label = "Validacao")
plt.legend()
plt.title("Aprendizado da rede conforme o tempo - erro")
plt.xlabel("Epocas")
plt.ylabel("Erro da funcao crossentropy")
plt.grid(True)
plt.show()




#%%teste
import keras
import numpy as np

#model = keras.models.load_model("/home/salomao/Desktop/Desenvovendo_AI/VISAO_COMPUTACIONAL/semantic-segmentation/insulators.h5")
teste1 = val_data[0:1]
res = model.predict(teste1)[0]
res = np.argmax(res,-1)
res = np.expand_dims(res, axis=-1)
masked = np.multiply(teste1[0],res)
cv.imshow("imagem original",teste1[0])
cv.waitKey()
cv.destroyAllWindows()
cv.imshow("previsto",masked)
cv.waitKey()
cv.destroyAllWindows()
