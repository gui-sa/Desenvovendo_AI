#Autor: Guilherme SalomÃ£o Agostini 
#Email: guime.sa9@gmail.com
#Referecias importantes:
#    https://pypi.org/project/keras-unet/#Customizable-U-Net
#    https://digitalslidearchive.github.io/HistomicsTK/examples/annotations_to_semantic_segmentation_masks.html

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
    
    ann = np.expand_dims(ann, axis=-1)
    img = np.multiply(img,ann) 
    # cv.imshow("preview",img)
    # ask = cv.waitKey()
    # cv.destroyAllWindows()
    # if ask==113:
    #     break
    
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

# model = custom_unet(
#     input_shape=(160, 160, 3),
#     use_batch_norm=False,
#     num_classes=2,
#     filters=32,
#     dropout=0.2,
#     output_activation='sigmoid')

# Free up RAM in case the model definition cells were run multiple times
keras.backend.clear_session()

# Build model
model = get_model(img_size, num_classes)
model.summary()


#%% Train the model

# Configure the model for training.
# We use the "sparse" version of categorical_crossentropy
# because our target data is integers.
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")

callbacks = [
    keras.callbacks.ModelCheckpoint("insulators.h5", save_best_only=True)
]

# Train the model, doing validation at the end of each epoch.
epochs = 100
batch_size = 5
history = model.fit(train_data,label_data,batch_size=batch_size, epochs=epochs, validation_data= (val_data,val_label_data), callbacks=callbacks)


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
from tensorflow import keras
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
