#Autor: Guilherme SalomÃ£o Agostini 
#Email: guime.sa9@gmail.com
#Referecias importantes:


#%% Prepare paths of input images and target segmentation masks
import os
import cv2 as cv
import numpy as np

img_size = (160, 160)

path_img = "/home/salomao/Desktop/insulators-dataset/jpg"

list_img = os.listdir(path_img)

images =  []
original = []
for img_idx in list_img:
    path_img_idx = os.path.join(path_img, img_idx)
    img  =  cv.imread(path_img_idx)
    original.append(img)
    img = cv.resize(img,img_size,interpolation = cv.INTER_AREA)
    img = img/255.0
    images.append(img)


images = np.array(images)
#%% Loadando modelo
from tensorflow import keras
import numpy as np

model = keras.models.load_model("/home/salomao/Desktop/insulators.h5")
pred = model.predict(images)

#%% Checando a segmentação de todas as imagens do dataset
for i in range(len(images)):
    res = pred[i]
    res = np.argmax(res,-1)
    res = np.expand_dims(res, axis=-1)
    masked = np.multiply(images[i],res)
    cv.imshow("imagem original " + str(i),images[i])
    cv.waitKey()
    cv.destroyAllWindows()
    cv.imshow("Segmented " + str(i),masked)
    ask = cv.waitKey()
    cv.destroyAllWindows()
    if ask==113:
        break


#%% Aproveitando o mesmo predict para a imagem em alta resolucao
i = 0
res = pred[i]
res = np.argmax(res,-1)
res = np.float32(res)
size =( original[i].shape[1],original[i].shape[0])
res = cv.resize(res, size, interpolation = cv.INTER_AREA)
res = np.round(res)
res = np.expand_dims(res, axis=-1)
masked = np.multiply(original[i]/255.0,res)
cv.imshow("Segmented Huge",masked)
ask = cv.waitKey()
cv.destroyAllWindows()

#%% Aproveitando o mesmo predict para a imagem em alta resolucao (controlando a resolucao)
i = 0
res = pred[i]
res = np.argmax(res,-1)
res = np.float32(res)
size =(800,800)
res = cv.resize(res, size, interpolation = cv.INTER_AREA)
res = np.round(res)
res = np.expand_dims(res, axis=-1)
img = cv.resize(original[i], size, interpolation = cv.INTER_AREA)
masked = np.multiply(img/255.0,res)
cv.imshow("Segmented Controlled",masked)
ask = cv.waitKey()
cv.destroyAllWindows()

