#Autor: Guilherme Salomão Agostini 
#Email: guime.sa9@gmail.com
#Referecias importantes:


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
    # teste = np.multiply(img,ann)
    # cv.imshow("Imagem Original",img)
    # cv.waitKey()
    # cv.destroyAllWindows()
    # cv.imshow("preview",teste)
    # ask = cv.waitKey()
    # cv.destroyAllWindows()
    # if ask==113:
    #     break

    
images = np.array(images)
annotation = np.array(annotation)
annotation = np.reshape(annotation,(len(annotation),img_size[0],img_size[1], -1))

#%% Inicio augmentation
from tensorflow import keras
import numpy as np
datagen_args = dict(    
    rotation_range=180,
    width_shift_range=0.3,
    height_shift_range=0.3,
    zoom_range=0.5,
    horizontal_flip=True,
    vertical_flip=True)

img_datagen = keras.preprocessing.image.ImageDataGenerator(**datagen_args)
mask_datagen = keras.preprocessing.image.ImageDataGenerator(**datagen_args)

seed = 1
batch_size = 5
teste_num = 3
img_datagen.fit(images, augment=True,seed = seed)
mask_datagen.fit(annotation, augment=True, seed =seed )



#%% Testando augmentation
it1 = img_datagen.flow(images , batch_size=batch_size, seed=seed)
it2 = mask_datagen.flow(annotation , batch_size=batch_size, seed=seed)

for i in range(teste_num):
    img_batch = it1[i]
    ann_batch = it2[i]
    for idx in range(batch_size):
        img = img_batch[idx]
        ann = ann_batch[idx]
        teste = np.multiply(img,ann)
        cv.imshow("Imagem Original",img)
        cv.waitKey()
        cv.destroyAllWindows()
        cv.imshow("preview",teste)
        cv.waitKey()
        cv.destroyAllWindows()       
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        

