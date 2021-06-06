#Esse codigo tem por objetivo aplicar a metodologia para geracao de mascaras proposto pelo link: 
#https://app.diagrams.net/#G14rW6Cdf-O6VqQt03aZP6jqBHK4PICo6D
#Autor: Guilherme Salomao Agostini
#Email: guime.sa9@gmail.com

#%% Preparando os caminhos e o ordenamentos dos nomes

import os
import cv2 as cv
import numpy as np

img_size = (800, 800)#quanto maior a imagem, mais preciso sera a segmentacao da rede... porem, maior deverá ser a rede.

path_img = "/home/salomao/Desktop/insulators-dataset/jpg"
path_ann = "/home/salomao/Desktop/insulators-dataset/tiff"

list_img = os.listdir(path_img)#lista todas as imagens que estao nesse path
list_img.sort()
list_img = sorted(list_img,key=len)
list_ann = os.listdir(path_ann)#lista todas as anotacoes que estao nesse path
list_ann.sort()
list_ann = sorted(list_ann,key=len)

#%% Leitura de todas as imagens e todas as anotações... a parte comentada é para teste

images =  []
annotation= []
for img_idx,ann_idx in zip(list_img,list_ann):
    path_img_idx = os.path.join(path_img, img_idx)#criando um caminho total para cada imagem
    path_ann_idx = os.path.join(path_ann, ann_idx)
    img  =  cv.imread(path_img_idx)#leitura da imagem em rgb
    ann =   cv.imread(path_ann_idx, cv.IMREAD_GRAYSCALE)#leitura da mascara em tons de cinza
    _, ann = cv.threshold(ann,1,1,cv.THRESH_BINARY)#convertendo de cinza para binario preto-branco
    img = cv.resize(img,img_size,interpolation = cv.INTER_AREA)/255.0#reescalando tamanho da imagem e passando para float
    ann = cv.resize(ann,img_size,interpolation = cv.INTER_AREA)  
    
    images.append(img)#append na lista de imagens
    annotation.append(ann)#append na lista de anotacoes
    ##==========================DEBUG
    # ann = np.expand_dims(ann, axise=-1)#cria-se uma nova dimensao para que seja possivel fazer as multiplicacoes entre ann e img
    # teste = np.multiply(img,ann)#multiplica-se imagem e anotacao para obter o png
    # cv.imshow("Imagem Original",img)
    # cv.waitKey()
    # cv.destroyAllWindows()
    # cv.imshow("preview",teste)
    # ask = cv.waitKey()
    # cv.destroyAllWindows()
    # if ask==113:#sai se precionar ´q´
    #     break

    
images = np.array(images)#passa para vetor de numpy
annotation = np.array(annotation)
annotation = np.reshape(annotation,(len(annotation),img_size[0],img_size[1], -1))#expandindo uma nova dimensao para que seja aplicavel no keras

#%% funcao que inverte a mascara binaria

#Para usá-la:
#ann =   cv.imread("/home/salomao/Desktop/generated_masks/0.jpg", cv.IMREAD_GRAYSCALE)
#ann = invert_mask(ann)
#cv.imshow("teste",ann)
def invert_mask(ann):
    ann = np.round(ann)
    ann = 1 - ann
    ann = ann
    return ann


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
batch_size = 5  
seed = 1#a randomizacao precisa ser identicas, nome seed, para que as imagens se casem apos a augmentacao
teste_num = 15

#%% Caminho das imagens que servirao de background.
path_background = "/home/salomao/Desktop/background/"

list_background = os.listdir(path_background)#lista todas as imagens que estao nesse path
for indx in range(len(list_background)):
    list_background[indx] = os.path.join(path_background,list_background[indx])


#%% Obtendo augmentation, casando as imagens e povoando a pasta
import random 
it1 = img_datagen.flow(images , batch_size=batch_size, seed=seed)
it2 = mask_datagen.flow(annotation , batch_size=batch_size, seed=seed)

for i in range(teste_num):
    img_batch = it1[i]
    ann_batch = it2[i]
    for idx in range(batch_size): 
        img = img_batch[idx]
        ann = ann_batch[idx]
        rand = random.randrange(0,(len(list_background)))
        background  =  cv.imread(list_background[rand])#leitura da imagem em rgb
        background = cv.resize(background,img_size,interpolation = cv.INTER_AREA)
        # cv.imshow("background_original",background)
        # ask = cv.waitKey()
        background = np.multiply(background,invert_mask(ann))
        background = np.uint8(background)
        img = np.multiply(img,ann)
        img = img*255
        img = np.uint8(img)
        img_final = background + img
        img_final = np.uint8(img_final)
        cv.imwrite("/home/salomao/Desktop/generated_png/" + str(idx + i*batch_size) + ".jpg", img_final)
        cv.imwrite("/home/salomao/Desktop/generated_masks/" + str(idx + i*batch_size) + ".jpg", ann*255)
        #=============DEBUG
        # cv.imshow("Imagem Original",img)
        # ask = cv.waitKey()
        # cv.imshow("Imagem montada",img_final)
        # ask = cv.waitKey()
        # cv.destroyAllWindows()       

