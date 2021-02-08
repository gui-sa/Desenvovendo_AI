#Autor: Guilherme SalomÃ£o Agostini 
#Email: guime.sa9@gmail.com
#Referecias importantes:


#%% Prepare paths of input images and target segmentation masks
import os
import cv2 as cv
import numpy as np

img_size = (800, 800)

path_img = "/home/salomao/Desktop/insulators-dataset/Teste"
# path_img = "/home/salomao/Desktop/insulators-dataset/jpg"

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

model = keras.models.load_model("/home/salomao/Desktop/CNN_try1_cleanlisses.h5")
pred = model.predict(images)

#%% Checando a segmentação de todas as imagens do dataset
for i in range(len(images)):
    color = (0, 0, 0)
    result = "limpo"
    if(pred[i]>0.5):
        result = "sujo"
    frame = cv.putText(
            images[i], 
            str(pred[i]) + " = " + result, 
            (int(img_size[0]/2),int(img_size[1]/2)), 
            cv.FONT_HERSHEY_SIMPLEX, 
            0.5,  
            color, 
            2, 
            cv.LINE_AA, 
            False) 
    cv.imshow("Segmented " + str(i),frame)
    ask = cv.waitKey()
    cv.destroyAllWindows()
    if ask==113:
        break

