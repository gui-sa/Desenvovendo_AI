#Autor: Guilherme Salomão Agostini 
#Email: guime.sa9@gmail.com
#Referecias importantes:


#%% Prepare paths of input images and target segmentation masks
import os
from tensorflow import keras
import cv2 as cv
import numpy as np



path_img = "/home/salomao/Desktop/insulators-dataset/jpg"
path_ann = "/home/salomao/Desktop/insulators-dataset/tiff"

list_img = os.listdir(path_img)
list_img.sort()
list_img = sorted(list_img,key=len)
list_ann = os.listdir(path_ann)
list_ann.sort()
list_ann = sorted(list_ann,key=len)

for img_idx,ann_idx in zip(list_img,list_ann):
    path_img_idx = os.path.join(path_img, img_idx)
    path_ann_idx = os.path.join(path_ann, ann_idx)
    img  =  cv.imread(path_img_idx)
    ann =   cv.imread(path_ann_idx)
    _, ann = cv.threshold(ann,1,1,cv.THRESH_BINARY)
    
    img = np.multiply(img,ann) 
    img_masked = cv.resize(img,(800,800),interpolation = cv.INTER_AREA)
    cv.imshow("preview",img_masked)
    cv.waitKey()
    cv.destroyAllWindows()


