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
    
    # img = np.multiply(img,ann) 
    # cv.imshow("preview",img)
    # cv.waitKey()
    # cv.destroyAllWindows()
    
images = np.array(images)
annotation = np.array(annotation)
annotation = np.reshape(annotation,(len(annotation),img_size[0],img_size[1], -1))
