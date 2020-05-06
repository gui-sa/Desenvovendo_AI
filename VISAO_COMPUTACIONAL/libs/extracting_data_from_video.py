#Todos sabem que para treinar uma rede neural convolucional com boa acuracia é necessário um grande dataset.
#Outro ponto, é importante que a rede aprenda o objeto de diferentes angulos e backgrounds.
#Para tal, nada melhor do que gravar videos circulando o objeto desejado, e pertubando de maneiras distintas....
#Um vídeo são muitas fotos agrupadas... Esta a ideia!
#Criamos um código que capta vídeos e cria um dataset para cada vídeo.
#Depois pegamos os datasets dos videos, juntamos e misturamos tudo!
#Pronto! temos um dataset com o minimo de dado criado possível.

import cv2 as cv
import numpy as np
import pickle

def capturing_frames(diretorio,div, shape = (500,500)):
    cap = cv.VideoCapture(diretorio)#Settando diretorio para o flow de video
    captured_frames = []
    i = 0 #Preset de variavel
    while(cap.isOpened()):
        i +=1#i Controla o numero de frame
        ret, frame = cap.read()#ret avalia se o video terminou ou não, retornando um booleano. frame é a imagem
        if ret == False:#Quando o video acaba o ret é falso e a janela quebra
            break
        if i%div==0:#Vou coletar somente a cada 10 imagens
            frame = cv.resize(frame, shape, interpolation = cv.INTER_AREA)
            captured_frames.append(frame)
            cv.imshow("preview", frame)
            cv.waitKey(80)#O tempo em ms para passar para o proximo frame
    
    print('\nVideo contém '+str(i-1)+' frames.\n')
    cap.release()#Liberando as memorias utilizadas para processar os videos
    cv.destroyAllWindows()#Fechando todas as janelas auxiliares
    
    captured_frames = np.array(captured_frames)#Retornando um numpy array
    
    return captured_frames

#teste = capturing_frames('/home/salomao/Desktop/Teste.mp4',1, shape=  (200,200))



#================================================================================================================================
#### A função a cima cria um dataset crú para apenas uma classe de imagem por video
#A função tem objetivo de usar o mouse para desenhar um retangulo em uma imagem qualquer com o mouse
#A função retorna para o usuario o centro do retangulo a altura e o comprimento
def on_Mouse(event, x, y, flags, param):
    global pont, status_mouse
    if event == cv.EVENT_LBUTTONDOWN:
#        print('yeah 1')
        pont = [(x,y)]
    else:
        if event == cv.EVENT_LBUTTONUP:
            pont.append((x,y))
            status_mouse = True
#            print('yeah2')

def image_rec(diretorio, shape=(500,500), diretorio_equal_var = False):
    global status_mouse
    status_mouse= False
    if diretorio_equal_var == True:
        frame = diretorio
    else:
        frame = cv.imread(diretorio)
    img = frame
    frame = cv.resize(frame, shape, interpolation = cv.INTER_AREA)
    cv.imshow('preview',frame)
    cv.setMouseCallback('preview', on_Mouse)
    cv.waitKey()
    cv.destroyAllWindows()
#    print('yeah semifinal')
    if status_mouse == True:
#        print('yeah final')
        color = (255,0,0)
        width = 3
        frame = cv.rectangle( frame, pont[0], pont[1], color, width)# desenhar um retangulo de um ponto iniciado no pt1 e terminado no pt2 de cor no espectro rgb 'color' e largura de linha 'width'  
        cv.imshow('preview',frame)
        cv.waitKey()
        cv.destroyAllWindows()
        img = cv.resize(img, shape, interpolation = cv.INTER_AREA)
        w = pont[1][0]-pont[0][0]
        h = pont[1][1]-pont[0][1]
        a = pont[0][0]
        b = pont[0][1]

        return (img,(a,b,h,w))
    img = cv.resize(img, shape, interpolation = cv.INTER_AREA)
    a = 0
    b = 0
    h = 0
    w = 0
    return (img,(a,b,h,w))



#teste = image_rec('/home/salomao/Desktop/Img_dog_1.jpg', shape=  (400,400))




def capturing_frames_rec(diretorio, div, shape = (500,500)):
    cap = cv.VideoCapture(diretorio)#Settando diretorio para o flow de video
    captured_frames = []#Criando lista em branco
    label_data = []
    i = 0 #Preset de variavel
    while(cap.isOpened()):
        i +=1#i Controla o numero de frame
        ret, frame = cap.read()#ret avalia se o video terminou ou não, retornando um booleano. frame é a imagem
        if ret == False:#Quando o video acaba o ret é falso e a janela quebra
            break
        if i%div==0:#Vou coletar somente a cada 10 imagens
            frame,img_t=image_rec(frame, shape, diretorio_equal_var=True)
            label_data.append(img_t)
            captured_frames.append(frame)

    print('\nVideo contém '+str(i-1)+' frames.\n')
    cap.release()#Liberando as memorias utilizadas para processar os videos
    cv.destroyAllWindows()#Fechando todas as janelas auxiliares
    
    captured_frames = np.array(captured_frames)#Retornando um numpy array
    label_data = np.array(label_data)
    return (captured_frames,label_data) 


#testeX,testeY = capturing_frames_rec('/home/salomao/Desktop/Teste.mp4',20, shape=  (200,200))

#Para loadar e salvar os dados criados,use a biblioteca pickle
def preprocess_img_3c(diretorio, shape=(500,500)):
    img = cv.imread(diretorio)
    img = cv.resize(img,shape)
    img = np.reshape(img,(-1,shape[0],shape[1],3))
    img = img/255.
    return img
    
def print_img_rec(img,a,b,w,h, shape=(500,500)):
    img = cv.resize(img,shape)
    img = cv.rectangle( img, (a,b),(a+w,b+h) , (255,0,0), 4)
    cv.imshow('prediction',img)
    cv.waitKey()
    cv.destroyAllWindows()

def Save_data(diretorio, var):   
    pickle_arq = open(diretorio,'wb')
    pickle.dump(var,pickle_arq)
    pickle_arq.close()


def Load_data(diretorio):   
    pickle_arq = open(diretorio,'rb')
    var = pickle.load(pickle_arq)
    pickle_arq.close()
    return var

#Essa funcao tem como objetivo dividir uma imagem ja lida no formato dist de tal forma que ela corta a figura em varias camadas x

def spliting_image(img, shape = (500,500), dist=(3,3)):
    comp_x = np.shape(img)[0]/dist[0]#Pegar o len do eixo x da imagem e dividir na dist x
    comp_y = np.shape(img)[1]/dist[1]#Pegar o len do eixo y da imagem e dividir na dist y
    comp_x = int(comp_x)#pixels sao integers
    comp_y = int(comp_y)#pixels sao integers 
    splited = []
    label = []
    for x in range(dist[0]):
        for y in range(dist[1]):
            lim_inf_x = comp_x*x
            lim_sup_x = comp_x*(x+1)
            lim_inf_y = comp_y*y
            lim_sup_y = comp_y*(y+1)
            frame = cv.resize(img[lim_inf_x:lim_sup_x,lim_inf_y:lim_sup_y],shape)
            print('digite y (minusculo), caso voce veja o objeto desejado')
            cv.imshow("cropped", frame)
            ei = cv.waitKey()
            cv.destroyAllWindows()
            if ei==121:
                label.append(1)
            else:
                label.append(0)
            splited.append(frame)
    return splited,label
    


def capturing_frames_splits(diretorio,div,shape = (500,500), dist=(3,3)):
    cap = cv.VideoCapture(diretorio)#Settando diretorio para o flow de video
    captured_frames = []#Criando lista em branco
    label_data = []
    i = 0 #Preset de variavel
    while(cap.isOpened()):
        i +=1#i Controla o numero de frame
        ret, frame = cap.read()#ret avalia se o video terminou ou não, retornando um booleano. frame é a imagem
        if ret == False:#Quando o video acaba o ret é falso e a janela quebra
            break
        if i%div==0:#Vou coletar somente a cada 10 imagens
            frame_split,img_t = spliting_image(frame, dist=dist, shape=shape)
            label_data = label_data + (img_t  )
            captured_frames = captured_frames + (frame_split)

    print('\nVideo contém '+str(i-1)+' frames.\n')
    cap.release()#Liberando as memorias utilizadas para processar os videos
    cv.destroyAllWindows()#Fechando todas as janelas auxiliares
    
    captured_frames = np.array(captured_frames)#Retornando um numpy array
    label_data = np.array(label_data)
    return (captured_frames,label_data) 

#cv.imshow("cropped", crop_img)
#cv.waitKey()
#cv.destroyAllWindows()    
    
    
    
    
    