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
import sklearn#Lib para misturar as paradinhas
import tensorflow as tf
import keras
import random   #num1 = random.randint(0, 9) para randomizar entre 0 e 9

def capturing_frames(diretorio,div, shape = (500,500), DEBUG = True):
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
            if DEBUG:
                cv.imshow("preview", frame)
                cv.waitKey(80)#O tempo em ms para passar para o proximo frame
    
    print('\nVideo contém '+str(i-1)+' frames.\n')
    cap.release()#Liberando as memorias utilizadas para processar os videos
    cv.destroyAllWindows()#Fechando todas as janelas auxiliares
    
    captured_frames = np.array(captured_frames)#Retornando um numpy array
    
    return captured_frames

#teste = capturing_frames('/home/salomao/Desktop/Lapizeira4.mp4',25, shape=  (500,500))
#Save_data('/home/salomao/Desktop/Lapizeira4', teste)


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

def spliting_image_labelling(img, shape = (500,500), dist=(3,3)):
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
            frame_split,img_t = spliting_image_labelling(frame, dist=dist, shape=shape)
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
    
    
    
def spliting_image(img, shape = (500,500), dist=(3,3)):
    comp_x = np.shape(img)[0]/dist[0]#Pegar o len do eixo x da imagem e dividir na dist x
    comp_y = np.shape(img)[1]/dist[1]#Pegar o len do eixo y da imagem e dividir na dist y
    comp_x = int(comp_x)#pixels sao integers
    comp_y = int(comp_y)#pixels sao integers 
    splited = []
    for x in range(dist[0]):
        for y in range(dist[1]):
            lim_inf_x = comp_x*x
            lim_sup_x = comp_x*(x+1)
            lim_inf_y = comp_y*y
            lim_sup_y = comp_y*(y+1)
            frame = cv.resize(img[lim_inf_x:lim_sup_x,lim_inf_y:lim_sup_y],shape)
            splited.append(frame)
    splited = np.array(splited)
    return splited
        
    

def spliting_image_labelling_mult_choices(img, shape = (500,500), dist=(3,3)):
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
            print('digite y (minusculo) para 100%, 0, para 0%, 1 para 10%, 2 para 20%, 3 para 30%, 4 para 40%, 5 para 50%, 6 para 60%, 7 para 70%, 8 para 80%, 9 para 90%\n')
            cv.imshow("cropped", frame)
            ei = cv.waitKey()
            cv.destroyAllWindows()
            if ei==121:
                label.append(1)    
            else:
                if ei==49:
                    label.append(0.1)
                elif ei==50:
                    label.append(0.2)
                elif ei==51:
                    label.append(0.3)
                elif ei==52:
                    label.append(0.4)
                elif ei==53:
                    label.append(0.5)
                elif ei==54:
                    label.append(0.6)
                elif ei==55:
                    label.append(0.7)
                elif ei==56:
                    label.append(0.8)
                elif ei==57:
                    label.append(0.9)
                else:
                    label.append(0)

            splited.append(frame)
    return splited,label

def capturing_frames_splits_mult_choices(diretorio,div,shape = (500,500), dist=(3,3)):
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
            frame_split,img_t = spliting_image_labelling_mult_choices(frame, dist=dist, shape=shape)
            label_data = label_data + (img_t  )
            captured_frames = captured_frames + (frame_split)
    print('\nVideo contém '+str(i-1)+' frames.\n')
    cap.release()#Liberando as memorias utilizadas para processar os videos
    cv.destroyAllWindows()#Fechando todas as janelas auxiliares
    
    captured_frames = np.array(captured_frames)#Retornando um numpy array
    label_data = np.array(label_data)
    return (captured_frames,label_data) 

#x,y = capturing_frames_splits_mult_choices('/home/salomao/Desktop/Lapizeira1.mp4',10,shape = (500,500), dist=(3,3))
#Save_data('/home/salomao/Desktop/Lapizeira1_x', x)
#Save_data('/home/salomao/Desktop/Lapizeira1_y', y)
#



def capturing_frames_appended(array_dir, shape = (500,500), DEBUG = True): #array_dir receberá um array de tuplas onde o primeiro elemento da tupla é o diretorio e o segundo, o div
    captured_frames = capturing_frames(array_dir[0][0],array_dir[0][1], shape = shape, DEBUG = DEBUG)
    for ind in array_dir:
            if ind == array_dir[0]:
                continue
            captured_frames = np.append(captured_frames,capturing_frames(ind[0],ind[1], shape = shape, DEBUG = DEBUG),axis=0)
        
    return captured_frames

#frame_teste = capturing_frames_appended([('/home/salomao/Desktop/Objeto1.mp4',10),('/home/salomao/Desktop/Objeto2.mp4',1)],DEBUG=0)
    


def shufle_balance(train_1,train_0): #Esta funcao recebe os datasets de label 1 e label 0, mistura eles separadamente e entrega um dataset balanceado
    if (len(train_1)>len(train_0)):#Se o label 1 for maior que o label 0
        train_1 = sklearn.utils.shuffle(train_1)#misturo o data de label 1
        train_data = train_1[0:len(train_0)]#Pego a mesma quantidade do label zero e coloco em uma nova variavel
        label_1 = np.ones(len(train_data))#Crio um vetor dizendo que tem um certa quantidade de label 1, o mesmo tamanho do data 
        label_0 = np.zeros(len(train_0))#Crio um vetor dizendo que tem um certa quantidade de label 0, o mesmo tamanho do data 
        label_array_train = np.append(label_1,label_0,axis=0)#Somo os vetores de label 1 e depois zero
        train_data = np.append(train_data,train_0,axis=0)#Somo os vetores de data 1 e depois zero
        train_data,label_array_train = sklearn.utils.shuffle(train_data,label_array_train)#Misturo o data juntamente com o label, sem deslinkar a posicao
    else:#Se o label 0 for maior que o label 1
        train_0 = sklearn.utils.shuffle(train_0)#misturo o data de label 0
        train_data = train_0[0:len(train_1)]#Pego a mesma quantidade do label um e coloco em uma nova variavel
        label_1 = np.ones(len(train_1))#Crio um vetor dizendo que tem um certa quantidade de label 1, o mesmo tamanho do data 
        label_0 = np.zeros(len(train_data))#Crio um vetor dizendo que tem um certa quantidade de label 0, o mesmo tamanho do data 
        label_array_train = np.append(label_1,label_0,axis=0)#Somo os vetores de label 1 e depois zero
        train_data = np.append(train_1,train_data,axis=0)#Somo os vetores de data 1 e depois zero
        train_data,label_array_train = sklearn.utils.shuffle(train_data,label_array_train)#Misturo o data juntamente com o label, sem deslinkar a posicao   
    return (train_data,label_array_train)


#train_1 = capturing_frames_appended([('/home/salomao/Desktop/Object_background_contant.mp4', 1),('/home/salomao/Desktop/Objeto2.mp4', 10)],DEBUG=0)
#train_0 = capturing_frames_appended([('/home/salomao/Desktop/Ambient_background_contant.mp4', 1),('/home/salomao/Desktop/Ambient1.mp4', 10),('/home/salomao/Desktop/Ambient2.mp4', 10)],DEBUG=0)
#train,label = shufle_balance(train_1,train_0)



def data_augment_balance_shufle(train_1,train_0, preferences): #Esta funcao recebe os datasets de label 1 e label 0, mistura eles separadamente e entrega um dataset balanceado; depois disso ele pega e augmenta os dados de acordo com a preferencia! preferencia  é uma lista de tuplas que contem strings e pesos ... se a string bater, ele augmenta daquela forma e cria um proporcional com o total baseado no seu peso 
    if (len(train_1)>len(train_0)):#Se o label 1 for maior que o label 0
        train_1 = sklearn.utils.shuffle(train_1)#misturo o data de label 1
        train_data = train_1[0:len(train_0)]#Pego a mesma quantidade do label zero e coloco em uma nova variavel
        label_1 = np.ones(len(train_data))#Crio um vetor dizendo que tem um certa quantidade de label 1, o mesmo tamanho do data 
        label_0 = np.zeros(len(train_0))#Crio um vetor dizendo que tem um certa quantidade de label 0, o mesmo tamanho do data 
        label_array_train = np.append(label_1,label_0,axis=0)#Somo os vetores de label 1 e depois zero
        train_data = np.append(train_data,train_0,axis=0)#Somo os vetores de data 1 e depois zero
        train_data,label_array_train = sklearn.utils.shuffle(train_data,label_array_train)#Misturo o data juntamente com o label, sem deslinkar a posicao
    else:#Se o label 0 for maior que o label 1
        train_0 = sklearn.utils.shuffle(train_0)#misturo o data de label 0
        train_data = train_0[0:len(train_1)]#Pego a mesma quantidade do label um e coloco em uma nova variavel
        label_1 = np.ones(len(train_1))#Crio um vetor dizendo que tem um certa quantidade de label 1, o mesmo tamanho do data 
        label_0 = np.zeros(len(train_data))#Crio um vetor dizendo que tem um certa quantidade de label 0, o mesmo tamanho do data 
        label_array_train = np.append(label_1,label_0,axis=0)#Somo os vetores de label 1 e depois zero
        train_data = np.append(train_1,train_data,axis=0)#Somo os vetores de data 1 e depois zero
        train_data,label_array_train = sklearn.utils.shuffle(train_data,label_array_train)#Misturo o data juntamente com o label, sem deslinkar a posicao   
           
    return (train_data,label_array_train)