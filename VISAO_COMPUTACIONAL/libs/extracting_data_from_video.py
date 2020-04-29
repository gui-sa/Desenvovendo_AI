#Todos sabem que para treinar uma rede neural convolucional com boa acuracia é necessário um grande dataset.
#Outro ponto, é importante que a rede aprenda o objeto de diferentes angulos e backgrounds.
#Para tal, nada melhor do que gravar videos circulando o objeto desejado, e pertubando de maneiras distintas....
#Um vídeo são muitas fotos agrupadas... Esta a ideia!
#Criamos um código que capta vídeos e cria um dataset para cada vídeo.
#Depois pegamos os datasets dos videos, juntamos e misturamos tudo!
#Pronto! temos um dataset com o minimo de dado criado possível.

import cv2 as cv
import numpy as np


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

teste = capturing_frames('/home/salomao/Desktop/Teste.mp4',1, shape=  (200,200))
