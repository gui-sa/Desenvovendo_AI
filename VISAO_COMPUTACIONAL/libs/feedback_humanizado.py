# Descricao da ideia: a ideia deste codigo é criar funçoes usando opencv e numpy para tornar a validação de um modelo HUMANIZADO!
#COMO?
#O modelo sera treinado.
#Quando acabar o treinamento o usuario faz a chamada da funçao presente neste arquivo e ela loada um video!
#O video mostrara uma tela com 9 quadros (10 - contando o quadro de origem). Desta maneira será possível validar de maneira humana
##se o dataset realmente aprendeu o objeto, ou se aprendeu o background.
#Como o modelo estara na mesma tela, fica facil fazer um treinamento mais orientado, desconstruindo noçoes de aprendizados errados..
#Esta funçao vem pra fazer combo como novo metodo de ensino equilibrado e com zero do infinito.

import cv2 as cv
from Desktop.Desenvovendo_AI.VISAO_COMPUTACIONAL.libs import extracting_data_from_video
import tensorflow as tf
import keras
import pickle
import numpy as np

def feedback_splitted_by_video( model , video_cap, limiar):
    cap = cv.VideoCapture(video_cap)#Settando diretorio para o flow de video
    while(cap.isOpened()):
        ret, frame = cap.read()#ret avalia se o video terminou ou não, retornando um booleano. frame é a imagem
        if ret == False:#Quando o video acaba o ret é falso e a janela quebra
            break
        frame_split= extracting_data_from_video.spliting_image(frame)
        frame = cv.resize(frame, (500,500), interpolation = cv.INTER_AREA)
        pred = model.predict(frame_split)#faço a prediçao com base no modelo 
        for j in range(3):
            for i in range(3):
                color = (0, 0, 255)
                if pred[j*3+i]>limiar:
                    color = (0, 255, 0)
                frame = cv.putText(
                        frame, 
                        str(pred[j*3+i]), 
                        (200*i,50+200*j), 
                        cv.FONT_HERSHEY_SIMPLEX, 
                        0.5,  
                        color, 
                        2, 
                        cv.LINE_AA, 
                        False) 
        cv.imshow("feed_back",frame)
        ei = cv.waitKey(1)
        if ei==121:
            break
    cv.destroyAllWindows()
        





