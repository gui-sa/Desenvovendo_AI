#Esta tentativa vem com objetivo de fazer minha propria rede neural, isto, seguindo ideias de treinamento contínuo imagem por imagem proposto pelo dia 12/03/2020

import os#biblioteca de açoes de sistema básica
import shutil#biblioteca de açoes de sistema avançada

ask = 'a'#preset de variavel, so para nao ocorrer coincidencias desagradaveis
while True:
    ask = input('/nDeseja criar uma nova rede neural?(y = yes / n = no / q = quit):       ')#Enquanto a pessoa nao digita y ou n, ele continua pergutando
    if ask=='y':
        while True:
            train_path = input('\nQual o path para a pasta de treinamento?\n')
            if os.path.exists(train_path):#Checa se o path existe
                lista_class = os.listdir(train_path)#Capturo as outras pastas dentro do path
                train_path_class_one = os.path.join(train_path,lista_class[0])#Crio novos paths, agora para cada classe
                train_path_class_two = os.path.join(train_path,lista_class[1])
                break#saio do loop que pergunta a pasta de treinamento
        file_parametros_path = os.path.join(train_path,'..')# Diretorio anterior à pasta de treinamento
        file_parametros = open('CNN_parametros','w')#Vai se chamar CNN_parametros
        shutil.move('CNN_parametros',file_parametros_path)#Vou salvar os parametros das redes no diretorio anterior à pasta de treinamento
        file_parametros.close()
        
        
    if ask=='q':#Se a pessoa digitar q, ela sai do programa sem executar nada.
        break
        
        
        
    