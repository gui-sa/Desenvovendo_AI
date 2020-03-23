#Esta tentativa vem com objetivo de fazer minha propria rede neural, isto, seguindo ideias de treinamento contínuo imagem por imagem proposto pelo dia 12/03/2020
#/home/salomao/Desktop/Detector_images/Pardal_Treinamento
import os#biblioteca de açoes de sistema básica

ask = 'a'#preset de variavel, so para nao ocorrer coincidencias desagradaveis
while True:
    ask = input('/nDeseja criar uma nova rede neural?(y = yes / n = no / q = quit):       ')#Enquanto a pessoa nao digita y ou n, ele continua pergutando
    if ask=='y':
        while True:
            train_path = input('\nQual o path para a pasta de treinamento?  ')
            if os.path.exists(train_path):#Checa se o path existe
                
                ##Mexendo com os paths
                
                nome_da_rede = 'CNN_parametros_' + train_path.split('/')[-1]
                lista_class = os.listdir(train_path)#Capturo as outras pastas dentro do path
                train_path_class_one = os.path.join(train_path,lista_class[0])#Crio novos paths, agora para cada classe
                train_path_class_two = os.path.join(train_path,lista_class[1])
                break#saio do loop que pergunta a pasta de treinamento
            
        ##Criando um arquivo no path desejado, com o nome desejado
        
        file_parametros_path = os.path.join(train_path,'..')+ '/' + nome_da_rede# Diretorio anterior a  pasta de treinamento
        file_parametros = open(file_parametros_path,'w')#Vai se chamar CNN_parametros
        
        ##Coletando parametros de rede e salvando no arquivo de forma ordenada:
        
        batch_size = input('\nQual o tamanho do batch? ' )
        Width = input('\nQual o comprimento da imagem que entrara na rede neural?  ')
        Height = input('\nQual a altura da imagem que entrar� na rede neural? ')
        Epochs = input('\nQuantas epocas? ' )
        
        file_parametros.write(int(batch_size),'\t',int(Width),'\t',int(Height),'\t',int(Epochs))
        
        file_parametros.close()#FEchando o arquivo.
        
        
    if ask=='q':#Se a pessoa digitar q, ela sai do programa sem executar nada.
        break
        
        
        
    