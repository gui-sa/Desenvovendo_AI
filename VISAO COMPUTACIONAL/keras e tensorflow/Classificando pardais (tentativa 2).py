#Esta tentativa vem com objetivo de fazer minha propria rede neural, isto, seguindo ideias de treinamento contínuo imagem por imagem proposto pelo dia 12/03/2020
#/home/salomao/Desktop/Detector_images/Pardal_Treinamento


import os#biblioteca de acoes de sistema basicas
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model


    
    
    
    
##Funcao que le os parametros do arquivo e treina a rede, salvando-a ================================================================================
def treinamento_cnn (path_parametro,path_train, nome, num_dados):
    
    file = open(path_parametro,'r')#Vou ler o arquivo dos parametros e captar os dados que foram salvos
    
    linhas  = file.read().split('\n')#Todos os dados de um arquivo sao strings. Desta forma quando quebramos por \n, obtemos as linhas
    parametros = linhas[0].split('\t')#Separei os elementos por \t, desta forma, na linha desejada, captarei de forma organizada os dados desejados
    batch_size = parametros[0]
    IMG_WIDTH = parametros[1]
    IMG_HEIGHT = parametros[2]
    epoch = parametros[3]
    file.close()#Fechar arquivo de leitura
    
    ## Preprocessamento  das imagens:
    
    train_image_generator = ImageDataGenerator(rescale = 1./255) # reescala os valores em float de 0 - 1
    train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,  #Do objeto train_image_generator, criar um fluxo de  matrizes de batch tamanho batch size
                                                           directory=path_train, #No diretorio: 
                                                           shuffle=True, #devo randomizar?sim!
                                                           target_size=(IMG_HEIGHT, IMG_WIDTH),#Imagens de tamanho...
                                                           class_mode='categorical')
    
    ##Arquitetura da minha rede CNN:
    
    
    model = Sequential([
    Conv2D(16, 3, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH ,3)),
    MaxPooling2D(),
    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(2, activation='sigmoid')
    ])
    
    
    model.compile(optimizer='adam',#ESTA FUNCAO CONFIGURA O MODELO (OBJETO) para um possivel treinamento
              loss='binary_crossentropy',
              metrics=['accuracy'])


    history = model.fit_generator(#Esta funcao treina sua rede neural.
    train_data_gen,
    steps_per_epoch=num_dados,
    epochs=epoch )


    model.save(nome)
    
#====================================================================================================================================================








#Inicio da funcao main:==============================================================================================================================
    
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
        Width = input('\nQual o comprimento da imagem que entrar na rede neural?  ')
        Height = input('\nQual a altura da imagem que entrar na rede neural? ')
        Epochs = input('\nQuantas epocas? ' )
        file_parametros.write( batch_size + '\t' + Width + '\t' + Height + '\t' + Epochs )
        
        file_parametros.close()#Fechando o arquivo.
        
        passos_epoch = len(os.listdir(train_path_class_one))+ len(os.listdir(train_path_class_two))#A soma do numero de imagens em cada diretorio
        treinamento_cnn( file_parametros_path , train_path , nome_da_rede, passos_epoch )
        
        
    if ask=='q':#Se a pessoa digitar q, ela sai do programa sem executar nada.
        break
        
        
###==================================================================================================================================================
    

    
    

    