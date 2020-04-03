import tensorflow as tf#tensorflow
tf.compat.v1.disable_eager_execution()#TensorFlow's eager execution is an imperative programming environment that evaluates operations immediately, without building graphs
import math#Para mexer com funcoes avancadas
import numpy as np#Para mexer com vetores
import sklearn as sk
import matplotlib.pyplot as plt


(input_train,target_train) ,(input_val,target_val) = tf.keras.datasets.mnist.load_data()#Loadando minist dataset #O database separa 60000 para treinamento e 10000 para validacao Separamos o database de treinamento entre o input e o target Separamos o database da validacao entre input e target
target_train = tf.one_hot(target_train,10)#Hotenconding o target data do treinamento
target_val = tf.one_hot(target_val,10)#Hotenconding o target data da validacao

#A arquitetura desejada é:
#Convolutional Layer 1
filter_size1=5
num_filters1 = 16

#Convolutional Layer 2
filter_size2=5
num_filters2 = 36

#Fully connected layer
fc_size = 128


def new_weights(shape):
    return tf.compat.v1.Variable(tf.compat.v1.random_normal(shape,stddev = 0.05))#Este shape representa o formato da matriz peso


def new_bias(length):
    return tf.compat.v1.Variable(tf.compat.v1.random_normal(shape = [length],stddev = 0.05))#Este shape representa o formato da matriz bias


def conv_layer(x,#A layer anterior
               num_input_channels,#Numero de canais da layer anterior
               filter_size,#Tamanho dos filtros (Width e Height)
               num_filter):#Numero de filtros
    shape = [filter_size,filter_size,num_input_channels,num_filter]#Este formato segue as recomendacoes da funcao tf.nn.conv2d
    
    weights = new_weights(shape=shape)#Criamos uma matriz de variaveis no formato do filtro desejado 
    
    biases = new_bias(length=num_filter)
    
    layer = tf.nn.conv2d(input = x,#O input deve ter formato [numero_de_imagens,altura_da_imagem,comprimento_da_imagem,numero_de_canais]
                       filters = weights,#Nao existem filtros prontos!!, Os filtros sao variaveis pesos que serão otimizadas de acordo com a imagem! São filtros personalizados para a imagem! São pesos compartilhados! Por isso a CNN possui pouco carater espacial!
                       strides = [1,1,1,1],#O stride é como o filtro se locomove por estes diferentes shape do input
                       padding = 'SAME')#mantendo como resultado o mesmo shape, ou seja, a borda entrara na jogada
    
    layer = layer + biases#Somando bias
    
    layer = tf.nn.max_pool( input= layer,
                           ksize = [1,2,2,1],#A matriz do maxpool percorre apenas 1 imagem, de 2 em 2 pixels em comprimento e altura e em apenas 1 canal
                           strides = [1,2,2,1],#O stride precisa ser dois, para efetuar o comportamento padrao do maxpool
                           padding = 'SAME')#Consideramos as bordas na jogada
    
    layer = tf.nn.relu(layer)#Aplicando a funcao relu
    return layer, weights

def flatten_layer(layer):
    layer_shape = layer.get_shape()#Aqui obteremos uma matriz do formato [num_img,imh_h,img_w,num_ch]
    num_features = layer_shape[1:4].num_elements()#Retorna a multiplicação de img_h, img_w e num_ch
    layer_flat = tf.reshape(layer,[-1,num_features])#Criamos um vetor linha com todos os features 
    return layer_flat,num_features
    

def new_fc_layer(x,          # The previous layer.
                 num_inputs,     # Num. inputs from prev. layer.
                 num_outputs,    # Num. outputs.
                 use_relu=True): # Use Rectified Linear Unit (ReLU)?

    weights = new_weights(shape=[num_inputs, num_outputs])# Criando variaveis de peso compativel com a operacao necessaria de rede neural
    biases = new_bias(length=num_outputs)#O bias é criado para a saida, pois nao faz logica cria-lo para a entrada.

    layer = tf.matmul(x, weights) + biases#Operação de rede neural fully connected padrao

    if use_relu:
        layer = tf.nn.relu(layer)#Aplicar na saida a operacao relu

    return layer


x = tf.compat.v1.placeholder(tf.float32,shape = (None,28,28),name='Input')#Primeiramente recebemos os dados do mnist (que estao NO DATABASE, neste formato)
x_conv = tf.reshape(x,[-1,28,28,1])#Conv layer precisa do formato [num_images,img_h,img_w,n_channel] Nossa figura tem 28x28x1 preto/branco

y_true = tf.compat.v1.placeholder(tf.float32,shape = (None,10),name='target')#Um place holder para guardar o target One hot encnde para 10 classes
y_true_class = tf.argmax(y_true)

#========================================================== Criando Modelo =============================================================================

#Conv 1
layer_conv1 , weights_conv1 = conv_layer(x=x_conv,#A entrada da imagem no formato da conv
                                         num_input_channels=1,#Escala cinza
                                         filter_size=filter_size1,#Filtro de tamanho de kernel planejado
                                         num_filter=num_filters1)#Numero de filtros

#Conv 2
layer_conv2 , weights_conv2 = conv_layer(x=layer_conv1,#A entrada da imagem no formato da conv
                                         num_input_channels=num_filters1,#Escala cinza
                                         filter_size=filter_size2,#Filtro de tamanho de kernel planejado
                                         num_filter=num_filters2)#Numero de filtros
#Flatten layer:

layer_flat, num_features = flatten_layer(layer_conv2)

#Fully cnnected layer

layer_fc1 = new_fc_layer(x = layer_flat,
                         num_inputs= num_features,
                         num_outputs= fc_size,
                         use_relu= True)




layer_fc2 = new_fc_layer(x = layer_fc1,
                         num_inputs= fc_size,
                         num_outputs= 10,
                         use_relu= False)


y_pred = tf.nn.softmax(layer_fc2)#Ao inves de aplicar Relu aplicamos softmax
y_pred_class = tf.argmax(y_pred)#Classe preditada

crossentropy = tf.nn.softmax_cross_entropy_with_logits(logits = layer_fc2, labels = y_true)#Esta funcao executa o softmax e ja calcula o erro cross entropy das categorias.

cost = tf.reduce_mean(crossentropy)#Vou otimizar a media do cross entropy

Optmizer = tf.compat.v1.train.AdamOptimizer(learning_rate= 1e-4).minimize(cost)# Falando para o otimizador Adam otimizar o custo à um learning rate de 0.0001

correct_pred = tf.equal(y_pred_class,y_true_class)#Cria um vetor de booleanos dizendo se acertou ou nao

accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))#Quero a media do vetor de booleanos == acuracia 

init = tf.compat.v1.global_variables_initializer()#Para inicializar as variaveis


#============================rodando o codigo \\\\\\\\\\\\\\\\\\\\

with tf.compat.v1.Session() as sess:
    sess.run(init)#variaveis iniciadas!!
    
    train_batch_size =64#tamanho do batch
    epochs = 20
     
    for i in range(epochs):
        x_batch,y_true_batch = 
        
    