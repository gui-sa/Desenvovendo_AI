
import tensorflow as tf
tf.compat.v1.disable_eager_execution()


w   =  tf.Variable([1], name = 'weights', dtype = tf.float32)
b   =  tf.Variable([3], name = 'biass', dtype = tf.float32)
x = tf.compat.v1.placeholder(name = 'input', dtype = tf.float32)
y = b + w*x 

init = tf.compat.v1.global_variables_initializer()#Criando objetos que inicializa todas as variaveis

with tf.compat.v1.Session() as sess:
    sess.run(init)#Inicia todas as variaveis
    tensorboard_file = tf.compat.v1.summary.FileWriter('/home/salomao/Desktop/Curso da udemy em visao computacional/tensorboard', sess.graph)
