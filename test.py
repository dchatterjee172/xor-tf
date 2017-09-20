import tensorflow as tf
import numpy as np
import random as rn
x=np.array([[0,0],[0,1],[1,0],[1,1]])
y=np.array([[0],[1],[1],[0]])
inp=tf.placeholder(dtype=tf.float32,shape=(1,2))
out=tf.placeholder(dtype=tf.float32,shape=(1,1))
w1=tf.get_variable("w1",initializer=tf.random_normal(shape=(2,1),mean=0,stddev=1))
w2=tf.get_variable("w2",initializer=tf.random_normal(shape=(2,1),mean=0,stddev=1))
b1=tf.get_variable("b1",initializer=tf.random_normal(shape=(1,1),mean=0,stddev=1)) 
b2=tf.get_variable("b2",initializer=tf.random_normal(shape=(1,1),mean=0,stddev=1))
w3=tf.get_variable("w3",initializer=tf.random_normal(shape=(2,1),mean=0,stddev=1)) 
y1=tf.nn.tanh(tf.matmul(inp,w1)+b1)
y2=tf.nn.tanh(tf.matmul(inp,w2)+b2)
inp3=tf.transpose(tf.concat([y1,y2],0))
y3=tf.nn.tanh(tf.matmul(inp3,w3))
loss=tf.pow(out-y3,2)
mini=tf.train.GradientDescentOptimizer(0.02).minimize(loss)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter("tfg", sess.graph)
    for i in range(0,1000):
        j=rn.randint(0,3)
        for k in range(0,25):
            sess.run(mini,feed_dict={inp:x[j].reshape(1,2),out:y[j].reshape(1,1)})
        r=sess.run([loss,y3,out],feed_dict={inp:x[j].reshape(1,2),out:y[j].reshape(1,1)})
        print("loss: "+str(r[0][0]))
        print("expected: "+str(r[2][0])+"  output: "+str(r[1][0]))
