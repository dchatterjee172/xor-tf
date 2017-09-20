import tensorflow as tf
import numpy as np
x=np.array([[0,0],[0,1],[1,0],[1,1]])
y=np.array([[0],[1],[1],[0]])
inp=tf.placeholder(dtype=tf.float32,shape=(1,2))
out=tf.placeholder(dtype=tf.float32,shape=(1,1))
w1=tf.get_variable("w1",[2,1])
w2=tf.get_variable("w2",[2,1])
b1=tf.get_variable("b1",[1,1]) 
b2=tf.get_variable("b2",[1,1])
w3=tf.get_variable("w3",[2,1]) 
y1=tf.tanh(tf.matmul(inp,w1)+b1)
y2=tf.tanh(tf.matmul(inp,w2)+b2)
inp3=tf.transpose(tf.concat([y1,y2],0))
y3=tf.sigmoid(tf.matmul(inp,w3))
loss=tf.square(out-y3)/2.0
mini=tf.train.GradientDescentOptimizer(0.001).minimize(loss)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    #writer = tf.summary.FileWriter("~/tfg", sess.graph)
    for i in range(0,500):
        for j in range(0,4):
            sess.run(mini,feed_dict={inp:x[j].reshape(1,2),out:y[j].reshape(1,1)})
            print(sess.run(loss,feed_dict={inp:x[j].reshape(1,2),out:y[j].reshape(1,1)})) 
