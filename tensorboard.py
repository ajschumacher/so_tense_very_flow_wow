# let's see if we can make some stuff and look at it with tensorboard

import tensorflow as tf

w = tf.Variable(2.0)
b = tf.Variable(0.9)
x = tf.constant(1.0)
y_ = tf.constant(0.0)
y = tf.sigmoid(w*x + b)
train_step = tf.train.GradientDescentOptimizer(0.15).minimize((y - y_)**2)

sess = tf.Session()
sess.run(tf.initialize_all_variables())
for i in range(300):
    sess.run(train_step)
    print i, sess.run(y)
