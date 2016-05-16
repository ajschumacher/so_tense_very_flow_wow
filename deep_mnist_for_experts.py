# this one:
# https://www.tensorflow.org/versions/r0.8/tutorials/mnist/pros/index.html
# ooh "pros"

# same loading stuff
# tutorial
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# this is the "pro" explanation I guess
#> Tensorflow relies on a highly efficient C++ backend to do its computation. The connection to this backend is called a session. The common usage for TensorFlow programs is to first create a graph and then launch it in a session.

# huh! the "pro" version uses `InteractiveSession`?
# seems like a good thing for beginners; meh?

# tutorial
import tensorflow as tf
sess = tf.InteractiveSession()

# huh! the "computational graph" stuff is just the same as in the beginner version

# same as beginner...
# tutorial
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# slightly less verbose than beginner version
# tutorial
sess.run(tf.initialize_all_variables())

# tutorial
y = tf.nn.softmax(tf.matmul(x,W) + b)
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

#> Because TensorFlow knows the entire computation graph, it can use automatic differentiation to find the gradients of the cost with respect to each of the variables. TensorFlow has a variety of builtin optimization algorithms. For this example, we will use steepest gradient descent, with a step length of 0.5, to descend the cross entropy.

# ooh! from https://www.tensorflow.org/versions/r0.8/api_docs/python/train.html#optimizers, you can:
#> Compute the gradients with compute_gradients().
# if you want to see them.
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
for i in range(1000):
   batch = mnist.train.next_batch(50)
   train_step.run(feed_dict={x: batch[0], y_: batch[1]})
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
## 0.9092
# huh; not 92%. but it does say both 92 and 91%... meh

#> Since we're using ReLU neurons, it is also good practice to initialize them with a slightly positive initial bias to avoid "dead neurons."

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# https://www.tensorflow.org/versions/r0.8/api_docs/python/constant_op.html#truncated_normal
# +/- 2sd max

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')
# [batch, in_height, in_width, in_channels]
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
x_image = tf.reshape(x, [-1, 28, 28, 1])
# https://www.tensorflow.org/versions/r0.8/api_docs/python/array_ops.html#reshape
#> If one component of shape is the special value -1, the size of that dimension is computed so that the total size remains constant. In particular, a shape of [-1] flattens into 1-D. At most one component of shape can be -1.
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

#> TensorFlow's tf.nn.dropout op automatically handles scaling neuron outputs in addition to masking them, so dropout just works without any additional scaling.
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# adding
import time
start = time.time()

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.initialize_all_variables())
for i in range(20000):
    batch = mnist.train.next_batch(50)
    if i%100 == 0:
        train_accuracy = accuracy.eval(feed_dict={
            x:batch[0], y_: batch[1], keep_prob: 1.0})
        print("step %d, training accuracy %g"%(i, train_accuracy))
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

print("test accuracy %g"%accuracy.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

print start - time.time()

# huh - that's it, I guess?
# okay, so with the interactive session I guess the nice thing is
# you get to do thing.run() instead of sess.run(thing)...
# is that even different?
# looks like it is different;
# tf.InteractiveSession()
# populates a default session; just
# tf.Session()
# and you get:
# ValueError: Cannot execute operation using Run(): No default session is registered. Use 'with default_session(sess)' or pass an explicit session to Run(session=sess)

# this thing takes forever to run
# by `top` / `htop` it looks like it uses all my cores though, which is nice!

# hey, it's done!
## -7755.13304281
# ugh wrong order
# about 129 minutes. two hours!
