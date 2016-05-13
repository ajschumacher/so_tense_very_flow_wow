#$ python --version
## Python 2.7.11 :: Anaconda 4.0.0 (x86_64)

# Extra steps because it was being finicky.
#$ pip uninstall protobuf
#$ pip install --upgrade protobuf==3.0.0b2
# As directed at https://www.tensorflow.org/versions/r0.8/get_started/os_setup.html
#$ pip install --upgrade https://storage.googleapis.com/tensorflow/mac/tensorflow-0.8.0-py2-none-any.whl

import tensorflow as tf

print tf.__version__
## 0.8.0


# Let's go through all the tutorials!

# First:
# https://www.tensorflow.org/versions/r0.8/tutorials/mnist/beginners/index.html

# Lines from tutorial:
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Whoa! It downloads and unpacks some gzip files.

type(mnist)
## Out[8]: tensorflow.contrib.learn.python.learn.datasets.mnist.DataSets

# Wow!

dir(mnist)
## ['__class__',
##  '__delattr__',
##  '__dict__',
##  '__doc__',
##  '__format__',
##  '__getattribute__',
##  '__hash__',
##  '__init__',
##  '__module__',
##  '__new__',
##  '__reduce__',
##  '__reduce_ex__',
##  '__repr__',
##  '__setattr__',
##  '__sizeof__',
##  '__str__',
##  '__subclasshook__',
##  '__weakref__',
##  'test',
##  'train',
##  'validation']

type(mnist.train)
## tensorflow.contrib.learn.python.learn.datasets.mnist.DataSet

type(mnist.test)
## tensorflow.contrib.learn.python.learn.datasets.mnist.DataSet

type(mnist.validation)
## tensorflow.contrib.learn.python.learn.datasets.mnist.DataSet

dir(mnist.validation)
## ['__class__',
##  '__delattr__',
##  '__dict__',
##  '__doc__',
##  '__format__',
##  '__getattribute__',
##  '__hash__',
##  '__init__',
##  '__module__',
##  '__new__',
##  '__reduce__',
##  '__reduce_ex__',
##  '__repr__',
##  '__setattr__',
##  '__sizeof__',
##  '__str__',
##  '__subclasshook__',
##  '__weakref__',
##  '_epochs_completed',
##  '_images',
##  '_index_in_epoch',
##  '_labels',
##  '_num_examples',
##  'epochs_completed',
##  'images',
##  'labels',
##  'next_batch',
##  'num_examples']

type(mnist.train.labels)
## numpy.ndarray

mnist.train.labels[:3]
## array([[ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.],
##        [ 0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.],
##        [ 0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.]])

type(mnist.train.epochs_completed)
## int

mnist.train.epochs_completed
## 0

mnist.train.num_examples
## 55000

mnist.train.images.shape
## (55000, 784)

mnist.test.images.shape
## (10000, 784)

mnist.validation.images.shape
## (5000, 784)

help(mnist.validation.next_batch)
## next_batch(self, batch_size, fake_data=False)
##     Return the next `batch_size` examples from this data set.

mnist.validation._index_in_epoch
## 0

x = mnist.validation.next_batch(10)
# Now x is a tuple ([10 examples], [10 one-hot labels])
# should use a number of examples different from
# the number of classes....

mnist.validation._index_in_epoch
## 10

# Okay! So this tensorflow.contrib.learn.python.learn.datasets.mnist.DataSet
# just lets us keep grabbing data; nice.


# tutorial has this now:
import tensorflow as tf

# tutorial
x = tf.placeholder(tf.float32, [None, 784])

type(x)
## tensorflow.python.framework.ops.Tensor
help(tf.placeholder)
## placeholder(dtype, shape=None, name=None)
##     Inserts a placeholder for a tensor that will be always fed.
##
##     **Important**: This tensor will produce an error if evaluated. Its value must
##     be fed using the `feed_dict` optional argument to `Session.run()`,
##     `Tensor.eval()`, or `Operation.run()`.
##
##     For example:
##
##     ```python
##     x = tf.placeholder(tf.float32, shape=(1024, 1024))
##     y = tf.matmul(x, x)
##
##     with tf.Session() as sess:
##       print(sess.run(y))  # ERROR: will fail because x was not fed.
##
##       rand_array = np.random.rand(1024, 1024)
##       print(sess.run(y, feed_dict={x: rand_array}))  # Will succeed.
##     ```
##
##     Args:
##       dtype: The type of elements in the tensor to be fed.
##       shape: The shape of the tensor to be fed (optional). If the shape is not
##         specified, you can feed a tensor of any shape.
##       name: A name for the operation (optional).
##
##     Returns:
##       A `Tensor` that may be used as a handle for feeding a value, but not
##       evaluated directly.

# tutorial
W = tf.Variable(tf.zeros([784, 10]))

help(tf.Variable)
# crazy long!

# tutorial
b = tf.Variable(tf.zeros([10]))

# tutorial
y = tf.nn.softmax(tf.matmul(x, W) + b)

help(tf.nn.softmax)
## softmax(logits, name=None)
##     Computes softmax activations.
##
##     For each batch `i` and class `j` we have
##
##         softmax[i, j] = exp(logits[i, j]) / sum(exp(logits[i]))
##
##     Args:
##       logits: A `Tensor`. Must be one of the following types: `float32`, `float64`.
##         2-D with shape `[batch_size, num_classes]`.
##       name: A name for the operation (optional).
##
##     Returns:
##       A `Tensor`. Has the same type as `logits`. Same shape as `logits`.

b.get_shape()
## TensorShape([Dimension(10)])
# So... column vector?
# But it behaves like a row vector just fine, adding to [n, 10] things...
# Okay whatevs
# .get_shape() is cool tho

import time

# tutorial
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

start = time.time()
# tutorial
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

print time.time() - start
## 4.34470200539

# tutorial
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
## 0.9185

# Huh! Have to totally sess.run everything. Wild! So DAG.
