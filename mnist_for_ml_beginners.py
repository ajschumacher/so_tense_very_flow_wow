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


# Okay! So...
# Sessions are the only things that "really" have values,
# and they have state,
# while the DAG (defined computations) never do.
# Cool.

# So to summarize the contents of this tutorial
# in a more imperative order:

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

import tensorflow as tf
sess = tf.Session()
sess.run(tf.initialize_all_variables())
# Hmm! Can this be run first? Probably not...
# so the imported `tf` is also managing state?
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, W) + b)
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y),
                                              reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# Yup! Gross! Fails at `sess.run`.
# So in a sense, it was more imperative than I realized at first.
# It's like we have one big global namespace for tf things. YUCK.

sess.run(tf.initialize_all_variables())
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# works fine, as expected

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: mnist.test.images,
                                    y_: mnist.test.labels}))

# colah is so good: http://colah.github.io/posts/2015-09-Visual-Information/
# sure enough, this is good too:
# http://neuralnetworksanddeeplearning.com/chap3.html#softmax


# oh hey, where is that tf namespace?
# looks like there's `tf.variable_scope`, which is a related user-space tool...

# okay so there's `_default_graph_stack`,
# which doesn't have an imported name; see
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/framework/ops.py
# you can do

g = tf.get_default_graph()
g._collections
## {'trainable_variables': [<tensorflow.python.ops.variables.Variable at 0x1175c72d0>,
##   <tensorflow.python.ops.variables.Variable at 0x1090d9210>],
##  'variables': [<tensorflow.python.ops.variables.Variable at 0x1175c72d0>,
##   <tensorflow.python.ops.variables.Variable at 0x1090d9210>],
##  ('__varscope',): [<tensorflow.python.ops.variable_scope.VariableScope at 0x117add850>]}

# Ha! Found you!

Where are constants though?

# other fun thing:
import inspect
for line in inspect.getsourcelines(tf.constant)[0]: print line


# constants!
x = tf.constant(22)
g = x.graph  # (same as tf.get_default_graph())
op = g.get_operations()[-1]  # (same as x.op)
op._node_def
## name: "Const"
## op: "Const"
## attr {
##   key: "dtype"
##   value {
##     type: DT_INT32
##   }
## }
## attr {
##   key: "value"
##   value {
##     tensor {
##       dtype: DT_INT32
##       tensor_shape {
##       }
##       int_val: 22
##     }
##   }
## }


# Okay! this is pretty cool!

# from https://www.tensorflow.org/versions/r0.8/tutorials/mnist/beginners/index.html
# again, a useful quote:
# > To do efficient numerical computing in Python, we typically use libraries like NumPy that do expensive operations such as matrix multiplication outside Python, using highly efficient code implemented in another language. Unfortunately, there can still be a lot of overhead from switching back to Python every operation. This overhead is especially bad if you want to run computations on GPUs or in a distributed manner, where there can be a high cost to transferring data.

# > TensorFlow also does its heavy lifting outside python, but it takes things a step further to avoid this overhead. Instead of running a single expensive operation independently from Python, TensorFlow lets us describe a graph of interacting operations that run entirely outside Python. (Approaches like this can be seen in a few machine learning libraries.)

# (the stuff) above was exploring how that looks in the python module


# cool now let's try to run the toy single-node examples from
# http://neuralnetworksanddeeplearning.com/chap3.html
# in tensorflow!
# ps the author:
# http://michaelnielsen.org/
# https://twitter.com/michael_nielsen

tf.reset_default_graph()  # not necessary, but neat

# Here's our neuron with just one input.
# Initial values as set by Nielsen
w = tf.Variable(0.6)
b = tf.Variable(0.9)
x = tf.constant(1.0)
y_ = tf.constant(0.0)


## In [171]: x.graph.get_operations()[0].name
## Out[171]: u'Variable/initial_value'
##
## In [172]: x.graph.get_operations()[1].name
## Out[172]: u'Variable'
##
## In [173]: x.graph.get_operations()[2].name
## Out[173]: u'Variable/Assign'
##
## In [174]: x.graph.get_operations()[3].name
## Out[174]: u'Variable/read'
##
## In [175]: x.graph.get_operations()[4].name
## Out[175]: u'Variable_1/initial_value'
##
## In [176]: x.graph.get_operations()[5].name
## Out[176]: u'Variable_1'
##
## In [177]: x.graph.get_operations()[6].name
## Out[177]: u'Variable_1/Assign'
##
## In [178]: x.graph.get_operations()[7].name
## Out[178]: u'Variable_1/read'
##
## In [179]: x.graph.get_operations()[8].name
## Out[179]: u'Const'
##
## In [180]: x.graph.get_operations()[9].name
## Out[180]: u'Const_1'

y = tf.mul(w, x)

# make a session...
s = tf.Session()

s.run(y)
## ...
## FailedPreconditionError: Attempting to use uninitialized value Variable
## ...

# gross! variables have to be initialized, okay okay

s.run(tf.initialize_all_variables())
s.run(y)
## 0.60000002

# oh haha that's not right...
y = w*x + b
s.run(y)
## 1.5


## In [216]: x.graph.get_operations()[-1].name
## Out[216]: u'add'
##
## In [217]: x.graph.get_operations()[-2].name
## Out[217]: u'mul'
##
## In [218]: x.graph.get_operations()[-3].name
## Out[218]: u'init'

# cool cool

# ah but it's supposed to be a sigmoid...
y = tf.sigmoid(y)  # don't do that for a demo, but it works
s.run(y)
## 0.81757444

# ooh! I should use tensorboard!
# https://www.tensorflow.org/versions/r0.8/how_tos/summaries_and_tensorboard/index.html

loss = y - y_


# Nielsen had a learning rate of 0.15
train_step = tf.train.GradientDescentOptimizer(0.15).minimize(loss)

# whoa - that adds a lot of stuff to the graph!

## In [233]: x.graph.get_operations()[-1].name
## Out[233]: u'GradientDescent'
##
## In [234]: x.graph.get_operations()[-2].name
## Out[234]: u'GradientDescent/update_Variable_1/ApplyGradientDescent'
##
## In [235]: x.graph.get_operations()[-3].name
## Out[235]: u'GradientDescent/update_Variable/ApplyGradientDescent'
##
## In [236]: x.graph.get_operations()[-4].name
## Out[236]: u'GradientDescent/learning_rate'
##
## In [237]: x.graph.get_operations()[-5].name
## Out[237]: u'gradients/mul_grad/tuple/control_dependency_1'
##
## In [238]: x.graph.get_operations()[-6].name
## Out[238]: u'gradients/mul_grad/tuple/control_dependency'
##
## In [239]: len(x.graph.get_operations())
## Out[239]: 62

print s.run(y)
## 0.817574

print s.run(train_step)
## None

print s.run(y)
## 0.810806

# hey, it's working!


for i in range(300):
    s.run(train_step)
    print i, s.run(y)
## 298 0.0132452
## 299 0.013194
# considerably better result than Nielsen's!
# due to some internal difference in the gradient descent method?
# maybe a different interpretation of learning rate?
# it does seem to have roughly the same "curve" as shown,
# though I'm not properly visualizing it here...
# maybe if I instrument it and use tensorboard

print s.run(w), s.run(b)
## -2.30736 -2.00735
# again, more extreme (tighter fit) than Nielsen's results


# second one!
# importantly, I'll have to redefine the computation,
# not just make a new constant with a Python variable name 'b'

b = tf.Variable(2.0)
# oh! should evaluate y here!
s.run(tf.initialize_all_variables())
# evaluate y again...
y = tf.sigmoid(w*x + b)
# could start new session...
s.run(y)
## 0.93086159
# huh; that isn't 0.98...
# ah! weight AND bias!
w = tf.Variable(2.0)
s.run(tf.initialize_all_variables())
y = tf.sigmoid(w*x + b)
## 0.98201376
# there we go
train_step = tf.train.GradientDescentOptimizer(0.15).minimize(y - y_)
for i in range(300):
    s.run(train_step)
    print i, s.run(y)
## 299 0.044994
# again, better final result than Nielsen's,
# but it shows the initial period of slow learning, as intended
