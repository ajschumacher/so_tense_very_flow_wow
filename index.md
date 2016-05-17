# TensorFlow from the Plumbing Up

The TensorFlow project might be bigger than you realize. It's a
library for deep learning, yes. That affiliation, and its connection
to Google, has helped TensorFlow attract a lot of attention. But
TensorFlow is more than "just" deep learning. The core library is
suited to a broad family of machine learning methods, and exposes a
lot of the details. Then, the execution model is unfamiliar to those
coming from, for example, Python's scikit-learn, or most tools in R.
And in addition to the core machine learning functionality, TensorFlow
also includes its own logging system, its own interactive log
visualizer, and even its own heavily engineered serving architecture.
Especially for someone hoping to explore machine learning for the
first time with TensorFlow, it can be a lot to take in.

How does TensorFlow work? We're going to zoom in on one simplified
neuron so that you can see every moving part. We'll explore the data
flow graph, and compare different activation functions. We'll see
exactly how TensorBoard can visualize any aspect of your TensorFlow
work. The examples here won't solve industrial machine learning
problems, but they'll help you understand the components underlying
everything built with TensorFlow, including whatever you build next!


### Python Points to the Graph

Let's connect the way TensorFlow manages computation with the usual
way we use Python. First it's important to remember, to paraphrase
[Hadley Wickham](https://twitter.com/hadleywickham/status/732288980549390336),
"an object has no name."

![An object has no name.](img/an_object_has_no_name.jpg)

The variable names that you use in your Python code aren't what they
represent; they're just pointing at objects. So when you say in Python
that `x = []` and `y = x`, it isn't just that `x` equals `y`; `x` _is_
`y`, in the sense that they both point at the same object.

```python
>>> x = []
>>> y = x
>>> x == y
## True
>>> x is y
## True
```

You can also see that `id(x)` and `id(y)` are the same. This identity,
especially with mutable data structures like lists, can lead to
surprising bugs for people who don't understand what Python is up to.
Internally, Python is managing all your objects and keeping track of
your variable names and which object they refer to.

When you enter a Python expression, for example at the interactive
interpreter or REPL (Read Evaluate Print Loop), whatever is read is
almost always evaluated right away. Python is eager to do what you
tell it. So if I tell Python to `x.append(y)`, it does that `append`
right away, even if I never use `x` again. A lazier alternative would
be to just remember that I said `x.append(y)`, and if I ever evaluate
`x` at some point in the future, Python could do the append then. This
would be closer to what we'll do with TensorFlow.

Recall that `x` and `y` are the same list. We've put a list inside
itself, which Python will not try to print in its entirety. You could
think of this structure as a graph with one node, pointing to itself.
Indeed, nesting lists is one way to represent a graph structure like a
TensorFlow computation graph.

```python
>>> x.append(y)
>>> x
## [[...]]
```


### The Simplest TensorFlow Graph

Inspired by examples in Michael Nielsen's
[Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/),
we're going to start simple. Let's import TensorFlow.

```python
>>> import tensorflow as tf
```

Already, TensorFlow has started managing a lot of state for us. There
is now an implicit default graph, for example. Internally the default
graph lives in the `_default_graph_stack`, but we don't have access to
that directly. We can use `tf.get_default_graph()` though.

```python
>>> graph = tf.get_default_graph()
```

The nodes of the TensorFlow graph are called "operations". We can see
what operations are in the graph by using `graph.get_operations()`.

```python
>>> graph.get_operations()
## []
```

Currently, there isn't anything in the graph. Everything we want
TensorFlow to compute with will have to get into that graph. Let's
start with a simple constant input value of one.

```python
input = tf.constant(1.0)
```

That constant now lives as a node, an operation, in the graph.

```python
>>> operations = graph.get_operations()
>>> operations
## [<tensorflow.python.framework.ops.Operation at 0x1185005d0>]
>>> input_value_op = operations[0]
>>> print(input_value_op)
## name: "Const"
## op: "Const"
## attr {
##   key: "dtype"
##   value {
##     type: DT_FLOAT
##   }
## }
## attr {
##   key: "value"
##   value {
##     tensor {
##       dtype: DT_FLOAT
##       tensor_shape {
##       }
##       float_val: 1.0
##     }
##   }
## }
```

You can see a protocol buffer representation of our simple number one.

People new to TensorFlow sometimes wonder why there's all this fuss
about making "TensorFlow versions" of things. Why can't we just use a
normal Python variable, like a NumPy array, without defining a
TensorFlow thing? [One of the tutorials](https://www.tensorflow.org/versions/r0.8/tutorials/mnist/pros/index.html#deep-mnist-for-experts)
has an explanation:

> To do efficient numerical computing in Python, we typically use
> libraries like NumPy that do expensive operations such as matrix
> multiplication outside Python, using highly efficient code
> implemented in another language. Unfortunately, there can still be a
> lot of overhead from switching back to Python every operation. This
> overhead is especially bad if you want to run computations on GPUs
> or in a distributed manner, where there can be a high cost to
> transferring data.

> TensorFlow also does its heavy lifting outside Python, but it takes
> things a step further to avoid this overhead. Instead of running a
> single expensive operation independently from Python, TensorFlow
> lets us describe a graph of interacting operations that run entirely
> outside Python. This approach is similar to that used in Theano or
> Torch.

This is true even for a single constant! If we inspect our `input`, we
see it is a constant 32-bit float tensor of no dimensions: just one
number.

```python
>>> input
## <tf.Tensor 'Const:0' shape=() dtype=float32>
```

Note that it _doesn't_ tell us what the number _is_! To evaluate
`input` and get its numerical value out, we need to create a "session"
where graph operations can be evaluated and then explicitly ask to
evaluate `input`.

```python
>>> sess = tf.Session()
>>> sess.run(input)
## 1.0
```

It feels a little strange to "run" a constant. But it isn't so
different from evaluating an expression as usual in Python; it's just
that TensorFlow is managing its own space of things - the computational
graph - and it has its own method of evaluation.


### The Simplest Neuron

The neuron itself will be have just one parameter, or "weight". Often
even these simple neurons will have an additional constant bias, but
we'll leave that out.

The neuron's weight isn't going to be a constant; we expect it to
change in order to learn based on the "true" input and output we use
for training. The weight will be a TensorFlow variable. We'll give
that variable a starting value of 0.9

```python
>>> weight = tf.Variable(0.9)
```

Now how many operations will be in the graph? You might expect it
would be two now, but in fact adding a variable with an initial value
adds four operations. We can check all the operation names:

```python
>>> for op in graph.get_operations(): print op.name
## Const
## Variable/initial_value
## Variable
## Variable/Assign
## Variable/read
```

We won't want to follow every operation individually for long, but it
will be nice to see at least one thing that feels like a real
computation.

```python
>>> output = weight * input
```

Now there are seven operations in the graph, and the last one is our
multiplication.

```python
>>> op = graph.get_operations()[-1]
>>> op.name
## 'mul'
>>> op.inputs
## <tensorflow.python.framework.ops._InputList at 0x1185b2950>
```

We could track down the graph connections via that input list, but
instead let's wait to see the TensorBoard graph visualization soon.

How do we find out what the product is? We have to "run" the `output`
operation. But that operation depends on variables, and we need to run
a special initialization operation for the variables first. The
`initialize_all_variables` function generates the appropriate
operation, which can then be run.

```python
>>> init = tf.initialize_all_variables()
>>> sess.run(init)
```

The `initialize_all_variables` function makes initializers for all the
variables _in the current graph_, so if you added more variables you
would want to run `initialize_all_variables` again; an old `init`
wouldn't include the new variables.

Now we're ready to run the `output` operation.

```python
>>> sess.run(output)
## 0.89999998
```

Recall that's `0.9 * 1.0` with 32-bit floats, and 32-bit floats have a
hard time with 0.9; that's as close as they can get.


### See Your Graph in TensorBoard

The graph to this point is simple, but already it would be nice to see
it represented in a diagram. We'll use TensorBoard to generate that
diagram. TensorBoard reads the `name` that is stored inside each
operation, quite distinct from Python variable names. This is a good
time to start using these TensorFlow names and switch to more
conventional Python variable names.

```python
import tensorflow as tf

x = tf.constant(1.0, name='input')
w = tf.Variable(0.9, name='weight')
y = tf.mul(w, x, name='output')
```

TensorBoard works by looking at a directory of output created inside
TensorFlow sessions. We can write this output with a `SummaryWriter`,
and if we do nothing beside creating one, it will write the session's
graph.

The first argument when creating the `SummaryWriter` is a directory
name, which will be created if it doesn't exist. Output will be go
there.

```python
>>> sess = tf.Session()
>>> summary_writer = tf.train.SummaryWriter('log_simple_graph', sess.graph)
```

Now, at the command line, we can start up TensorBoard.

```bash
tensorboard --logdir=log_simple_graph
```

TensorBoard runs as a local web app, on port 6006, which is "goog"
upside-down. If you go to
[localhost:6006/#graphs](http://localhost:6006/#graphs) you should see
a diagram of the graph you created in TensorFlow.

It should look something like this:

![Simple graph.](img/simple_graph.png)


### Making the Neuron Learn

What is our neuron supposed to learn? We set up an input value of 1.0.
Let's say the correct output value is 0. That is, we have a very
simple training set of just one example with one feature, which has
the value one, and one label, which is zero. We want the neuron to
learn the function taking one to zero.

Currently, the system takes the input one and returns 0.9, which is
not correct. We'll call the level of incorrectness the "loss" and give
our system the goal of minimizing the loss. If the loss can be
negative then minimizing it is a bit silly, so let's make the loss the
square of the difference between the current output and the desired
output.

```python
```python
import tensorflow as tf

x = tf.constant(1.0, name='input')
w = tf.Variable(0.9, name='weight')
y = tf.mul(w, x, name='output')
y_ = tf.constant(0.0, name='correct')
loss = tf.pow(y - y_, 2, name='loss')
```

So far, nothing in the graph does any learning. For that, we need an
optimizer. The optimizer takes a learning rate, which we'll set at
0.05.

```python
>>> opt = tf.train.GradientDescentOptimizer(learning_rate=0.05)
```

The optimizer if remarkably clever. It can automatically work out and
apply the appropriate gradients through a whole network, carrying out
the backward step for learning.

Let's see what the gradient looks like for our simple example.

```python
>>> grads_and_vars = opt.compute_gradients(loss)
>>> sess = tf.Session()
>>> sess.run(tf.initialize_all_variables())
>>> sess.run(grads_and_vars[0][0])
## 1.8
```

Why is it 1.8? Our loss is error squared, and the derivative of that
is two times the error. Currently the system says 0.9 instead of 0, so
the error is 0.9, and two times 0.9 is 1.8. It's working!

For more complex systems, it will be very nice indeed that TensorFlow
calculates and then applies these gradients for us automatically.

Let's apply the gradient, finishing the backpropagation.

```
>>> sess.run(opt.apply_gradients(grads_and_vars))
>>> sess.run(w)
## 0.81
```

The weight decreased by 0.09 because the optimizer subtracted the
gradient times the learning rate (1.8 * 0.05), pushing the weight in
the right direction.

Instead of hand-holding the optimizer like this, we can make one
operation that calculates and applies the gradients.

```python
>>> train_step = tf.train.GradientDescentOptimizer(0.05).minimize(loss)
>>> for i in range(300):
>>>     sess.run(train_step)
>>> sess.run(y)
## 1.5178819e-14
```

Running the training step many times, the weight and so the output
value are very close to zero. The neuron has learned!


### Training Diagnostics in TensorBoard

We're may be interested in what's happening during training. Say we
want to follow what our system is predicting after every training
step. We could `print` from inside the training loop.

```python
import tensorflow as tf

x = tf.constant(1.0, name='input')
w = tf.Variable(0.9, name='weight')
y = tf.mul(w, x, name='output')
y_ = tf.constant(0.0, name='correct')
loss = tf.pow(y - y_, 2, name='loss')
train_step = tf.train.GradientDescentOptimizer(0.05).minimize(loss)

sess = tf.Session()
sess.run(tf.initialize_all_variables())
for i in range(100):
    print('at step {}, y is {}'.format(i, sess.run(y)))
    sess.run(train_step)
## at step 0, y is 0.899999976158
## at step 1, y is 0.810000002384
## at step 2, y is 0.728999972343
## at step 3, y is 0.656099975109
## ...
## at step 99, y is 2.65614053205e-05
```

This works, but there are some problems. It's hard to understand a
list of numbers. A graph would be better. And even with only one value
to monitor, there's too much output to read. We're likely to want to
monitor many things. It would be nice to record everything in some
organized way.

Luckily, the same system that we used earlier to visualize the graph
also has mechanisms that are just what we need.

We instrument our graph by adding operations that summarize its state.
Here we'll create one that reports the current value of `y`, our
neuron's current prediction.

```python
>>> y_summary = tf.scalar_summary('output', y)
```

When it's run, a summary returns a string of protocol buffer text
which can be written to a log directory with a `SummaryWriter`.

```python
>>> summary_writer = tf.train.SummaryWriter('log_simple_stat', sess.graph)
>>> sess.run(tf.initialize_all_variables())
>>> for i in range(100):
>>>    summary_str = sess.run(y_summary)
>>>    summary_writer.add_summary(summary_str, i)
>>>    sess.run(train_step)
```

Now after running `tensorboard --logdir=log_simple_stat`, you get an
interactive plot at
[localhost:6006/#events](http://localhost:6006/#events).

![Output at each train step.](img/simple_stat.png)


### Exploring Activation Functions

So far our neuron has been very simple, even as individual neurons go.
In as much as it has an
[activation function](https://en.wikipedia.org/wiki/Activation_function#Comparison_of_activation_functions),
it's the identity function. It's a purely linear neuron, and these are
not very interesting. We'll want to add some sort of nonlinearity, or
it will be pretty pointless to make multi-layer networks.

We're also using a quadratic loss, which is kind of dull.

```python
import tensorflow as tf

x = tf.constant(1.0, name='input')
w = tf.Variable(0.9, name='weight')
y = tf.mul(w, x, name='output')
y_summary = tf.scalar_summary('output', y)
y_ = tf.constant(0.0, name='correct')
loss = tf.pow(y - y_, 2, name='loss')
train_step = tf.train.GradientDescentOptimizer(0.05).minimize(loss)

summary_writer = tf.train.SummaryWriter('log_stats_test/identity')
sess.run(tf.initialize_all_variables())
for i in range(100):
   summary_str = sess.run(y_summary)
   summary_writer.add_summary(summary_str, i)
   sess.run(train_step)

y = tf.sigmoid(tf.mul(w, x), name='output')
y_summary = tf.scalar_summary('output', y)
loss = tf.pow(y - y_, 2, name='loss')
train_step = tf.train.GradientDescentOptimizer(0.05).minimize(loss)
summary_writer = tf.train.SummaryWriter('log_stats_test/sigmoid')
sess.run(tf.initialize_all_variables())
for i in range(100):
   summary_str = sess.run(y_summary)
   summary_writer.add_summary(summary_str, i)
   sess.run(train_step)
```
