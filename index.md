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


### Python Pointing to the Graph

Let's connect the way TensorFlow manages computation with the usual
way we use Python. First it's important to remember, to paraphrase
[Hadley Wickham](https://twitter.com/hadleywickham/status/732288980549390336),
"An object has no name."

![An object has no name.](img/an_object_has_no_name.jpg)

The variable names that you use in your Python code aren't what they
represent; they're just pointing at objects. So when you say in Python
that `x = []` and `y = x`, it isn't then just that `x` equals `y`; `x`
_is_ `y`, in the sense that they both point at the same object.

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
tell it. So if I tell Python to `x.append(y)`, it does that append
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

We're going to build a single neuron and task it with learning a
function that takes the input number one to the output number zero.
(The setup is inspired by examples in Michael Nielsen's
[Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/).)
Let's import TensorFlow.

```python
>>> import tensorflow as tf
```

Already, TensorFlow has started managing a lot of state for us. There
is now an implicit default graph, for example. Internally the default
graph lives in the `_default_graph_stack`, but we don't have access to
that directly. We can however use `tf.get_default_graph()`.

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
start with our simple constant input value of one.

```python
input_value = tf.constant(1)
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
##       int_val: 1
##     }
##   }
}
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

This is true even for a single constant! If we inspect our
`input_value`, we see it is a constant 32-bit integer tensor of no
dimensions: just one number.

```python
>>> input_value
## <tf.Tensor 'Const:0' shape=() dtype=int32>
```

Note that it _doesn't_ tell us what the number _is_! To evaluate
`input_value` and get its numerical value out, we need to create a
"session" where graph operations can be evaluated and then explicitly
ask to evaluate `input_value`.

```python
>>> sess = tf.Session()
>>> sess.run(input_value)
## 1
```

It feels a little strange to "run" a constant. But it isn't so
different from evaluating an expression as usual in Python; it's just
that TensorFlow is managing its own space of things - the computational
graph - and it has its own method of evaluation.
