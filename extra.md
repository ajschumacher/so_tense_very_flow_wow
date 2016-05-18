This seems like too much for the original post... (Also more work to
fix up and make interesting.)


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

```python
import tensorflow as tf

x = tf.constant([1.0], name='input')
w = tf.Variable([0.9], name='weight')
y_ = tf.constant([0.0], name='correct')

activations = {'identity': lambda y: y,
               'sigmoid': tf.sigmoid,
               'softmax': lambda y: tf.exp(y)/(tf.exp(y) + tf.exp(1 - y)),
               'tanh': tf.nn.tanh,
               'relu': tf.nn.relu,
               'elu': tf.nn.elu}
losses = {'quadratic': lambda y, y_: (y - y_)**2,
          'xentropy': lambda y, y_: -1*(y_*tf.log(y) + (1-y_)*(tf.log(1-y))),}

sess = tf.Session()
for activation_name, activation_fun in activations.items():
    for loss_name, loss_fun in losses.items():
        print activation_name, loss_name
        y = activation_fun(w * x)
        y_summary = tf.scalar_summary('output', y[0])
        loss = loss_fun(y, y_)
        train_step = tf.train.GradientDescentOptimizer(0.05).minimize(loss)
        outdir = 'log_stats/{}_{}'.format(activation_name, loss_name)
        summary_writer = tf.train.SummaryWriter(outdir)
        sess.run(tf.initialize_all_variables())
        for i in range(500):
            summary_writer.add_summary(sess.run(y_summary), i)
            sess.run(train_step)
```
