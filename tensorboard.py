# let's see if we can make some stuff and look at it with tensorboard

#! mkdir output

import tensorflow as tf

w = tf.Variable(2.0)
b = tf.Variable(0.9)
x = tf.constant(1.0)
y_ = tf.constant(0.0)
y = tf.sigmoid(w*x + b)
y_observer = tf.scalar_summary(y.op.name, y)
# train_step = tf.train.GradientDescentOptimizer(0.15).minimize((y - y_)**2)
train_step = tf.train.GradientDescentOptimizer(0.15).minimize(y - y_)

summary_op = tf.merge_all_summaries()
# why is this None?
# ah! the scalar_summary-s are part of the graph!

sess = tf.Session()
summary_writer = tf.train.SummaryWriter('ooutput', sess.graph)
# ah! it will make the output dir!

# even just running to this point produces a graph
# of 9825 bytes: `events.out.tfevents.1463495977.phab.local` in `output`

# let's see it!
#! tensorboard --logdir=output
# yup works

sess.run(tf.initialize_all_variables())

for i in range(300):
    summary_str = sess.run(summary_op)
    summary_writer.add_summary(summary_str, i)
    sess.run(train_step)
    # print i, sess.run(y)

summary_str = sess.run(summary_op)
summary_writer.add_summary(summary_str, i+1)
