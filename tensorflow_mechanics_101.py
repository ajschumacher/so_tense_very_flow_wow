# https://www.tensorflow.org/versions/r0.8/tutorials/mnist/tf/index.html#tensorflow-mechanics-101

# in https://github.com/tensorflow/tensorflow/blob/r0.8/tensorflow/examples/tutorials/mnist/fully_connected_feed.py
#> To inspect the values of your Ops or variables, you may include them in the list passed to sess.run() and the value tensors will be returned in the tuple from the call.

# This one seems to be much more about reading the code...

# huh; funny that they don't link to https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/mnist/input_data.py and also don't really need it, since it's just doing another import which gets used directly. doesn't seem ideal.

# yadda yadda, this is a lot of junk
# sort of the tutorial for software engineers, maybe
# it's a different architecture, sure, but whatevs
# they use
tf.nn.sparse_softmax_cross_entropy_with_logits
# instead of doing that by hand, which is whatever

# oh this is worth learning:
#> it takes the loss tensor from the loss() function and hands it to a tf.scalar_summary, an op for generating summary values into the events file when used with a SummaryWriter (see below)
# I think I need that for using tensorboard

# to log for tensorboard:
tf.scalar_summary(loss.op.name, loss)
summary_op = tf.merge_all_summaries()
# make session
summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, sess.graph)
summary_str = sess.run(summary_op, feed_dict=feed_dict)
summary_writer.add_summary(summary_str, step)

# okay, this is good too:
saver = tf.train.Saver()
saver.save(sess, FLAGS.train_dir, global_step=step)

# oh neat, there's this:
tf.nn.in_top_k

# okay, so this introduces some new stuff. cool cool.
