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
visualizer (TensorBoard), and even its own heavily engineered serving
architecture. Especially for someone hoping to explore machine
learning for the first time with TensorFlow, it can be a lot to take
in.

How does TensorFlow work? We're going to zoom in on just one
simplified neuron so that you can see every moving part and how
TensorFlow handles them. We'll explore the data flow graph, and
compare different activation functions. We'll see exactly how
TensorBoard can visualize any aspect of your TensorFlow work. The
examples here won't solve industrial machine learning problems, but
they'll help you understand the components underlying everything built
with TensorFlow.



![An object has no name.](an_object_has_no_name.jpg)

https://twitter.com/hadleywickham/status/732288980549390336

