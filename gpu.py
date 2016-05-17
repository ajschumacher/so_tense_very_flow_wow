# running on a real machine~

# following deep_mnist_for_experts.py
# cool output!

## I tensorflow/core/common_runtime/gpu/gpu_init.cc:102] Found device 0 with properties: 
## name: GeForce GTX TITAN X
## major: 5 minor: 2 memoryClockRate (GHz) 1.2155
## pciBusID 0000:09:00.0
## Total memory: 12.00GiB
## Free memory: 11.39GiB
## I tensorflow/core/common_runtime/gpu/gpu_init.cc:102] Found device 1 with properties: 
## name: GeForce GTX TITAN X
## major: 5 minor: 2 memoryClockRate (GHz) 1.2155
## pciBusID 0000:05:00.0
## Total memory: 12.00GiB
## Free memory: 11.86GiB
## I tensorflow/core/common_runtime/gpu/gpu_init.cc:126] DMA: 0 1 
## I tensorflow/core/common_runtime/gpu/gpu_init.cc:136] 0:   Y Y 
## I tensorflow/core/common_runtime/gpu/gpu_init.cc:136] 1:   Y Y 
## I tensorflow/core/common_runtime/gpu/gpu_device.cc:755] Creating TensorFlow device (/gpu:0) -> (device: 0, name: GeForce GTX TITAN X, pci bus id: 0000:09:00.0)
## I tensorflow/core/common_runtime/gpu/gpu_device.cc:755] Creating TensorFlow device (/gpu:1) -> (device: 1, name: GeForce GTX TITAN X, pci bus id: 0000:05:00.0)

# confirms that the result called "92%" is really 0.9092

# runs 63x as fast - wild
# seems to very automatically use (a single) GPU
# and only at 2/3 capacity for this task
# also though, looks like it spawns hella python tasks
