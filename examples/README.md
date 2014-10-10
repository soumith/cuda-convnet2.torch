Usage
-----

This is not a general way to load Caffe networks in Torch, but you can tweak it to load other than imagenet_deploy network. One could also make the bindings more general by using Lua protobuf implementation for parsing prototxt and model file.

As we can't redistribute Caffe weights you have to install and make caffe with matlab support.
Load imagenet network and save it's weights to a matlab file with the following command from matlab:
```
save_caffe_weights(prototxtfilename,modelfilename);
```
It will generate a .mat file with as many variables as there are weights in the network. For imagenet, the weights are stored as
```
conv1_w
conv1_b
conv2_w
conv2_w
conv3_w
conv3_b
conv4_w
conv4_b
conv5_w
conv5_b
fc6_w
fc6_b
fc7_w
fc7_b
fc8_w
fc8_b
```
Then you can load the network from Torch like this:
```
net = load_imagenet('path_to_imagenet_weight.mat')
```
It will give you the usual Torch network. To preprocess the input image use preprocess function as in the test.lua file. Example:
```
th test.lua caffe_reference_imagenet_model_weights.mat cat.jpg ilsvrc_2012_mean.mat
```
