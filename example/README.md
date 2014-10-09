Usage
-----

As we can't redistribute Caffe weights you have to install and make caffe with matlab or python support.
Load imagenet network and save it's weights to a matlab file with the following weights:

  conv1_w
  conv1_b
  conv2_w
  conv2_w
  conv3_w
  conv3_b
  conv4_b
  conv4_b
  conv5_b
  conv5_b
  conv6_b
  conv6_b
  conv7_b
  conv7_b
  conv8_b
  conv8_b

Then you can load the network with like this:

  net = load_imagenet('path_to_imagenet_weight.mat')

It will give you the usual Torch network. To preprocess the input image use preprocess function as in the test.lua file
