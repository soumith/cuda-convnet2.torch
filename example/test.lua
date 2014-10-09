require 'image'
require 'mattorch'
dofile 'load.lua'

im2 = image.load('/opt/caffe/examples/images/cat.jpg')
im3 = image.scale(im2,227,227,'bilinear')*256
im4 = im3:clone()
im4[{1,{},{}}] = im3[{3,{},{}}]
im4[{3,{},{}}] = im3[{1,{},{}}]

img_mean = mattorch.load('./ilsvrc_2012_mean.mat')['img_mean']:transpose(3,1)
im5 = im4 - image.scale(img_mean, 227, 227,'bilinear')
I = im5
--im = mattorch.load('impros.mat')

--I = im.image:transpose(2,3)

bb = torch.Tensor(32,3,227,227)

for i=1,32 do
bb[i] = I
end

model:forward(bb:cuda())

_,i = model:get(25).output[{1,{}}]:float():max(1)
print('predicted class: ', i)

--conv5 = model.modules[15].output
--mattorch.save('conv5_rcnn.mat',{conv5_t = conv5:double()})
