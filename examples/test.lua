require 'image'
require 'mattorch'
dofile 'load_imagenet.lua'

matfilename = arg[1]
imagename = arg[2]
meanmatname = arg[3]

model = load_imagenet(matfilename)
model:evaluate()

im = image.load(imagename)

I = preprocess(im,meanmatname)
batch = torch.Tensor(32,3,227,227)

for i=1,32 do
  batch[i] = I
end

model:forward(batch:cuda())

_,i = model:get(25).output[{1,{}}]:float():max(1)
print('predicted class: ', i)
