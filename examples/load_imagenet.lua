-- Copyright (c) 2014 by 
--    Sergey Zagoruyko <sergey.zagoruyko@imagine.enpc.fr>
--    Francisco Massa <fvsmassa@gmail.com>
-- Universite Paris-Est Marne-la-Vallee/ENPC, LIGM, IMAGINE group

require 'cunn'
require 'ccn2'
require 'mattorch'


function load_imagenet(matfilename)
  local fSize = {3, 96, 256, 384, 384, 256, 256*6*6, 4096, 4096, 1000}

  local model = nn.Sequential()

  model:add(nn.Transpose({1,4}, {1,3}, {1,2}))

  model:add(ccn2.SpatialConvolution(fSize[1], fSize[2], 11, 4))		-- conv1
  model:add(nn.ReLU())							-- relu1
  model:add(ccn2.SpatialMaxPooling(3,2))					-- pool1
  model:add(ccn2.SpatialCrossResponseNormalization(5))			-- norm1

  model:add(ccn2.SpatialConvolution(fSize[2], fSize[3], 5, 1, 2, 2))	-- conv2
  model:add(nn.ReLU())							-- relu2
  model:add(ccn2.SpatialMaxPooling(3,2))					-- pool2
  model:add(ccn2.SpatialCrossResponseNormalization(5))			-- norm2

  model:add(ccn2.SpatialConvolution(fSize[3], fSize[4], 3, 1, 1))		-- conv3
  model:add(nn.ReLU())							-- relu3

  model:add(ccn2.SpatialConvolution(fSize[4], fSize[5], 3, 1, 1, 2))	-- conv4
  model:add(nn.ReLU())							-- relu4

  model:add(ccn2.SpatialConvolution(fSize[5], fSize[6], 3, 1, 1, 2))	-- conv5
  model:add(nn.ReLU())							-- relu5

  model:add(ccn2.SpatialMaxPooling(3,2))					-- pool5
  model:add(nn.Transpose({4,1},{4,2},{4,3}))
  model:add(nn.Reshape(fSize[7]))

  model:add(nn.Linear(fSize[7], fSize[8]))	-- fc6
  model:add(nn.ReLU())				-- relu6
  model:add(nn.Dropout(0.5))			-- drop6

  model:add(nn.Linear(fSize[8], fSize[9]))	-- fc7
  model:add(nn.ReLU())				-- relu7
  model:add(nn.Dropout(0.5))			-- drop7

  model:add(nn.Linear(fSize[9], fSize[10]))	-- fc8

  model:cuda()

  -- run to check consistency
  local input = torch.randn(32, 3, 227, 227):cuda()
  local output = model:forward(input)
  print(output:size())

  local mat = mattorch.load(matfilename)

  local i = 2
  model:get(i).weight = mat['conv1_w']:transpose(1,4):transpose(1,3):transpose(1,2):reshape(fSize[1]*11*11, fSize[2]):contiguous():cuda()
  model:get(i).bias = mat['conv1_b']:squeeze():cuda()

  i = 6
  model:get(i).weight = mat['conv2_w']:transpose(1,4):transpose(1,3):transpose(1,2):reshape(fSize[2]*5*5/2, fSize[3]):contiguous():cuda()
  model:get(i).bias = mat['conv2_b']:squeeze():cuda()

  i = 10
  model:get(i).weight = mat['conv3_w']:transpose(1,4):transpose(1,3):transpose(1,2):reshape(fSize[3]*3*3, fSize[4]):contiguous():cuda()
  model:get(i).bias = mat['conv3_b']:squeeze():cuda()

  i = 12
  model:get(i).weight = mat['conv4_w']:transpose(1,4):transpose(1,3):transpose(1,2):reshape(fSize[4]*3*3/2, fSize[5]):contiguous():cuda()
  model:get(i).bias = mat['conv4_b']:squeeze():cuda()

  i = 14
  model:get(i).weight = mat['conv5_w']:transpose(1,4):transpose(1,3):transpose(1,2):reshape(fSize[5]*3*3/2, fSize[6]):contiguous():cuda()
  model:get(i).bias = mat['conv5_b']:squeeze():cuda()

  i = 19
  model:get(i).weight = mat['fc6_w']:cuda()
  model:get(i).bias = mat['fc6_b']:squeeze():cuda()

  i = 22
  model:get(i).weight = mat['fc7_w']:cuda()
  model:get(i).bias = mat['fc7_b']:squeeze():cuda()

  i = 25
  model:get(i).weight = mat['fc8_w']:cuda()
  model:get(i).bias = mat['fc8_b']:squeeze():cuda()

  -- run again to check consistency
  output = model:forward(input)
  print(output:size())
  print(model)

  return model
end

function preprocess(im, meanfilename)
  -- rescale the image
  local im3 = image.scale(im,227,227,'bilinear')*255
  -- RGB2BGR
  local im4 = im3:clone()
  im4[{1,{},{}}] = im3[{3,{},{}}]
  im4[{3,{},{}}] = im3[{1,{},{}}]

  -- subtract imagenet mean
  local img_mean = mattorch.load(meanfilename)['img_mean']:transpose(3,1)
  return im4 - image.scale(img_mean, 227, 227,'bilinear')
end

