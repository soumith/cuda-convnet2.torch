require 'nn'
require 'cunn'
require 'ccn2'
require 'xlua'

fSize = {3, 96, 128, 128, 384}
inputSize = {3, 64, 64}
batchSize = 128

model = nn.Sequential()
model:add(ccn2.SpatialConvolution(fSize[1], fSize[2], 9))
model:add(nn.ReLU())
model:add(ccn2.SpatialMaxPooling(2,2))
model:add(ccn2.SpatialConvolution(fSize[2], fSize[3], 5))
model:add(nn.ReLU())
model:add(ccn2.SpatialMaxPooling(2,2))
model:add(ccn2.SpatialCrossResponseNormalization(4))
model:add(ccn2.SpatialConvolution(fSize[3], fSize[4], 4))
model:add(nn.ReLU())
model:add(ccn2.SpatialConvolution(fSize[4], fSize[5], 3))
model:add(nn.ReLU())
model:add(ccn2.SpatialMaxPooling(2,2))

model = model:cuda()


input = torch.rand(inputSize[1], inputSize[2], inputSize[3], batchSize):cuda()

do
   for i=1,1000 do
      if i % 20 == 0 then
         xlua.progress(i, 1000)
         collectgarbage()
      end
      local output = model:forward(input)
      model:backward(input, output)      
   end
end

