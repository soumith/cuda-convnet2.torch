require 'ccn2'
require 'os'
require 'cunn'

model = ccn2.SpatialConvolution(3, 96, 11)
model2 = nn.SpatialConvolutionCUDA(3, 96, 11, 11):cuda()

model2.weight:copy(model.weight)
model2.bias:copy(model.bias)

image = torch.randn(3, 128, 128, 128):cuda()
output = model:forward(image)
output2 = model2:forward(image)

print("Max Error in outputs: " .. (output:add(output2:mul(-1))):abs():max())

cutorch:synchronize()
runs = 10

local clk = os.clock()
for i=1,runs do
   output = model:forward(image)
   cutorch:synchronize()
end
print('Time taken: ' .. (os.clock()-clk)/runs)
