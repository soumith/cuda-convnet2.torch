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

print("Max Error in outputs: " .. (output:float()-output2:float()):abs():max())
-- print((output:float()-output2:float()):abs():sum())
-- print(output[{{1,32},1,1,1}])
-- print(output2[32][1][1][1])

cutorch:synchronize()
runs = 1

local clk = os.clock()
for i=1,runs do
   output = model:forward(image)
   cutorch:synchronize()
end
print('Time taken: ' .. (os.clock()-clk)/runs)
