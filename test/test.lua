require 'ccn2'
require 'os'

model = ccn2.SpatialConvolution(3, 96, 11)
image = torch.randn(3, 128, 128, 128):cuda()
output = model:forward(image)
cutorch:synchronize()
runs = 10

local clk = os.clock()
for i=1,runs do
   output = model:forward(image)
   cutorch:synchronize()
end
print((os.clock()-clk)/runs)
