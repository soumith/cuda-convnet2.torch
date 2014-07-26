require 'ccn2'

model = ccn2.SpatialConvolution(3, 96, 11, 1, 0)
image = torch.randn(3, 64, 64, 128):cuda()
output = model:forward(image)
