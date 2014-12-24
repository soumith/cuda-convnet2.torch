local C = ccn2.C

local SpatialResizeBilinear, parent = torch.class('ccn2.SpatialResizeBilinear', 'nn.Module')

function SpatialResizeBilinear:__init(scale, tgtsize)
  parent.__init(self)

  self.scale = scale or 1
  self.tgtsize = tgtsize

  self.output = torch.Tensor()
end

function SpatialResizeBilinear:updateOutput(input)
  ccn2.typecheck(input)
  ccn2.inputcheck(input)
  local nBatch = input:size(4)
  local inputC = input:view(input:size(1) * input:size(2) * input:size(3), input:size(4))

  local imgsize = input:size(3)
  local tgtsize = self.tgtsize or imgsize

  C['convResizeBilinear'](inputC:cdata(), self.output:cdata(), imgsize, tgtsize, self.scale)

  self.output = self.output:view(input:size(1), tgtsize, tgtsize, nBatch)
  return self.output
end
