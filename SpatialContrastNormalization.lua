local C = ccn2.C

local SpatialContrastNormalization, parent = torch.class('ccn2.SpatialContrastNormalization', 'nn.Module')

function SpatialContrastNormalization:__init(sizeX, scale, pow)
  parent.__init(self)

  self.sizeX = sizeX;
  self.scale = scale;
  self.pow = pow;

  self.output = torch.Tensor()
  self.gradInput = torch.Tensor()
  self.meanDiffs = torch.Tensor()

end

function SpatialContrastNormalization:updateOutput(input)
  ccn2.typecheck(input)
  ccn2.inputcheck(input)

  local nBatch = input:size(4)
  local inputC = input:view(input:size(1) * input:size(2) * input:size(3), input:size(4))

  local outputX = input:size(2) - self.kW + 1

  self.output:resize(inputC:size())

  C['convLocalAvgPool'](inputC:cdata(), self.meanDiffs:cdata(), input:size(1), self.size, -self.size/2, 1, outputX)
end
