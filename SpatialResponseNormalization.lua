local C = ccn2.C

local SpatialResponseNormalization, parent = torch.class('ccn2.SpatialResponseNormalization', 'nn.Module')

function SpatialResponseNormalization:__init(size, addScale, powScale, minDiv)
  parent.__init(self)

  self.size = size
  self.addScale = addScale or 0.001
  -- dic['scale'] /= dic['size'] if self.norm_type == self.CROSSMAP_RESPONSE_NORM else dic['size']**2
  self.addScale = self.addScale / (self.size * self.size)
  self.powScale = powScale or 0.75
  self.minDiv = minDiv or 1.0
  -- TODO: check layer.py:1333

  self.output = torch.Tensor()
  self.gradInput = torch.Tensor()
  self.denoms = torch.Tensor()

end


function SpatialResponseNormalization:updateOutput(input)
  ccn2.typecheck(input)
  ccn2.inputcheck(input)
  local nBatch = input:size(4)
  local inputC = input:view(input:size(1) * input:size(2) * input:size(3), input:size(4))
  self.output:resize(inputC:size())

  C['convResponseNorm'](inputC:cdata(), self.denoms:cdata(), self.output:cdata(),
                           input:size(1), self.size,
                           self.addScale, self.powScale, self.minDiv)

  self.output = self.output:view(input:size(1), input:size(2), input:size(3), input:size(4))
  return self.output
end


function SpatialResponseNormalization:updateGradInput(input, gradOutput)
  ccn2.typecheck(input); ccn2.typecheck(gradOutput);
  ccn2.inputcheck(input); ccn2.inputcheck(gradOutput);
  local nBatch = input:size(4)
  local inputC = input:view(input:size(1) * input:size(2) * input:size(3), input:size(4))
  local gradOutputC = gradOutput:view(gradOutput:size(1) * gradOutput:size(2) * gradOutput:size(3), gradOutput:size(4))
  local outputC = self.output:view(gradOutput:size(1) * gradOutput:size(2) * gradOutput:size(3), gradOutput:size(4))

  self.gradInput:resize(inputC:size())

  C['convResponseNormUndo'](gradOutputC:cdata(), self.denoms:cdata(), inputC:cdata(), outputC:cdata(),
                                    self.gradInput:cdata(), input:size(1), self.size,
                                    self.addScale, self.powScale, 0, 1)
  self.gradInput = self.gradInput:view(input:size(1), input:size(2), input:size(3), input:size(4))
  return self.gradInput
end
