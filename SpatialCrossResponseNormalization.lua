local C = ccn2.C

local SpatialCrossResponseNormalization, parent = torch.class('ccn2.SpatialCrossResponseNormalization', 'nn.Module')

function SpatialCrossResponseNormalization:__init(size, addScale, powScale, minDiv, blocked)
  parent.__init(self)

  self.size = size
  self.addScale = addScale or 0.0001
  -- dic['scale'] /= dic['size'] if self.norm_type == self.CROSSMAP_RESPONSE_NORM else dic['size']**2
  self.addScale = self.addScale / self.size
  self.powScale = powScale or 0.75
  self.minDiv = minDiv or 1.0
  self.blocked = blocked or false
  -- TODO: check layer.py:1333

  self.output = torch.Tensor()
  self.gradInput = torch.Tensor()

end


function SpatialCrossResponseNormalization:updateOutput(input)
  ccn2.typecheck(input)
  ccn2.inputcheck(input)
  local nBatch = input:size(4)
  local inputC = input:view(input:size(1) * input:size(2) * input:size(3), input:size(4))
  self.output:resize(inputC:size())

  C['convResponseNormCrossMap'](inputC:cdata(), self.output:cdata(),
                                input:size(1), self.size,
                                self.addScale, self.powScale, self.minDiv, self.blocked)

  self.output = self.output:view(input:size(1), input:size(2), input:size(3), input:size(4))
  return self.output
end


function SpatialCrossResponseNormalization:updateGradInput(input, gradOutput)
  ccn2.typecheck(input); ccn2.typecheck(gradOutput);
  ccn2.inputcheck(input); ccn2.inputcheck(gradOutput);
  local nBatch = input:size(4)
  local inputC = input:view(input:size(1) * input:size(2) * input:size(3), input:size(4))
  local gradOutputC = gradOutput:view(gradOutput:size(1) * gradOutput:size(2) * gradOutput:size(3), gradOutput:size(4))
  local outputC = self.output:view(gradOutput:size(1) * gradOutput:size(2) * gradOutput:size(3), gradOutput:size(4))

  self.gradInput:resize(inputC:size())

  C['convResponseNormCrossMapUndo'](gradOutputC:cdata(), inputC:cdata(), outputC:cdata(),
                                    self.gradInput:cdata(), input:size(1), self.size,
                                    self.addScale, self.powScale, self.minDiv, self.blocked, 0, 1)
  self.gradInput = self.gradInput:view(input:size(1), input:size(2), input:size(3), input:size(4))
  return self.gradInput
end
