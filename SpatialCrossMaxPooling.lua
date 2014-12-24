local C = ccn2.C

local SpatialCrossMaxPooling, parent = torch.class('ccn2.SpatialCrossMaxPooling', 'nn.Module')

function SpatialCrossMaxPooling:__init(kD, dD)
  parent.__init(self)

  self.kD = kD
  self.dD = dD or kD

  self.output = torch.Tensor()
  self.gradInput = torch.Tensor()

end

function SpatialCrossMaxPooling:updateOutput(input)
  ccn2.typecheck(input)
  ccn2.inputcheck(input)
  local nBatch = input:size(4)
  local inputC = input:view(input:size(1) * input:size(2) * input:size(3), input:size(4))
  local outputD = math.ceil((input:size(1) - self.kD)/self.dD + 1)
  C['convCrossMapMaxPool'](inputC:cdata(), self.output:cdata(), 0, self.kD, outputD, self.dD, input:size(2))
  self.output = self.output:view(outputD, input:size(2), input:size(3), nBatch)
  return self.output
end


function SpatialCrossMaxPooling:updateGradInput(input, gradOutput)
  ccn2.typecheck(input); ccn2.typecheck(gradOutput);
  ccn2.inputcheck(input); ccn2.inputcheck(gradOutput);
  local nBatch = input:size(4)
  local inputC = input:view(input:size(1) * input:size(2) * input:size(3), input:size(4))
  local gradOutputC = gradOutput:view(gradOutput:size(1) * gradOutput:size(2) * gradOutput:size(3), gradOutput:size(4))
  local outputC = self.output:view(gradOutput:size(1) * gradOutput:size(2) * gradOutput:size(3), gradOutput:size(4))

  self.gradInput:resize(inputC:size())
  local outputX = math.ceil((input:size(2) - self.kD)/self.dD + 1)
  C['convCrossMapMaxPoolUndo'](inputC:cdata(), gradOutputC:cdata(), outputC:cdata(), self.gradInput:cdata(),
                        input:size(2), 0, self.kD, self.dD, 0, 1)
  self.gradInput = self.gradInput:view(input:size(1), input:size(2), input:size(3), input:size(4))
  return self.gradInput
end
