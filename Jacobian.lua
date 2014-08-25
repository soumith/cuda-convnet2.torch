ccn2.Jacobian = {}

function ccn2.Jacobian.backward(module, input, batchNum)
   local batchSize = input:size(4)
   -- output deriv
   module:forward(input)
   local dout = module.output:clone()
   -- 1D batch view
   local sdout = dout:view(dout:nElement()/batchSize, batchSize)
   -- jacobian matrix to calculate
   local jacobian = torch.FloatTensor(input:nElement()/batchSize,dout:nElement()/batchSize):zero()

   for i=1,sdout:nElement()/batchSize do
      dout:zero()
      sdout[i][batchNum] = 1
      module:zeroGradParameters()
      local din = module:updateGradInput(input, dout)[{{},{},{},{batchNum}}]:contiguous()
      module:accGradParameters(input, dout)
      jacobian:select(2,i):copy(din)
   end
   return jacobian
end

function ccn2.Jacobian.forward(module, input, batchNum)
   local batchSize = input:size(4)
   -- perturbation amount
   local small = 1e-3
   -- 1D batch view of input
   local sin = input:view(input:nElement()/batchSize, batchSize)
   -- jacobian matrix to calculate
   local jacobian = torch.FloatTensor():resize(input:nElement()/batchSize, module:forward(input):nElement()/batchSize)
   
   local outa = input.new(jacobian:size(2))
   local outb = input.new(jacobian:size(2))
   
   for i=1,sin:size(1) do
      sin[i][batchNum] = sin[i][batchNum] - small -- x - eps
      outa:copy(module:forward(input)[{{},{},{},{batchNum}}]:contiguous())  -- f(x - eps)
      sin[i][batchNum] = sin[i][batchNum] + 2*small  -- x + eps
      outb:copy(module:forward(input)[{{},{},{},{batchNum}}]:contiguous())  -- f(x + eps)
      sin[i][batchNum] = sin[i][batchNum] - small  -- x

      outb:add(-1,outa):div(2*small) -- f(x+eps) - f(x-eps) / 2*eps
      jacobian:select(1,i):copy(outb)
   end

   return jacobian
end

function ccn2.Jacobian.testJacobian (module, input, minval, maxval)
   minval = minval or -2
   maxval = maxval or 2
   local inrange = maxval - minval
   input:copy(torch.rand(input:nElement()):mul(inrange):add(minval))
   local batch = math.random(1, input:size(4))
   local jac_fprop = ccn2.Jacobian.forward(module, input, batch)
   local jac_bprop = ccn2.Jacobian.backward(module, input, batch)
   local error = jac_fprop-jac_bprop
   return error:abs():max(), error:abs():mean()
end

