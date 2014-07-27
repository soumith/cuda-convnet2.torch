require 'ccn2'
require 'cunn'

local ccntest = {}
local precision_forward = 1e-4
local precision_backward = 1e-2
local nloop = 1
local times = {}


function ccntest.SpatialConvolution_forward_batch()
   local bs = math.random(1,4) * 32
   local from = math.random(1,3); 
   if math.random(1,2) == 2 then 
      from = 16 * math.random(1,8)
   end
   local to = math.random(1,8) * 16
   local ki = math.random(3,15)
   local kj = ki
   local si = 1 -- not supported by CPU version yet
   local sj = si
   local outi = math.random(1,64)
   local outj = outi
   local ini = (outi-1)*si+ki
   local inj = ini

   local tm = {}
   local title = string.format('ccn2.SpatialConvolution.forward %dx%dx%dx%d o %dx%d -> %dx%dx%dx%d [s: %dx%d]', 
                               bs, from, inj, ini, kj, ki, bs, to, outj, outi, sj, si)
   times[title] = tm

   local input = torch.randn(from,inj,ini,bs):cuda()
   local sconv = nn.SpatialConvolutionCUDA(from,to,ki,kj,si,sj):cuda()
   local groundtruth = sconv:forward(input)
   local a = torch.Timer()
   for i = 1,nloop do
      groundtruth = sconv:forward(input)
   end
   tm.cpu = a:time().real

   local gconv = ccn2.SpatialConvolution(from,to,ki,si):cuda()
   gconv.weight:copy(sconv.weight)
   gconv.bias:copy(sconv.bias)
   local rescuda = gconv:forward(input)
   a:reset()
   for i = 1,nloop do
      rescuda = gconv:forward(input)
   end
   cutorch.synchronize()
   tm.gpu = a:time().real

   local error = rescuda:add(groundtruth:mul(-1))
   mytester:assertlt(error:abs():max(), precision_forward, 'error on state (forward) ')
end

function ccntest.SpatialConvolution_backward_batch()
   local bs = math.random(1,4) * 32
   local from = math.random(1,3); 
   if math.random(1,2) == 2 then 
      from = 16 * math.random(1,8)
   end
   local to = math.random(1,8) * 16
   local ki = math.random(3,15)
   local kj = ki
   local si = 1 -- not supported by CPU version yet
   local sj = si
   local outi = math.random(1,64)
   local outj = outi
   local ini = (outi-1)*si+ki
   local inj = ini
   local tm = {}

   local title = string.format('ccn2.SpatialConvolution.backward %dx%dx%dx%d o %dx%d -> %dx%dx%dx%d', 
                               bs, from, inj, ini, kj, ki, bs, to, outj, outi)
   times[title] = tm

   local input = torch.randn(from,inj,ini,bs):cuda()
   local gradOutput = torch.randn(to,outj,outi,bs):cuda()
   local sconv = nn.SpatialConvolutionCUDA(from,to,ki,kj,si,sj):cuda()
   sconv:forward(input)
   sconv:zeroGradParameters()
   local groundgrad = sconv:updateGradInput(input, gradOutput)
   local a = torch.Timer()
   for i = 1,nloop do
      sconv:zeroGradParameters()
      groundgrad = sconv:updateGradInput(input, gradOutput)
   end
   local groundweight = sconv.gradWeight
   local groundbias = sconv.gradBias
   tm.cpu = a:time().real

   local gconv = ccn2.SpatialConvolution(from,to,ki,si):cuda()
   gconv.weight:copy(sconv.weight)
   gconv.bias:copy(sconv.bias)
   gconv:forward(input)
   gconv:zeroGradParameters()
   local rescuda = gconv:updateGradInput(input, gradOutput)
   a:reset()
   for i = 1,nloop do
      gconv:zeroGradParameters()
      rescuda = gconv:updateGradInput(input, gradOutput)
   end
   local weightcuda = gconv.gradWeight
   local biascuda = gconv.gradBias
   cutorch.synchronize()
   tm.gpu = a:time().real

   error = rescuda:add(groundgrad:mul(-1))
   werror = weightcuda:add(groundweight:mul(-1))
   berror = biascuda:add(groundbias:mul(-1))

   mytester:assertlt(error:abs():max(), precision_backward, 'error on state (backward) ')
   -- mytester:assertlt(werror:abs():max(), precision_backward, 'error on weight (backward) ')
   -- mytester:assertlt(berror:abs():max(), precision_backward, 'error on bias (backward) ')
end


torch.setdefaulttensortype('torch.FloatTensor')
math.randomseed(os.time())
jac = nn.Jacobian
mytester = torch.Tester()
mytester:add(ccntest)
mytester:run(tests)
print ''
print ' ------------------------------------------------------------------------------------------------'
print '|  Module                                                                          |  Speedup    |'
print ' ------------------------------------------------------------------------------------------------'
for module,tm in pairs(times) do
   local str = string.format('| %-80s | %4.2f        |', module, (tm.cpu / (tm.gpu or 1e6)))
   print(str)
end
print ' ------------------------------------------------------------------------------------------------'
