require 'ccn2'
require 'cunn'

local jac_ccntest = {}
local precision = 1e-5


function jac_ccntest.SpatialConvolution_CPU_Jacobian2()
    local from = math.random(1,10)
    local to = math.random(1,10)
    local ki = math.random(1,10)
    local kj = math.random(1,10)
    local si = math.random(1,4)
    local sj = math.random(1,4)
    local outi = math.random(10,20)
    local outj = math.random(10,20)
    local ini = (outi-1)*si+ki
    local inj = (outj-1)*sj+kj
    local module = nn.SpatialConvolution(from, to, ki, kj, si, sj)
    local input = torch.Tensor(from, inj, ini):zero()

    -- stochastic

    local err = jac.testJacobian(module, input)
    mytester:assertlt(err, precision, 'error on state ')
end

function jac_ccntest.SpatialConvolution_CPU_Jacobian()
    local batch = 32
    local from = 1
    local to = 32
    local ki = 3
    local si = 1
    local outi = 4
    local ini = (outi-1)*si+ki

    local module = nn.SpatialConvolution(from, to, ki, ki, si, si)
    local input = torch.Tensor(batch,from,ini,ini):zero()

    local err = jac.testJacobian(module, input)
    mytester:assertlt(err, precision, 'batch error on state ')
end

function jac_ccntest.SpatialConvolution_Jacobian()
    local bs = 32
    local from = 1
    local to = 32
    local ki = 3
    local si = 1 -- not supported by CPU version yet
    local outi = 4
    local ini = (outi-1)*si+ki

    local title = string.format('ccn2.SpatialConvolution.forward %dx%dx%dx%d o %dx%d -> %dx%dx%dx%d [s: %dx%d]',
                                bs, from, ini, ini, ki, ki, bs, to, outi, outi, si, si)
    print(title)

    local module = ccn2.SpatialConvolution(from,to,ki,si):cuda()
    local input = torch.randn(from,ini,ini,bs):cuda():zero()

    local err = jac.testJacobian(module,input)
    mytester:assertlt(err,precision, 'error on state ')
end

function jac_ccntest.LocalSpatialConvolution_Jacobian()
    local bs = 32
    local from = math.random(1,3)
    local to = 32
    local ki = 3
    local si = 1 -- not supported by CPU version yet
    local outi = 4
    local ini = (outi-1)*si+ki

    local module = ccn2.LocalSpatialConvolution(from,to,ini,ki,si):cuda()
    local input = torch.randn(from,ini,ini,bs):cuda():zero()

    local err = jac.testJacobian(module,input)
    mytester:assertlt(err,precision, 'error on state ')
end

torch.setdefaulttensortype('torch.FloatTensor')
math.randomseed(os.time())
jac = nn.Jacobian
mytester = torch.Tester()
mytester:add(jac_ccntest)
mytester:run(tests)
print ''
print ' ------------------------------------------------------------------------------------------------'
print '|  Module                                                                          |  Speedup    |'
print ' ------------------------------------------------------------------------------------------------'
--for module,tm in pairs(times) do
--    local str = string.format('| %-80s | %4.2f        |', module, (tm.cpu / (tm.gpu or 1e6)))
--    print(str)
--end
print ' ------------------------------------------------------------------------------------------------'
