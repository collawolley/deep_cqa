--[[
	测试能否使用GPU
--]]
require('nn')
require('cunn')
-- we define an MLP
local ninput=  30
local noutput = 10
mlp = nn.Sequential()
mlp:add(nn.Linear(ninput, 1000))
mlp:add(nn.Tanh())
mlp:add(nn.Linear(1000, 1000))
mlp:add(nn.Tanh())
mlp:add(nn.Linear(1000, 1000))
mlp:add(nn.Tanh())
mlp:add(nn.Linear(1000, noutput))
-- and move it to the GPU:
mlp:cuda()
crit = nn.AbsCriterion()	--误差计算
crit:cuda()
input = torch.randn(ninput)
output = torch.randn(noutput)
input = input:cuda()
result = mlp:forward( input:cuda() )
loss = crit:forward(result,output:cuda())
obj_los = crit:backward(result:cuda(),output:cuda())
in_loss = mlp:backward(input,obj_los:cuda())
result_cpu = loss
print(result_cpu)
