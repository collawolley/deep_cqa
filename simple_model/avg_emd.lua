--[[
	答案和问题的词向量分别求均值，二者做拼接，送给分类器做分类
--]]
local AvgEmd,parent = torch.class('deep_cqa.AvgEmd','nn.Module')
--神经网络层，求词向量的均值
function AvgEmd:__init()
	self.in_dim = deep_cqa.config.emd_dim
	self.out_dim = deep_cqa.config.emd_dim
end

function AvgEmd:forward(inputs)
	local size = inputs:size(1)
	local sum = torch.Tensor(self.in_dim):zero()
	for i=1,size do
		sum = sum + inputs[i]
	end
	sum  = sum/size
	self.output = sum
	return self.output
end
