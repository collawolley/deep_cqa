--[[
将LSTM单元进行封装，将其封装分为双向的LSTM单元，继承nn.Module
输入为Tensor，代表了一个句子的词向量表达
输出为两个多时刻的输出序列，两个排列均为正向，但是其产生的方式一正一反
默认两个单元之间的参数不共享
--]]

local BiLSTM, parent = torch.class('deep_cqa.BiLSTM', 'nn.Module')

function BiLSTM:__init(config)
	parent.__init(self)

 	self.in_dim = config.in_dim
 	self.mem_dim = config.mem_dim or 150
  	self.num_layers = config.num_layers or 1
	self.gate_output = config.gate_output
	self.cuda = config.cuda or false
  	if self.gate_output == nil then self.gate_output = true end
	
	self.fwd = deep_cqa.LSTM(config)	--正向序列
	self.rwd = deep_cqa.LSTM(config)	--反向序列
end
-- Forward propagate.
-- inputs: T x in_dim tensor, where T is the number of time steps.
-- reverse: if true, read the input from right to left (useful for bidirectional LSTMs).
-- Returns the final hidden state of the LSTM.
function BiLSTM:forward(inputs)
	self.output = {}
	self.output[1] = self.fwd:forward(inputs)	--正向序列
	self.output[2] = self.rwd:forward(inputs,true)	--反向序列
 	return self.output
end

-- Backpropagate. forward() must have been called previously on the same input.
-- inputs: T x in_dim tensor, where T is the number of time steps.
-- grad_outputs: 2 x T x num_layers x mem_dima tensor.
-- Returns the gradients with respect to the inputs (in the same order as the inputs).
function BiLSTM:backward(inputs, grad_outputs)	
	local fi_grads = self.fwd:backward(inputs,grad_outputs[1])	--正向序列的求导
	local ri_grads = self.fwd:backward(inputs,grad_outputs[2],true)	--反向序列的求导
	local input_grads = {}
	for i= 1,#inputs do
		input_grads[i] = fi_grads[i]+ri_grads[i]
	end
	return input_grads
end
function BiLSTM:zeroGradParameters()
	self.fwd:zeroGradParameters()
	self.rwd:zeroGradParameters()
end
function BiLSTM:updateParameters(learningRate)
	self.fwd:updateParameters(learningRate)
	self.rwd:updateParameters(learningRate)
end


