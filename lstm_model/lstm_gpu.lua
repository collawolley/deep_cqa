--[[
	使用rnn的LSTM代码实现正确答案和错误答案的分类	
	author:	liangjz
	time:	2015-12-22
--]]
require('..')
local cfg = {}
cfg.vecs = nil
cfg.dict = nil
cfg.emd = nil
cfg.dim = deep_cqa.config.emd_dim
cfg.mem = 10
cfg.batch  = 10
-----------------------
function get_index(sent)
	--	获取一个句子的索引表达，作为整个模型的输入，可以直接应用到词向量层
	if cfg.dict == nil then
		--	载入字典和词向量查询层
		cfg.dict, cfg.vecs = deep_cqa.get_sub_embedding()
		cfg.emd = nn.LookupTable(cfg.vecs:size(1),cfg.dim)
		cfg.emd.weight:copy(cfg.vecs)
		cfg.vecs = nil
	end
	return deep_cqa.read_one_sentence(sent,cfg.dict)
end
--------------------------
function get_emd(index)
	--	备用函数，输入词的index组，输出查询到的词向量
	if cfg.emd == nil then 
		get_index('good start')
	end
	return  cfg.emd:forward(index)
end
------------------------------------------------------------------------
function getlm()
	get_index('today is')
	local lm = {}
	lm.emd = cfg.emd

	lm.qlstm = nn.Sequential()
	lm.qlstm:add(nn.Identity())
	lm.qlstm:add(nn.SplitTable(1))
	lm.qlstm:add(nn.Sequencer(nn.FastLSTM(cfg.dim,cfg.mem)))
	lm.qlstm:add(nn.SelectTable(-1))
	lm.qlstm:cuda()
	
	lm.tlstm = nn.Sequential()
	lm.tlstm:add(nn.Identity())
	lm.tlstm:add(nn.SplitTable(1))
	lm.tlstm:add(nn.Sequencer(nn.FastLSTM(cfg.dim,cfg.mem)))
	lm.tlstm:add(nn.SelectTable(-1))
	lm.tlstm:cuda()

	lm.flstm = nn.Sequential()
	lm.flstm:add(nn.Identity())
	lm.flstm:add(nn.SplitTable(1))
	lm.flstm:add(nn.Sequencer(nn.FastLSTM(cfg.dim,cfg.mem)))
	lm.flstm:add(nn.SelectTable(-1))
	lm.flstm:cuda()
	
	lm.sub = nn.CSubTable():cuda()
	
	lm.qrsp = nn.Reshape(cfg.mem,1):cuda()
	lm.arsp = nn.Reshape(1,cfg.mem):cuda()
	
	lm.mm = nn.MM():cuda()
	lm.lin = nn.Linear(cfg.mem^2,1):cuda()
	lm.cm = nn.Sequential()
	lm.cm:add(lm.mm)
	lm.cm:add(nn.Reshape(cfg.mem^2))
	lm.cm:add(lm.lin)
	lm.cm:add(nn.SoftSign())
	lm.cm:cuda()

	return lm
end
-------------------------
function train()
	local lm = getlm()

	local modules = nn.Parallel():add(lm.qlstm):add(lm.tlstm):add(lm.flstm):add(lm.sub):add(lm.qrsp):add(lm.arsp):add(lm.mm)
	params,grad_params = modules:getParameters()

	local train_set = torch.load(deep_cqa.ins_meth.train)
	local indices = torch.randperm(train_set.size)
	local criterion = nn.MarginCriterion(1):cuda()
	local gold = torch.Tensor({1}):cuda()
	local batch_size = 100
	local optim_state = {learningRate = 0.05 }
	
	for i= 1,train_set.size,batch_size do
		local size = math.min(i+batch_size-1,train_set.size)-i+1
		local feval = function(x)
			grad_params:zero()
			local loss = 0	--累积的损失
			for j = 1,size do
				xlua.progress(i+j-1,train_set.size)
				local idx = indices[i+j-1]
				local sample = train_set[idx]
				local vecs={}
				for k =1,#sample do
					local index = get_index(sample[k])
					vecs[k] = lm.emd:forward(index):clone():cuda()
				end
				
				if(idx %2 ==0) then
					vecs[3],vecs[2] = vecs[2],vecs[3]
					gold[1] = -1
				else
					gold[1] = 1
				end

				local r1 = lm.qlstm:forward(vecs[1])
				local r2 = lm.tlstm:forward(vecs[2])
				local r3 = lm.flstm:forward(vecs[3])
				local r4 = lm.sub:forward({r2,r3})
				local r5 = lm.qrsp:forward(r1)
				local r6 = lm.arsp:forward(r4)
				local pred = lm.cm:forward({r5,r6})
				local loss = loss + criterion:forward(pred,gold)

				local e1 = criterion:backward(pred,gold)
				local e2 = lm.cm:backward({r5,r6},e1)
				local e3 = lm.qrsp:backward(r1,e2[1])
				local e4 = lm.arsp:backward(r4,e2[2])
				local e5 = lm.sub:backward({r2,r3},e4)
				local e6 = lm.tlstm:backward(vecs[2],e5[1])
				local e7 = lm.flstm:backward(vecs[3],e5[2])
				local e8 = lm.qlstm:backward(vecs[1],e3)
				
			end
			grad_params = grad_params/size
			loss = loss / size
			loss = loss + 1e-4*params:norm()^2
			return loss,grad_params		
		end
		optim.adagrad(feval,params,optim_state)
	end

end
------------------------------------------------------------------------
train()
print('\n')

