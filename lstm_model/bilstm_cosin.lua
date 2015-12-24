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
cfg.mem = 30
cfg.batch  = 50
deep_cqa.ins_meth.load_binary()
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
	lm.emd = cfg.emd:cuda()

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

cfg.lm = getlm()
-------------------------
function train()
	local lm = cfg.lm

	local modules = nn.Parallel():add(lm.emd):add(lm.qlstm):add(lm.tlstm):add(lm.flstm):add(lm.sub):add(lm.qrsp):add(lm.arsp):add(lm.mm)
	params,grad_params = modules:getParameters()

	local train_set = torch.load(deep_cqa.ins_meth.train)
	local indices = torch.randperm(train_set.size)
	local criterion = nn.MarginCriterion(1):cuda()
	local gold = torch.Tensor({1}):cuda()
	local batch_size = cfg.batch
	local optim_state = {learningRate = 0.05 }
	train_set.size =2000
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
					local index = get_index(sample[k]):cuda()
					vecs[k] = lm.emd:forward(index):clone()
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
function test_one_pair(qst,ans)
	--给定一个问答pair，计算其相似度	
	--传入的qst为已经计算好的向量，ans为未经处理的句子
--[
	local lm = cfg.lm
	local aidx = get_index(ans):cuda()
	local aemd = lm.emd:forward(aidx):clone()
	local avec = lm.tlstm:forward(aemd)
	local r5 = lm.qrsp:forward(qst)
	local r6 = lm.arsp:forward(avec)
	local pred = lm.cm:forward({r5,r6})

	return pred[1]
--]
end
function evaluate(name)
	--评估训练好的模型的精度，top 1是正确答案的比例
	local test_set = deep_cqa.insurance[name]
	local answer_set = deep_cqa.insurance['answer']
	if(test_set == nil) then
		print('测试集载入为空！') return 
	end
	local lm = cfg.lm	--语言模型
	local results = {}
	
	for i,v in pairs(test_set) do
		local gold = v[1]	--正确答案的集合
		local qst = v[2]	--问题
		local candidates = v[3] --候选的答案
		
		local qidx = get_index(qst):cuda()
		local qemd = lm.emd:forward(qidx):clone()
		local qvec = lm.qlstm:forward(qemd)
		
		local sc = {}	
		local gold_sc ={}
		local gold_rank = {}
		
		for k,c in pairs(gold) do 
			c =tostring(tonumber(c))
			local score = test_one_pair(qvec,answer_set[c])	--标准答案的得分
			gold_sc[k] = score
			gold_rank[k] = 1	--初始化排名
		end
		thr = 20
		for k,c in pairs(candidates) do 
			thr = thr -1
			if thr ==0 then break end
			c =tostring(tonumber(c))
			local score = test_one_pair(qvec,answer_set[c])
			for m,n in pairs(gold_sc) do
		
				if score > n then
					gold_rank[m] = gold_rank[m]+1
				end
			end
		end
		
		local mark =false
		local mrr = 0
		for k,c in pairs(gold_rank) do
			if c==1 then 
				mark = true
			end
			mrr = mrr + 1.0/c
		end
		print(mrr,i)
		if mark then 
			results[i] = {mrr,1.0}
		else
			results[i] = {mrr,0.0}
		end

	end
	local results = torch.Tensor(results)
	print(torch.sum(results,1)/results:size()[1])
end
train()
evaluate('dev')

