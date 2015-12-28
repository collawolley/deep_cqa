--[[
	使用rnn的LSTM代码实现正确答案和错误答案的分类	
	author:	liangjz
	time:	2015-12-28
--]]
require('..')
local cfg = {}
cfg.vecs = nil
cfg.dict = nil
cfg.emd = nil
cfg.dim = deep_cqa.config.emd_dim
cfg.mem = 100
cfg.batch  = 10
deep_cqa.ins_meth.load_binary()
-----------------------------------
function get_index(sent)
	--	获取一个句子的索引表达，作为整个模型的输入，可以直接应用到词向量层
	if cfg.dict == nil then		--	载入字典和词向量查询层
		cfg.dict, cfg.vecs = deep_cqa.get_sub_embedding()
		cfg.emd = nn.LookupTable(cfg.vecs:size(1),cfg.dim)
		cfg.emd.weight:copy(cfg.vecs)
		cfg.vecs = nil
	end
	return deep_cqa.read_one_sentence(sent,cfg.dict)
end
------------------------------------------------------------------------
function getlm()
	get_index('today is')
	local lm = {}
	lm.emd = cfg.emd:cuda()
		
	local lstm = nn.Sequential()
	lstm:add(nn.Identity())
	lstm:add(nn.SplitTable(1))
	lstm:add(nn.Sequencer(nn.FastLSTM(cfg.dim,cfg.mem)))
	lstm:add(nn.SelectTable(-1))
	lstm:cuda()


	lm.qlstm = lstm:clone()
	lm.tlstm = lstm:clone()
	lm.flstm = lstm:clone()
	lm.flstm:share(lm.tlstm, 'weight', 'bias')

	lm.sub = nn.CSubTable():cuda()
	local score = nn.Sequential()
	score:add(nn.DotProduct())
	score:add(nn.SoftSign()):cuda()
	lm.score = score
	return lm
end

cfg.lm = getlm()

-------------------------
function train()
	local lm = cfg.lm
	
	local train_set = torch.load(deep_cqa.ins_meth.train)
	local indices = torch.randperm(train_set.size)
	local criterion = nn.MarginCriterion(0.5):cuda()
	local gold = torch.Tensor({0.5}):cuda()
	local batch_size = cfg.batch
	local learningRate = 0.01
--	train_set.size =20
	for i= 1,train_set.size do
		xlua.progress(i,train_set.size)
		local idx = indices[i]
		local sample = train_set[idx]
		local vecs={}
		for k =1,#sample do
			local index = get_index(sample[k]):clone():cuda()
			vecs[k] = lm.emd:forward(index):clone()
		end
		if(idx %2 ==0) then
			vecs[3],vecs[2] = vecs[2],vecs[3]
			gold[1] = -0.5
		else
			gold[1] = 0.5
		end
		
				
		local r1 = lm.qlstm:forward(vecs[1])
		local r2 = lm.tlstm:forward(vecs[2])
		local r3 = lm.flstm:forward(vecs[3])
		local r4 = lm.sub:forward({r2,r3})
		local pred = lm.score:forward({r4,r1})

		criterion:forward(pred,gold)
		
		lm.score:zeroGradParameters()
		lm.sub:zeroGradParameters()
		lm.qlstm:zeroGradParameters()
		lm.tlstm:zeroGradParameters()
		lm.flstm:zeroGradParameters()

		local e1 = criterion:backward(pred,gold)
		local e2 = lm.score:backward({r4,r1},e1)
		local e3 = lm.sub:backward({r2,r3},e2[1])
		local e4 = lm.qlstm:backward(vecs[1],e2[2])
		local e5 = lm.tlstm:backward(vecs[2],e3[1])
		local e6 = lm.flstm:backward(vecs[3],e3[2])

		lm.score:updateParameters(learningRate)
		lm.sub:updateParameters(learningRate)
		lm.qlstm:updateParameters(learningRate)
		lm.tlstm:updateParameters(learningRate)
		lm.flstm:updateParameters(learningRate)
				
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
	local r5 = lm.score:forward({qst,avec})

	return r5[1]
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
		xlua.progress(i,1000)
		local gold = v[1]	--正确答案的集合
		local qst = v[2]	--问题
		local candidates = v[3] --候选的答案
		
		local qidx = get_index(qst):clone():cuda()
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

