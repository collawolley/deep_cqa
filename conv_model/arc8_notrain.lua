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
cfg.batch  = 5-- or deep_cqa.config.batch_size
cfg.gpu = true
deep_cqa.ins_meth.load_binary()	--保险数据集，这里载入是为了获得测试集和答案
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
------------------------------------------------------------------------
function getlm()
	get_index('today is')
-------------------------------------
	local qcov = nn.SpatialConvolution(1,1000,100,2,10,1,0,1)	--input需要是3维tensor
	local tcov = qcov:clone('weights','bias')
	local fcov = qcov:clone('weights','bias')
	--share_params(qcov,tcov)
	--share_params(qcov,fcov)
-------------------------------------
	local pt = nn.Sequential()
	pt:add(nn.SpatialAdaptiveMaxPooling(1,1))
	pt:add(nn.Reshape(1000))
	pt:add(nn.Tanh())
-------------------------------------
	local hlq = nn.Linear(cfg.dim,200)
	local hlt = hlq:clone('weights','bias')
	local hlf = hlq:clone('weights','bias')
	--share_params(hlq,hlt)
	--share_params(hlq,hlf)

-------------------------------------
	local lm = {}	--待返回的语言模型
	lm.qemd = cfg.emd	--词嵌入部分
	lm.temd = lm.qemd:clone('weights','bias')
	lm.femd = lm.qemd:clone('weights','bias')
	lm.qst = nn.Sequential()
	lm.tas = nn.Sequential()
	lm.fas = nn.Sequential()

	lm.qst:add(hlq)
	lm.tas:add(hlt)
	lm.fas:add(hlf)
	---------------------
	lm.qst:add(nn.Replicate(1))
	lm.tas:add(nn.Replicate(1))
	lm.fas:add(nn.Replicate(1))
	---------------------
	lm.qst:add(qcov)
	lm.tas:add(tcov)
	lm.fas:add(fcov)
	---------------------
	lm.qst:add(pt:clone())
	lm.tas:add(pt:clone())
	lm.fas:add(pt:clone())
	----------------------
	lm.qt = nn.CosineDistance()	--nn包里的cosine distance实际上计算方式为wiki的cosine similarity
	lm.qf = nn.CosineDistance()
	lm.sub = nn.PairwiseDistance(1)
-------------------------------
	if cfg.gpu then
		lm.qemd:cuda()
		lm.temd:cuda()
		lm.femd:cuda()
		lm.qst:cuda()
		lm.tas:cuda()
		lm.fas:cuda()
		lm.qt:cuda()
		lm.qf:cuda()
		lm.sub:cuda()
	end
-------------------------------
	return lm

end
function testlm()	--应用修改模型后测试模型是否按照预期执行
	local lm = getlm()
	local criterion = nn.MarginCriterion(1)
	local gold = torch.Tensor({1})
	
	local index1 = get_index('today is a good day'):clone()
	local index2 = get_index('today is a very good day'):clone()
	local index3 = get_index('This class creates an output where the input is replicated'):clone()
	if cfg.gpu then
		criterion:cuda()
		gold =gold:cuda()
		index1 = index1:cuda()
		index2 = index2:cuda()
		index3 = index3:cuda()
	end
	local vec1 = lm.emd:forward(index1):clone()
	local vec2 = lm.emd:forward(index2):clone()
	local vec3 = lm.emd:forward(index3):clone()
	
	local hl = nn.Linear(cfg.dim,200)
	local q =  lm.qst:forward(vec1)
	local t =  lm.tas:forward(vec2)
	local f =  lm.fas:forward(vec3)
	local qt = lm.qt:forward({q,t})
	local qf = lm.qf:forward({q,f})
	local sub = lm.sub:forward({qf,qt})
	print(qt,qf,sub)
end
--------------------------
cfg.lm = getlm()
-------------------------
function train()
	local lm = cfg.lm
	
	local modules = nn.Parallel()
	--modules:add(lm.qemd)
	--modules:add(lm.temd)
	--modules:add(lm.femd)
	modules:add(lm.qst)
	modules:add(lm.tas)
	modules:add(lm.fas)
	modules:add(lm.tq)
	modules:add(lm.fq)
	modules:add(lm.sub)
	params,grad_params = modules:getParameters()

	local train_set = torch.load(deep_cqa.ins_meth.train)
	local indices = torch.randperm(train_set.size)
	local criterion = nn.MarginCriterion(0.009)
	local gold = torch.Tensor({1})
	if cfg.gpu then
		criterion:cuda()
		gold = gold:cuda()
	end
	local batch_size = cfg.batch
	local learningRate = 0.01
--	train_set.size =5000
	for i= 1,train_set.size do
		xlua.progress(i,train_set.size)
		local idx = indices[i]
		local sample = train_set[idx]
--		print('\n',sample)
		local vecs={}
		
		local indexq = get_index(sample[1]):clone()
		local indext = get_index(sample[2]):clone()
		local indexf = get_index(sample[3]):clone()
		if(cfg.gpu) then
			 indexq = indexq:cuda() 
			 indext = indext:cuda() 
			 indexf = indexf:cuda() 
		end
		vecs[1] = lm.qemd:forward(indexq):clone()
		vecs[2] = lm.temd:forward(indext):clone()
		vecs[3] = lm.femd:forward(indexf):clone()
		
		if i %2 == 0 then 
			vecs[2],vecs[3] = vecs[3],vecs[2]
			gold[1] = -1
		else
			gold[1] = 1
		end
		
		local rep1 = lm.qst:forward(vecs[1])
		local rep2 = lm.tas:forward(vecs[2])
		local rep3 = lm.fas:forward(vecs[3])
				
		local sc_1 = lm.qt:forward({rep1,rep2})
		local sc_2 = lm.qf:forward({rep1,rep3})
		local pred = lm.sub:forward({sc_1,sc_2})	-- 因为是距离参数转换为相似度参数，所以是负样本减正样本
				
		criterion:forward(pred,gold)

		lm.sub:zeroGradParameters()
		lm.qt:zeroGradParameters()
		lm.qf:zeroGradParameters()
		lm.qst:zeroGradParameters()
		lm.tas:zeroGradParameters()
		lm.fas:zeroGradParameters()
		lm.qemd:zeroGradParameters()
		lm.temd:zeroGradParameters()
		lm.femd:zeroGradParameters()
--		print('forward done')
					
		local e1 = criterion:backward(pred,gold)
		local e2 = lm.sub:backward({sc_1,sc_2},e1)
		local e3 = lm.qt:backward({rep1,rep2},e2[1])
		local e4 = lm.qf:backward({rep1,rep3},e2[2])
		
		local e5 = lm.qst:backward(vecs[1],(e3[1]+e4[1]))
		local e7 = lm.tas:backward(vecs[2],e3[2])
		local e8 = lm.fas:backward(vecs[3],e4[2])
--		print('准备更新参数')	
		lm.sub:updateParameters(learningRate)
		lm.qt:updateParameters(learningRate)
		lm.qf:updateParameters(learningRate)
		lm.qst:updateParameters(learningRate)
		lm.tas:updateParameters(learningRate)
		lm.fas:updateParameters(learningRate)
--[	
--		print('q',#indexq)
		lm.qemd:backward(indexq,e5)
		lm.qemd:updateParameters(learningRate)
--		print('t',#indext)
		lm.temd:backward(indext,e7)
		lm.temd:updateParameters(learningRate)
--		print('f',#indexf)
		lm.femd:backward(indexf,e8)
		lm.femd:updateParameters(learningRate)
--]
	end
end
------------------------------------------------------------------------
function test_one_pair(qst,ans)
	--给定一个问答pair，计算其相似度	
	--传入的qst为已经计算好的向量，ans为未经处理的句子
--[
	local lm = cfg.lm
	local aidx = get_index(ans)
	if cfg.gpu then aidx = aidx:cuda() end
	local aemd = lm.temd:forward(aidx):clone()
	local arep = lm.tas:forward(aemd)
	local sim_sc = lm.qt:forward({qst,arep})
	return sim_sc[1]
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
	print('test process:')
	for i,v in pairs(test_set) do
		xlua.progress(i,1000)
		local gold = v[1]	--正确答案的集合
		local qst = v[2]	--问题
		local candidates = v[3] --候选的答案
		local qidx = get_index(qst)
		if cfg.gpu then qidx = qidx:cuda() end
		local qemd = lm.qemd:forward(qidx):clone()
		local qvec = lm.qst:forward(qemd)
		
		local sc = {}	
		local gold_sc ={}
		local gold_rank = {}
		
		for k,c in pairs(gold) do 
			c =tostring(tonumber(c))
			local score = test_one_pair(qvec,answer_set[c])	--标准答案的得分
			gold_sc[k] = score
			gold_rank[k] = 1	--初始化排名
		end
		for k,c in pairs(candidates) do 
			c =tostring(tonumber(c))
			local score = test_one_pair(qvec,answer_set[c])
			for m,n in pairs(gold_sc) do
				if score > n then
					gold_rank[m] = gold_rank[m]+1
				end
			end
		end
		
		local mark =0.0
		local mrr = 0
		for k,c in pairs(gold_rank) do
			if c==1 then 
				mark = 1.0
			end
			mrr = mrr + 1.0/c
		end
		results[i] = {mrr,mark}

--	if i >99 then break end
	end
	local results = torch.Tensor(results)
	print(torch.sum(results,1)/results:size()[1])
end
--getlm()
--testlm()
train()
evaluate('dev')

