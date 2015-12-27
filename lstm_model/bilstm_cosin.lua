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
cfg.batch  = deep_cqa.config.batch_size
-- deep_cqa.ins_meth.load_binary()	--保险数据集，这里载入是为了获得测试集和答案
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
	------------------------------------
	local lstm = nn.Sequential()	--包装，顺序执行的网络
	lstm:add(nn.Identity())	--原样送出
	lstm:add(nn.SplitTable(1))	--tensor分割成为table，调整输出格式方便使用lstm
	lstm:add(nn.BiSequencerLM(nn.FastLSTM(cfg.dim,cfg.mem)))
	lstm:add(nn.JoinTable(1))
	lstm:add(nn.View(-1,cfg.mem*2))	--双向的lstm输出维度是内部维度的2倍

-------------------------------------
	--下面是cos sim的计算nn图
	local m1 = nn.ConcatTable()
	m1:add(nn.Identity())
	m1:add(nn.Identity())
	local m2 = nn.Sequential()
	m2:add(m1)
	m2:add(nn.CMulTable())
	m2:add(nn.Sum(2))
	m2:add(nn.Reshape(1))
	local m3 = m2:clone()
	local m4 = nn.ParallelTable()
	m4:add(m2)
	m4:add(m3)
	local m5 = nn.Sequential()
	m5:add(m4)
	m5:add(nn.MM(false,true))
	m5:add(nn.Sqrt())
	local m6 = nn.ConcatTable():add(nn.MM(false,true)):add(m5)
	local m7 = nn.Sequential():add(m6):add(nn.CDivTable())
	local cosine = m7
	-- cos 相似度模块构建完成
------------------------------------

	local cov = nn.Sequential()
	cov:add(nn.Replicate(1))	--增加维度的功能，好不容易才发现
	cov:add(nn.SpatialConvolution(1,20,3,3,1,1,2,2))	--input需要是3维tensor
	cov:add(nn.SpatialAdaptiveMaxPooling(1,1))
	cov:add(nn.Reshape(20))
	cov:add(nn.Tanh())
	
	local mlp = nn.Sequential()	--一个简单的感知机
	mlp:add(nn.Linear(20,3))
	mlp:add(nn.Tanh())
	mlp:add(nn.Linear(3,1))
	mlp:add(nn.SoftSign())	--输出为归一化后的评分
-------------------------------------
	local lm = {}	--待返回的语言模型
	lm.emd = cfg.emd:cuda()	--词嵌入部分
-------------------------------
	lm.qlstm = lstm:clone():cuda()	--问题部分的bilstm模型
	lm.tlstm = lstm:clone():cuda()	--两种答案的bilstm模型
	lm.flstm = lstm:clone():cuda()
	share_params(lm.tlstm,lm.flstm)
-------------------------------
	lm.tq_cos = cosine:clone():cuda()	--这个模型内部没有参数，所以无需共享也是相同的
	lm.fq_cos = cosine:clone():cuda() 
	
	lm.tq_cov = cov:clone():cuda()	--卷积模型+全图最大值pooling
	lm.fq_cov = cov:clone():cuda()
	share_params(lm.tq_cov,lm.fq_cov)

	lm.tq_mlp = mlp:clone():cuda()	--两个模型的多层感知机层
	lm.fq_mlp = mlp:clone():cuda()
	share_params(lm.tq_mlp,lm.fq_mlp)
	
	lm.tq = nn.Sequential():add(lm.tq_cos):add(lm.tq_cov):add(lm.tq_mlp):cuda()
	lm.fq = nn.Sequential():add(lm.fq_cos):add(lm.fq_cov):add(lm.fq_mlp):cuda()
--------------------------------
	lm.sub = nn.PairwiseDistance(1):cuda()

	return lm

end
function testlm()
	local ml = getlm()
	local index1 = get_index('today is a good day'):clone()
	local index2 = get_index('today is a very good day'):clone()
	local index3 = get_index('This class creates an output where the input is replicated'):clone()
	local vec1 = ml.emd:forward(index1):clone()
	local vec2 = ml.emd:forward(index2):clone()
	local vec3 = ml.emd:forward(index3):clone()
	local rep1 = ml.qlstm:forward(vec1)
	local rep2 = ml.tlstm:forward(vec2)
	local rep3 = ml.flstm:forward(vec3)
	--print(rep1:size(),rep2:size(),rep3:size())
	local cos1 = ml.tq_cos:forward({rep1,rep2})
	local cos2 = ml.fq_cos:forward({rep1,rep3})
	local cov1= ml.tq_cov:forward(cos1)
	local cov2= ml.fq_cov:forward(cos2)
	print(cov1,cov2)
	print(ml.tq_mlp:forward(cov1))
	print(ml.fq_mlp:forward(cov2))
	print('-------------------')
	local r5 = ml.tq:forward({rep1,rep2})
	local r6 = ml.fq:forward({rep1,rep3})
	print(r5,r6)
	local sub = ml.sub:forward({r5,r6})
	print(sub)
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
getlm()
testlm()
--train()
--evaluate('dev')

