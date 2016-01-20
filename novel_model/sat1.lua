--[[
	使用双向lstm验证一下bilstm+maxpooling是否有作用
	单个负样本训练大概需要5小时，每次测试大概需要2个小时
	autor: liangjz
	2016-1-18
--]]
local Sat1 = torch.class('Sat1')
function Sat1:__init(useGPU)
	self.cfg = {
		vecs	= nil,
		dict	= nil,
		emd	= nil,
		dim	= deep_cqa.config.emd_dim,	--词向量的维度
		mem	= 100,	--Memory的维度
		gpu	= useGPU or false,	--是否使用gpu模式
		margin	= 0.1,
		l2Rate	= 0,	--L2范式的约束
		learningRate	= 0.1	--L2范式的约束
	}
	self.cfg.dict, self.cfg.vecs = deep_cqa.get_sub_embedding()
	self.cfg.emd = nn.LookupTable(self.cfg.vecs:size(1),self.cfg.dim)
	self.cfg.emd.weight:copy(self.cfg.vecs)
	self.cfg.vecs = nil
	
	self.LM = {}
	self:getLM()
	self.dataSet = InsSet()	--保险数据集
end

function Sat1:getLM()
	self.LM = {}	-- 清空语言模型
	lm = self.LM	-- 简写
	cfg = self.cfg	-- 简写

	lm.qemd = cfg.emd	--直接用这个词向量层就好
	lm.temd	= lm.qemd:clone('weight','bias')	--共享权重和偏置
	lm.femd	= lm.qemd:clone('weight','bias')	--共享权重和偏置
	
	local flstm = nn.FastLSTM(cfg.dim,cfg.mem)
	local rlstm = nn.FastLSTM(cfg.dim,cfg.mem)
	lm.qlstm = nn.BiSequencer(flstm:clone('weight','bias'),rlstm:clone('weight','bias'))
	lm.tlstm = nn.BiSequencer(flstm:clone('weight','bias'),rlstm:clone('weight','bias'))
	lm.flstm = nn.BiSequencer(flstm:clone('weight','bias'),rlstm:clone('weight','bias'))
	
	lm.pip = nn.Sequential()
	lm.pip:add(nn.JoinTable(1))
	lm.pip:add(nn.View(-1,cfg.mem*2))
	lm.pip:add(nn.Replicate(1))
	lm.pip:add(nn.SpatialAdaptiveMaxPooling(cfg.mem*2,1))
	lm.pip:add(nn.Reshape(cfg.mem*2))

	lm.qst = nn.Sequential()
	lm.qst:add(nn.SplitTable(1))
	lm.qst:add(lm.qlstm)
	lm.qst:add(lm.pip:clone())
	
	lm.tas = nn.Sequential()
	lm.tas:add(nn.SplitTable(1))
	lm.tas:add(lm.tlstm)
	lm.tas:add(lm.pip:clone())

	lm.fas = nn.Sequential()
	lm.fas:add(nn.SplitTable(1))
	lm.fas:add(lm.flstm)
	lm.fas:add(lm.pip:clone())

	lm.qt = nn.CosineDistance()
	lm.qf = nn.CosineDistance()
	lm.sub = nn.CSubTable()
	
	if self.cfg.gpu then
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
	
end
function Sat1:getIndex(sent) --	获取一个句子的索引表达，作为整个模型的输入，可以直接应用到词向量层
	return deep_cqa.read_one_sentence(sent,self.cfg.dict)
end

function Sat1:testLM()
	
	lstm = nn.FastLSTM(self.cfg.dim,self.cfg.mem)
	local index1 = self:getIndex('today is a good day'):clone()
	local index2 = self:getIndex('today is a very good day'):clone()
	local index3 = self:getIndex('This class creates an output where the input is replicated'):clone()
	local rep1 = self.LM.qemd:forward(index1):clone()
	local rep2 = self.LM.qemd:forward(index2):clone()
	local rep3 = self.LM.qemd:forward(index3):clone()
	
	local m1 = nn.SplitTable(1)
	local m2 = nn.BiSequencer(lstm,lstm:clone('weight','bias'))
	local m3 = nn.JoinTable(1)
	print(m3)
	local m4 = nn.View(-1,cfg.mem*2)
	print(m4)
	local m5 = nn.Replicate(1)
	local m6 = nn.SpatialAdaptiveMaxPooling(cfg.mem*2,1)
	local m7 = nn.Reshape(cfg.mem*2)
	local m8 = nn.CosineDistance()
	
	local t1 = m1:forward(rep1)
	local t2 = m2:forward(t1)
	print('t2',t2)
	local t3 = m3:forward(t2)
	print('t3',t3)
	local t4 = m4:forward(t3)
	print('view',t4)
	local t5 = m5:forward(t4)
	print('replicate',t5)
	local t6 = m6:forward(t5)
	print('maxpooling',t6)
	local t7 = m7:forward(t6)
	local t8 = m8:forward({t7,t7})

	local e1 = m8:backward({t7,t7},torch.Tensor({0.5}))
	print(e1[1],e1[2])
	local e2 = m7:backward(t6,e1[1])
	print('e2',e2)
	local e3 = m6:backward(t5,e2)
	print('e3',e3)
	local e4 = m5:backward(t4,e3)
	print('e4',e4)
	local e5 = m4:backward(t3,e4)
	print('e5',e5)
	local e6 = m3:backward(t2,e5)
	print('e6',e6[1],e6[2],e6[3],e6[4],e6[5])
	local e7 = m2:backward(t1,e6)
	print('e7',e7)
--[[

	local r1 = nn.SplitTable(1):forward(rep1)
	local r2 = nn.BiSequencer(lstm,lstm:clone('weight','bias')):forward(r1)
	local r3 = nn.JoinTable(1):forward(r2)
	local r4 = nn.View(-1,cfg.mem*2):forward(r3)
	local r5 = nn.Replicate(1):forward(r4)
	local r6 = nn.SpatialAdaptiveMaxPooling(cfg.mem*2,1):forward(r5)

	local qst = self.LM.qst:forward(rep1)
	local tas = self.LM.tas:forward(rep2)
	local fas = self.LM.fas:forward(rep3)
	local qt  = self.LM.qt:forward({qst,tas})
	local qf  = self.LM.qf:forward({qst,fas})
	local sub = self.LM.sub:forward({qt,qf})
	print(qst,tas,fas)
	print(qt[1],qf[1],sub[1])

--]]
end

function Sat1:train(negativeSize)
	self.dataSet:resetTrainset(negativeSize)
	local modules = nn.Parallel()
	modules:add(self.LM.qst)
	modules:add(self.LM.tas)
	modules:add(self.LM.fas)
	modules:add(self.LM.qt)
	modules:add(self.LM.qf)
	modules:add(self.LM.sub)
	params,grad_params = modules:getParameters()
	
	local criterion = nn.MarginCriterion(self.cfg.margin)
	local gold = torch.Tensor({1})
	if self.cfg.gpu then
		criterion:cuda()
		gold = gold:cuda()
	end
	local sample =1	--占个坑
	local vecs={}	--存储词向量
	local index ={}	--存储字符下标
	local loop = 0

	local sample_count = 0
	local right_sample = 0
	while sample ~= nil do	--数据集跑完？
		loop = loop + 1
		if loop %1 ==0 then xlua.progress(self.dataSet.current_train,self.dataSet.train_set.size) end
		sample = self.dataSet:getNextPair()
		if sample == nil then break end	--数据集获取完毕
		index[1] = self:getIndex(sample[1]):clone()
		index[2] = self:getIndex(sample[2]):clone()
		index[3] = self:getIndex(sample[3]):clone()
		
		if loop % 2  == 0 then
			index[2],index[3] = index[3],index[2]
			gold[1] = -1
		else
			gold[1] = 1
		end

		if(self.cfg.gpu) then
			index[1] = index[1]:cuda() 
		 	index[2] = index[2]:cuda() 
		 	index[3]= index[3]:cuda() 
		end
		
		vecs[1] = self.LM.qemd:forward(index[1]):clone()
		vecs[2] = self.LM.temd:forward(index[2]):clone()
		vecs[3] = self.LM.femd:forward(index[3]):clone()	
		
		local rep1 = self.LM.qst:forward(vecs[1])
		local rep2 = self.LM.tas:forward(vecs[2])
		local rep3 = self.LM.fas:forward(vecs[3])
		local sc_1 = self.LM.qt:forward({rep1,rep2})
		local sc_2 = self.LM.qf:forward({rep1,rep3})
		local pred = self.LM.sub:forward({sc_1,sc_2})	-- 因为是距离参数转换为相似度参数，所以是负样本减正样本
		local err = criterion:forward(pred,gold)
		sample_count = sample_count + 1
		if err <= 0 then
			right_sample = right_sample + 1
		end
		
		self.LM.sub:zeroGradParameters()
		self.LM.qt:zeroGradParameters()
		self.LM.qf:zeroGradParameters()
		self.LM.qst:zeroGradParameters()
		self.LM.tas:zeroGradParameters()
		self.LM.fas:zeroGradParameters()
		self.LM.qemd:zeroGradParameters()
		self.LM.temd:zeroGradParameters()
		self.LM.femd:zeroGradParameters()
		
		local e0 = criterion:backward(pred,gold)
		e1 = e0  + self.cfg.l2Rate*0.5*params:norm()^2	--二阶范数
		local e2 = self.LM.sub:backward({sc_1,sc_2},e1)
		local e3 = self.LM.qt:backward({rep1,rep2},e2[1])
		local e4 = self.LM.qf:backward({rep1,rep3},e2[2])
		local e5 = self.LM.qst:backward(vecs[1],(e3[1]+e4[1])/2)
		local e7 = self.LM.tas:backward(vecs[2],e3[2])
		local e8 = self.LM.fas:backward(vecs[3],e4[2])
		local learningRate  = self.cfg.learningRate
		self.LM.sub:updateParameters(learningRate)
		self.LM.qt:updateParameters(learningRate)
		self.LM.qf:updateParameters(learningRate)
		self.LM.qst:updateParameters(learningRate)
		self.LM.tas:updateParameters(learningRate)
		self.LM.fas:updateParameters(learningRate)
		self.LM.qlstm:forget()
		self.LM.tlstm:forget()
		self.LM.flstm:forget()
		
		self.LM.qemd:backward(index[1],e5)
		self.LM.qemd:updateParameters(learningRate)
		self.LM.temd:backward(index[2],e7)
		self.LM.temd:updateParameters(learningRate)
		self.LM.femd:backward(index[3],e8)
		self.LM.femd:updateParameters(learningRate)
	end
	print('训练集的准确率：',right_sample/sample_count)
end

function Sat1:test_one_pair(question_vec,answer_id) 	--给定一个问答pair，计算其相似度	
	--传入的qst为已经计算好的向量，ans为问题的id
	local answer_rep = self.dataSet:getAnswerVec(answer_id)	--获取答案的表达
	--print('ans_rep',answer_id,answer_rep[1][1])
	if self.cfg.gpu then
		answer_rep = answer_rep:cuda()
	end
	local sim_sc = self.LM.qt:forward({question_vec,answer_rep})
	return sim_sc[1]
end


function Sat1:evaluate(name)
	local results = {}	--记录每个测试样本的结果
	local test_size = 0
	local loop = 0
	if name =='dev' then 
		test_size = self.dataSet.dev_set.size 
	else
		test_size = self.dataSet.test_set.size 
	end
	print('\nCalculating answers')
	local answer_pair =self.dataSet:getNextAnswer(true)	--从头开始计算answer的向量
	while answer_pair~=nil do
		loop = loop+1
		xlua.progress(loop,self.dataSet.answer_set.size)
		local answer = answer_pair[2]	--获取问题内容
		local word_index = self:getIndex(answer)	--获取词下标
		if self.cfg.gpu then word_index = word_index:cuda() end
		local answer_emd = self.LM.temd:forward(word_index):clone()
		local answer_rep = self.LM.tas:forward(answer_emd):clone()
		self.dataSet:saveAnswerVec(answer_pair[1],answer_rep)
		answer_pair = self.dataSet:getNextAnswer()
	end	
	collectgarbage() 
	print('Test process:')
	local test_pair =nil
	if name =='dev' then
		test_pair = self.dataSet:getNextDev(true)
	else
		test_pair = self.dataSet:getNextTest(true)
	end
	loop = 0
	while test_pair~=nil do
		loop = loop+1
		xlua.progress(loop,test_size)

		local gold = test_pair[1]	--正确答案的集合
		local qst = test_pair[2]	--问题
		local candidates = test_pair[3] --候选的答案
		local qst_idx = self:getIndex(qst)
		if self.cfg.gpu then qst_idx = qst_idx:cuda() end
		local qst_emd = self.LM.qemd:forward(qst_idx):clone()
		local qst_vec = self.LM.qst:forward(qst_emd):clone()

		local sc = {}	
		local gold_sc ={}
		local gold_rank = {}
		
		for k,c in pairs(gold) do 
			local score = self:test_one_pair(qst_vec,c)	--标准答案的得分,传入内容为问题的表达和答案的编号
			gold_sc[k] = score
			gold_rank[k] = 1	--初始化排名
		end

		for k,c in pairs(candidates) do 
			local score = self:test_one_pair(qst_vec,c)
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
		results[loop] = {mrr,mark}
		if name =='dev' then
			test_pair = self.dataSet:getNextDev()
		else
			test_pair = self.dataSet:getNextTest()
		end

		if loop%10==0 then collectgarbage() end
	end

	local results = torch.Tensor(results)
	print('Results:',torch.sum(results,1)/results:size()[1])
end




