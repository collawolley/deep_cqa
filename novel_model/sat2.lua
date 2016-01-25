--[[
	使用 tree-lstm 实现的lstm作为bi-lism的底层实现
	autor: liangjz
	2016-1-23
--]]
local Sat = torch.class('Sat2')
function Sat:__init(useGPU)
	self.cfg = {
		vecs	= nil,
		dict	= nil,
		emd	= nil,
		dim	=  deep_cqa.config.emd_dim,	--词向量的维度
		mem	= 10,	--Memory的维度
		gpu	= useGPU or false,	--是否使用gpu模式
		dp = 0.5,
		margin	= 0.1,
		l2Rate	= 0,	--L2范式的约束
		learningRate	= 0.1	--L2范式的约束
	}
	self.cfg.dict, self.cfg.vecs = deep_cqa.get_sub_embedding()
	self.cfg.emd = nn.LookupTable(self.cfg.vecs:size(1),self.cfg.dim)
	self.cfg.emd.weight:copy(self.cfg.vecs)
	self.cfg.vecs = nil
	self.lstm_cfg = {
		in_dim = self.cfg.dim,
    	mem_dim = self.cfg.mem,
	    num_layers = 1,
	    gate_output = true,		
		cuda = gpu
	}
	self.LM = {}
	self:getLM()
	self.dataSet = InsSet()	--保险数据集
end
function Sat:getConv()
	local cfg = self.cfg
	local i1 = nn.Identity()()	--正向序列
	local i2 = nn.Identity()()	--反向序列首先需要将输出序列进行反转
	local j1 = nn.JoinTable(1)(i1)
	local j2 = nn.JoinTable(1)(i2)
	local j3 = nn.JoinTable(1)({j1,j2})
	local v = nn.View(-1,cfg.mem*2)(j3)
	local r1 = nn.Replicate(1)(v)
	local s = nn.SpatialAdaptiveMaxPooling(cfg.mem*2,1)(r1)
	local r2 = nn.Reshape(cfg.mem*2)(s)
	local conv = nn.gModule({i1,i2},{r2})
	if cfg.gpu then
		conv:cuda()
	end
	return conv
end

function Sat:getLM()
	self.LM = {}	-- 清空语言模型
	local lm = self.LM	-- 简写
	local cfg = self.cfg	-- 简写
	lm.emd ={}
	lm.fwd = {}	
	lm.rwd = {}									
	lm.input = {}
	lm.rep = {}
	lm.join_1= {}
	lm.join_2= {}
	lm.view  = {}
	lm.replicate = {}
	lm.reshape = {}
	lm.max = {}
	for i =1,3 do
		lm.emd[i] = cfg.emd:clone('weight','bias')
		lm.fwd[i] = deep_cqa.LSTM(self.lstm_cfg)	--三个正向序列
		lm.rwd[i] = deep_cqa.LSTM(self.lstm_cfg)	--三个反向序列
	end
	for i = 1,3 do
		lm.input[i*2-1] = nn.Identity()()	--正向lstm序列的输出
		lm.input[i*2] = nn.Identity()()	--正向lstm序列的输出
		lm.join_1[i*2-1] = nn.JoinTable(1)(lm.input[i*2-1])		--将输入合并成为tensor
		lm.join_1[i*2] = nn.JoinTable(1)(lm.input[i*2])		--将输入合并成为tensor
		lm.view[2*i-1] = nn.View(-1,cfg.mem)(lm.join_1[2*i-1])	
		lm.view[2*i] = nn.View(-1,cfg.mem)(lm.join_1[2*i])
		lm.join_2[i] = nn.JoinTable(2)({lm.view[2*i-1],lm.view[2*i]})
		lm.replicate[i] = nn.Replicate(1)(lm.join_2[i])
		lm.max[i] = nn.SpatialAdaptiveMaxPooling(cfg.mem*2,1)(lm.replicate[i])
		lm.reshape[i] = nn.Reshape(cfg.mem*2)(lm.max[i])
		lm.rep[i] = nn.Identity()(lm.reshape[i])
	end
	lm.md1 = nn.gModule(lm.input,lm.rep)
	
	local cos_in ={}
	for i = 1, 3 do 	
		cos_in[i] = nn.Identity()()
	end
	lm.cosine = {}
	lm.cosine[1]  = nn.CosineDistance()({cos_in[1],cos_in[2]})
	lm.cosine[2]  = nn.CosineDistance()({cos_in[1],cos_in[3]})
	lm.sub = nn.CSubTable()(lm.cosine)
	lm.md2 = nn.gModule(cos_in,{lm.sub})
	lm.dp = nn.Dropout(cfg.dp)
	if cfg.gpu then
		lm.md1:cuda()
		lm.md2:cuda()
		lm.dp:cuda()
	end
end
function Sat:getIndex(sent) --	获取一个句子的索引表达，作为整个模型的输入，可以直接应用到词向量层
	return deep_cqa.read_one_sentence(sent,self.cfg.dict)
end

function Sat:testLM()
	local index = {}
	local wdvec = {}	
	local lstm ={}
	index[1] = self:getIndex('today is a good day'):clone()
	index[2] = self:getIndex('day good a is today'):clone()
	index[3] = self:getIndex('This class creates an output where the input is replicated'):clone()
	for i = 1,3 do
		wdvec[i] = self.LM.emd[i]:forward(index[i])
		lstm[2*i-1] = self.LM.fwd[i]:forward(wdvec[i])
		lstm[2*i] = self.LM.rwd[i]:forward(wdvec[i],true)
	end
	print(lstm[1][1],lstm[1][2],lstm[1][3],lstm[1][4],lstm[1][5])
	local o = self.LM.md1:forward(lstm)
	print(o)
	local score = self.LM.md2:forward(o)
	print(score)
end

function Sat:train(negativeSize)
	self.dataSet:resetTrainset(negativeSize)
	local modules = nn.Parallel()
	for i =1,3 do 	
		modules:add(self.LM.fwd[i])
		modules:add(self.LM.rwd[i])
	end
	modules:add(self.LM.md1)
	modules:add(self.LM.md2)
	params,grad_params = modules:getParameters()
	share_params(self.LM.fwd[2], self.LM.fwd[1])
	share_params(self.LM.fwd[3], self.LM.fwd[1])
	share_params(self.LM.rwd[2], self.LM.rwd[1])
	share_params(self.LM.rwd[3], self.LM.rwd[1])
	self.LM.dp:training()

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
		collectgarbage() 
		if loop > 3000 then break end
		sample = self.dataSet:getNextPair()
		if sample == nil then break end	--数据集获取完毕
		index[1] = self:getIndex(sample[1]):clone()
		index[2] = self:getIndex(sample[2]):clone()
		index[3] = self:getIndex(sample[3]):clone()
--[
		if loop % 2  == 0 then
			index[2],index[3] = index[3],index[2]
			gold[1] = -1
		else
			gold[1] = 1
		end
--]
		local mask =torch.ones(self.cfg.mem*2)	--实现统一的dropout
		if(self.cfg.gpu) then
			index[1] = index[1]:cuda() 
		 	index[2] = index[2]:cuda() 
		 	index[3]= index[3]:cuda()
			mask = mask:cuda()
		end
		local rep ={}
		for i=1,3 do 
			vecs[i] = self.LM.emd[i]:forward(index[i]):clone()
			rep[2*i-1] = self.LM.fwd[i]:forward(vecs[i])
			rep[2*i] = self.LM.rwd[i]:forward(vecs[i])
		end
		local feat  = self.LM.md1:forward(rep)
		mask = self.LM.dp:forward(mask)
		local feat2 ={}
		for i =1,3 do
			feat2[i] = feat[i]:cmul(mask)
		end
		local pred = self.LM.md2:forward(feat2)

		local err = criterion:forward(pred,gold)
		sample_count = sample_count + 1
		if err <= 0 then
			right_sample = right_sample + 1
		end
		print(pred[1],err,right_sample/sample_count)
		self.LM.md1:zeroGradParameters()
		self.LM.md2:zeroGradParameters()
		for i =1,3 do
			self.LM.emd[i]:zeroGradParameters()
			self.LM.fwd[i]:zeroGradParameters()
			self.LM.rwd[i]:zeroGradParameters()
		end
		local e0 = criterion:backward(pred,gold)
		e1 = e0  + self.cfg.l2Rate*0.5*params:norm()^2	--二阶范数
		local e2 = self.LM.md2:backward(feat2,e1)
		local e3 = {}
		for i =1,3 do 
			e3[i] = e2[i]:cmul(mask)
		end
		local e4 = self.LM.md1:backward(rep,e3)
		local e5 = {}
		for i = 1,3 do 
			e5[i] = self.LM.fwd[i]:backward(vecs[i],e4[i*2-1])
			e5[i] = e5[i] + self.LM.rwd[i]:backward(vecs[i],e4[i*2])
			self.LM.emd[i]:backward(index[i],e5[i])
		end
		
		local learningRate  = self.cfg.learningRate
		self.LM.md2:updateParameters(learningRate)
		self.LM.md1:updateParameters(learningRate)
		for i =1,3 do
			self.LM.fwd[i]:updateParameters(learningRate)
			self.LM.rwd[i]:updateParameters(learningRate)
			self.LM.emd[i]:updateParameters(learningRate)
		end
	end
	print('训练集的准确率：',right_sample/sample_count)
end

function Sat:test_one_pair(question_vec,answer_id) 	--给定一个问答pair，计算其相似度	
	--传入的qst为已经计算好的向量，ans为问题的id
	local answer_rep = self.dataSet:getAnswerVec(answer_id)	--获取答案的表达
	--print('ans_rep',answer_id,answer_rep[1][1])
	if self.cfg.gpu then
		answer_rep = answer_rep:cuda()
	end
	local sim_sc = self.LM.qt:forward({question_vec,answer_rep})
	return sim_sc[1]
end


function Sat:evaluate(name)
	self.LM.dp:evaluate()
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




