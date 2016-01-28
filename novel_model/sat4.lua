--[[
	在bilstm上添加自我attention的参数
	autor: liangjz
	2016-1-24
--]]
local Sat = torch.class('Sat4')
function Sat:__init(useGPU)
	self.cfg = {
		vecs	= nil,
		dict	= nil,
		emd	= nil,
		dim	=  deep_cqa.config.emd_dim,	--词向量的维度
		mem	= 5,	--Memory的维度
		gpu	= useGPU or false,	--是否使用gpu模式
		dp = 0.5,
		margin	= 0.1,
		l2	= 0,	--L2范式的约束
		lr	= 0.1	--学习率
	}
	self.cfg.dict, self.cfg.vecs = deep_cqa.get_sub_embedding()
	self.cfg.emd = nn.LookupTable(self.cfg.vecs:size(1),self.cfg.dim)
	self.cfg.emd.weight:copy(self.cfg.vecs)
	self.cfg.vecs = nil
	
	self.LM = {}
	self:getLM()
--	self.dataSet = InsSet()	--保险数据集
	print('Dataset Load Over ...')
end
function Sat:getLM()
	self.LM = {}	-- 清空语言模型
	lm = self.LM	-- 简写
	cfg = self.cfg	-- 简写
	lm.emd = {}

	lm.bilstm = {}
	lm.lstm = nn.FastLSTM(cfg.dim, cfg.mem)

	lm.rep = {}
	lm.pip = nn.Sequential()
	lm.pip:add(nn.JoinTable(1))
	lm.pip:add(nn.View(-1,cfg.mem*2))
	lm.pip:add(nn.Replicate(1))
	lm.pip:add(nn.SpatialAdaptiveMaxPooling(cfg.mem*2,1))
	lm.pip:add(nn.Reshape(cfg.mem*2))
	lm.max= {}
	for i = 1,3 do	
		lm.emd[i] = cfg.emd:clone('weight','bias')
		lm.bilstm[i] = nn.BiSequencer(lm.lstm:clone(),lm.lstm:clone(),3)
		lm.bilstm[i]:zeroGradParameters()
		lm.bilstm[i]:remember('both')

		lm.rep[i] = nn.Sequential()
		lm.rep[i]:add(nn.SplitTable(1))
		lm.rep[i]:add(lm.bilstm[i])
		lm.rep[i]:add(nn.JoinTable(1))
		lm.rep[i]:add(nn.View(-1,2,cfg.mem))
		lm.rep[i]:add(nn.SplitTable(2))	--这个这个时候的输出为{n*dim,n*dim},将正反两个序列完全分开了
		lm.max[i] = lm.pip:clone()
	
	end
	lm.diff = {}
	lm.diffWeight = {}
	lm.join1 = {}
	lm.join2 = {}
	lm.join3 = {}
	lm.wordWeight ={}	
	lm.zoom = {}
	lm.scale = {}
	for i = 1,3 do
		lm.diff[2*i-1] = nn.MM(false,false)	--正向序列求差
		lm.diff[2*i] = nn.MM(false,false)	--反向序列求差
		lm.diffWeight[2*i-1] = nil
		lm.diffWeight[2*i] = nil
		lm.join1[i] = nn.JoinTable(2)	--将正反序列进行拼接
		lm.join2[2*i-1] = nn.Concat():add(nn.Identity()):add(nn.Identity())	--正反序列差补全
		lm.join3[i] = nn.JoinTable(2)	--正反序列的差补全
		lm.wordWeight[i] = nn.CosineDistance()	--根据相似性计算权重
		lm.zoom[i] = nn.MM()	--通过给不同timestep的矩阵左乘对角矩阵，来完成对向量表达的缩放
		lm.scale[i] = nil	--空的对角矩阵，到时候根据具体情况进行填充
	end
	
	lm.cosine = {}
	lm.cosine[1] = nn.CosineDistance()
	lm.cosine[2] = nn.CosineDistance()
	lm.sub = nn.CSubTable()
	lm.dp = nn.Dropout(self.cfg.dp)
	
	if self.cfg.gpu then
		for i =1,3 do
			lm.emd[i]:cuda()
			lm.rep[i]:cuda()
			lm.max[i]:cuda()
			lm.diff[2*i-1]:cuda()
			lm.diff[2*i]:cuda()
			lm.join1[i]:cuda()
			lm.join2[i]:cuda()
			lm.join3[i]:cuda()
			lm.wordWeight[i]:cuda()
			lm.zoom[i]:cuda()
		end
		lm.cosine[1]:cuda()
		lm.cosine[2]:cuda()
		lm.sub:cuda()
		lm.dp:cuda()
	end
	for i =2,3 do
		lm.bilstm[i].forwardModule:share(lm.bilstm[1].forwardModule,'weight','bias','gradWeight','gradBias')	
		lm.bilstm[i].backwardModule:share(lm.bilstm[1].backwardModule,'weight','bias','gradWeight','gradBias')
		lm.emd[i]:share(lm.emd[1],'weight','bias')
	end
	self.LM.sub:zeroGradParameters()
	self.LM.cosine[1]:zeroGradParameters()
	self.LM.cosine[2]:zeroGradParameters()
	for i = 1,3 do
		self.LM.rep[i]:zeroGradParameters()
		self.LM.emd[i]:zeroGradParameters()
		self.LM.bilstm[i]:forget()
	end


	
end
function Sat:getIndex(sent) --	获取一个句子的索引表达，作为整个模型的输入，可以直接应用到词向量层
	return deep_cqa.read_one_sentence(sent,self.cfg.dict)
end

function Sat:testLM()
	msg = 'today is a good day !' 
	idx =  self:getIndex(msg)
	vec = self.LM.emd[1]:forward(idx):clone()
	o1 =  self.LM.rep[1]:forward(vec)
	self.LM.bilstm[1]:forget()
--	o2 = self.LM.base[1]:forward(vec)
--	self.LM.bilstm[1]:forget()
	for i,v in pairs(o1) do
		print(v)
	end
--[[
	for i,v in pairs(o2) do
		print(v)
	end
	print(o2)
--]]
	print(idx:size(),'sentence length')
	self.LM.diffWeight[1] =  torch.Tensor(idx:size()[1]-1,idx:size()[1]):zero()
	self.LM.diffWeight[2] =  torch.Tensor(idx:size()[1]-1,idx:size()[1]):zero()
	for i = 1,idx:size()[1]-1 do --构造
		self.LM.diffWeight[1][i][i] = 1
		self.LM.diffWeight[1][i][i+1] = -1	
		self.LM.diffWeight[2][i][i] = -1
		self.LM.diffWeight[2][i][i+1] = 1
	end
	print('fsubw',self.LM.diffWeight[1])
	print('rsubw',self.LM.diffWeight[2])
	local j1 = self.LM.join1[1]:forward(o1)
	print('join1',j1)
	sub ={}
	for i =1,2 do
		sub[i] = self.LM.diff[i]:forward({self.LM.diffWeight[i],o1[i]})
	end
	print(sub[1])
	print(sub[2])
	
	
	
end































function Sat:train(negativeSize)
	self.dataSet:resetTrainset(negativeSize)
	local params= {}
	local grad_params= {}
	for i = 1,3 do
		params[i],grad_params[i] = self.LM.bilstm[i]:parameters()
		for j,p in pairs(params[i]) do
			p:uniform(0,0.1)
		end
	end

	self.LM.dp:training()

	local criterion = nn.MarginCriterion(self.cfg.margin)
	local gold = torch.Tensor({1})
	if self.cfg.gpu then
		criterion:cuda()
		gold = gold:cuda()
	end
	local sample =1			--占个坑
	local vecs={}			--存储词向量
	local index ={}			--存储字符下标
	local rep ={}			--存储双向lstm和maxpooling的输出
	local rep_mask ={}		--存储dropout之后的输出
	local score = {}		--存储问题和正确/错误答案直接的相似性得分
	local loop = 0			--循环变量
	local sample_count = 0	--总样本计数
	local right_sample = 0	--正确样本计数
	while sample ~= nil do	--数据集跑完？
		loop = loop + 1
		if loop %1 ==0 then xlua.progress(self.dataSet.current_train,self.dataSet.train_set.size) end
		sample = self.dataSet:getNextPair()
		if sample == nil then break end	--数据集获取完毕
		for i =1,3 do
			index[i] = self:getIndex(sample[i]):clone()
		end

		local mask =torch.ones(self.cfg.mem*2)	--实现统一的dropout
		if(self.cfg.gpu) then
			index[1] = index[1]:cuda() 
		 	index[2] = index[2]:cuda() 
		 	index[3]= index[3]:cuda()
			mask = mask:cuda()
		end
		mask = self.LM.dp:forward(mask)
		for i = 1,3 do
			vecs[i] = self.LM.emd[i]:forward(index[i]):clone()
			rep[i] = self.LM.rep[i]:forward(vecs[i])
			rep_mask[i] = rep[i]:cmul(mask)
		end
		for i =1,2 do 
			score[i] = self.LM.cosine[i]:forward({rep_mask[1],rep_mask[i+1]})
		end
		local pred = self.LM.sub:forward({score[1],score[2]})	-- 因为是距离参数转换为相似度参数，所以是负样本减正样本
		local err = criterion:forward(pred,gold)
		sample_count = sample_count + 1
		if err <= 0 then
			right_sample = right_sample + 1			
			self.LM.bilstm[1]:forget()
			self.LM.bilstm[2]:forget()
			self.LM.bilstm[3]:forget()
		else
					--print('\n',pred[1],gold[1],err,right_sample/sample_count)
			local epred = criterion:backward(pred,gold)
			local esub = epred 	--此处留二阶范数的约束空
			local ecosine = self.LM.sub:backward(score,esub)
			local erep_mask = {}
			for i =1,2 do
				local tmp = self.LM.cosine[i]:backward({rep_mask[1],rep_mask[1+i]},ecosine[i])
				erep_mask[1] = (erep_mask[1] == nil) and tmp[1] or erep_mask[1]+tmp[1]
				erep_mask[i+1] = tmp[2]
			end
			local erep = {}
			local eemd = {}
			for i = 1,3 do
				erep[i] = erep_mask[i]:cmul(mask)
				eemd[i] = self.LM.rep[i]:backward(vecs[i],erep[i])
				if i==1 then
					eemd[i] = eemd[i]/2		--问题的误差来自两个部分，所以除2标准化
				end
				self.LM.emd[i]:backward(index[i],eemd[i])
			end
			local lr = self.cfg.lr
			self.LM.sub:zeroGradParameters()
			self.LM.cosine[1]:zeroGradParameters()
			self.LM.cosine[2]:zeroGradParameters()
	
			for i = 1,3 do
				self.LM.rep[i]:updateParameters(lr)
				self.LM.rep[i]:zeroGradParameters()
				self.LM.emd[i]:updateParameters(lr)
				self.LM.emd[i]:zeroGradParameters()
				self.LM.bilstm[i]:forget()
			end
		end
	end
	print('训练集的准确率：',right_sample/sample_count)
end

function Sat:testOnePair(question_vec,answer_id) 	--给定一个问答pair，计算其相似度	
	--传入的qst为已经计算好的向量，ans为问题的id
	local answer_rep = self.dataSet:getAnswerVec(answer_id)	--获取答案的表达
	if self.cfg.gpu then
		answer_rep = answer_rep:cuda()
	end
	local sim_sc = self.LM.cosine[1]:forward({question_vec,answer_rep})
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
		local answer_emd = self.LM.emd[2]:forward(word_index):clone()
		local answer_rep = self.LM.rep[2]:forward(answer_emd):clone()
		self.LM.bilstm[2]:forget()
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
		local qst_emd = self.LM.emd[1]:forward(qst_idx):clone()
		local qst_vec = self.LM.rep[1]:forward(qst_emd):clone()
		self.LM.bilstm[1]:forget()

		local sc = {}	
		local gold_sc ={}
		local gold_rank = {}
		
		for k,c in pairs(gold) do 
			local score = self:testOnePair(qst_vec,c)	--标准答案的得分,传入内容为问题的表达和答案的编号
			gold_sc[k] = score
			gold_rank[k] = 1	--初始化排名
		end

		for k,c in pairs(candidates) do 
			local score = self:testOnePair(qst_vec,c)
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
