--[[
	使用rnn库实现bilstm,主要目标，解决参数同步问题
	autor: liangjz
	2016-1-24
--]]
local Sat = torch.class('Sat3')
function Sat:__init(useGPU)
	self.cfg = {
		vecs	= nil,
		dict	= nil,
		emd	= nil,
		dim	=  deep_cqa.config.emd_dim,	--词向量的维度
		mem	= 100,	--Memory的维度
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
	self.dataSet = InsSet()	--保险数据集
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

	for i = 1,3 do	
		lm.emd[i] = cfg.emd:clone('weight','bias')
		lm.bilstm[i] = nn.BiSequencer(lm.lstm:clone(),lm.lstm:clone())
		lm.bilstm[i]:zeroGradParameters()
		lm.bilstm[i]:remember('both')
		lm.rep[i] = nn.Sequential()
		lm.rep[i]:add(nn.SplitTable(1))
		lm.rep[i]:add(lm.bilstm[i])
		lm.rep[i]:add(lm.pip:clone())
	end
	lm.cosine = {}
	lm.cosine[1] = nn.CosineDistance()
	lm.cosine[2] = nn.CosineDistance()
	lm.sub = nn.CSubTable()
	lm.dp = nn.Dropout(self.cfg.dp)
	
	if self.cfg.gpu then
		lm.emd[1]:cuda()
		lm.emd[2]:cuda()
		lm.emd[3]:cuda()
		lm.rep[1]:cuda()
		lm.rep[2]:cuda()
		lm.rep[3]:cuda()
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
	local params= {}
	local grad_params= {}
	for i = 1,3 do
		params[i],grad_params[i] = self.LM.bilstm[i]:parameters()
		for j,p in pairs(params[i]) do
			p = p:uniform(-0.01,0.01) 	-- ndeel some little trick
		end
	end	
end
function Sat:getIndex(sent) --	获取一个句子的索引表达，作为整个模型的输入，可以直接应用到词向量层
	return deep_cqa.read_one_sentence(sent,self.cfg.dict)
end

function Sat:testLM()
	local brnn = self.LM.bilstm[1]
	local brnn2 = self.LM.bilstm[2]
  	local inputs, gradOutputs = {}, {}
   	local inputs2, gradOutputs2 = {}, {}
	for loop =1,3 do
		nStep =3
		nStep = nStep + loop
		for i=1,nStep do
      		inputs[i] = torch.randn(self.cfg.dim)
			inputs2[i] = torch.randn(self.cfg.dim)
      		gradOutputs[i] = torch.randn(self.cfg.mem*2)
		  	gradOutputs2[i] = torch.randn(self.cfg.mem*2)+loop
   		end
   		local outputs = brnn:forward(inputs)
		local outputs2 = brnn2:forward(inputs2)
   		local gradInputs = brnn:backward(inputs, gradOutputs)
   		local gradInputs2= brnn2:backward(inputs2, gradOutputs2)
   -- params
   		local params, gradParams = brnn:parameters()
  		local params2, gradParams2 = brnn2:parameters()
   
   -- updateParameters
		brnn:updateParameters(0.1)
   		brnn2:updateParameters(0.1)
   		brnn:zeroGradParameters()
   		brnn2:zeroGradParameters()
		for i,param in pairs(params) do
			if i< 3 then
				print(param)
				print(params2[i])
			end
   		end
	print('------------')
	end
end

function Sat:train(negativeSize)
	self.dataSet:resetTrainset(negativeSize)
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
--		if loop %1 ==0 then xlua.progress(self.dataSet.current_train,self.dataSet.train_set.size) end
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
			--print('right sample',loop)
		else
			--print('\n',pred[1],gold[1],err,right_sample/loop)
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
--	local answer_rep = self:refine('today is a good day')
	local answer_rep = self.dataSet:getAnswerVec(answer_id)	--获取答案的表达
	--[
	if self.cfg.gpu then
		answer_rep = answer_rep:cuda()
	end
--]
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
--[
	while answer_pair~=nil do
		loop = loop+1
--		xlua.progress(loop,self.dataSet.answer_set.size)
		local answer = answer_pair[2]	--获取问题内容
--[
		local word_index = self:getIndex(answer)	--获取词下标
		if self.cfg.gpu then word_index = word_index:cuda() end
		local answer_emd = self.LM.emd[2]:forward(word_index):clone()
		local answer_rep = self.LM.rep[2]:forward(answer_emd):clone()
		self.LM.bilstm[2]:forget()
--]
		--local answer_rep = self:refine(answer):clone()
		self.dataSet:saveAnswerVec(answer_pair[1],answer_rep)
		answer_pair = self.dataSet:getNextAnswer()
	end	
--]
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
--		xlua.progress(loop,test_size)

		local gold = test_pair[1]	--正确答案的集合
		local qst = test_pair[2]	--问题
		local candidates = test_pair[3] --候选的答案
--[
		local qst_idx = self:getIndex(qst)
		if self.cfg.gpu then qst_idx = qst_idx:cuda() end
		local qst_emd = self.LM.emd[1]:forward(qst_idx):clone()
		local qst_vec = self.LM.rep[1]:forward(qst_emd):clone()
		self.LM.bilstm[1]:forget()
--]
--		local qst_vec = self:refine(qst):clone()
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
function Sat:refine(sent)
	if self.vlm == nil then	--创建重构语言模型
		self.vlm = {}
		self.vlm.rep = nn.Sequential()
		self.vlm.rep:add(self.LM.emd[1])	--获取词向量
		self.vlm.rep:add(nn.SplitTable(1))	--分割成为table
		self.vlm.rep:add(self.LM.bilstm[1])	--获取双向lstm的表达
		self.vlm.resize = nn.Sequential()	--分离bilstm的表达
		self.vlm.resize:add(nn.JoinTable(1))	
		self.vlm.resize:add(nn.View(-1,2,self.cfg.mem))
		self.vlm.resize:add(nn.SplitTable(2))
		self.vlm.diff ={}
		self.vlm.join1 = {}
		self.vlm.join2 = {}
		self.vlm.diffW = {}
		for i = 1,2 do
			self.vlm.diff[i] =  nn.MM(false,false)
			self.vlm.diffW[i] = nil
			self.vlm.join1[i] = nn.JoinTable(1)
			self.vlm.join2 = nn.JoinTable(2)
			self.vlm.join3 = nn.JoinTable(2)
			self.vlm.cos = nn.CosineDistance()
		end
		if self.cfg.gpu then 
			self.vlm.rep:cuda() 
			self.vlm.resize:cuda() 
			self.vlm.join2:cuda()
			self.vlm.join3:cuda()
			self.vlm.cos:cuda()
			for i = 1,2 do 
				self.vlm.diff[i]:cuda()
				self.vlm.join1[i]:cuda()
			end
		end
	end	
	local idx =  self:getIndex(sent):clone()
	self.LM.bilstm[1]:forget()
	local size = idx:size()[1]
	self.vlm.diffW[1] =  torch.Tensor(size-1,size):zero()
	self.vlm.diffW[2] =  torch.Tensor(size-1,size):zero()
	for i = 1,size-1 do --构造
		self.vlm.diffW[1][i][i] = -1
		self.vlm.diffW[1][i][i+1] = 1	
		self.vlm.diffW[2][i][i] = 1
		self.vlm.diffW[2][i][i+1] = -1
	end
	if self.cfg.gpu then 
		self.vlm.diffW[1] = self.vlm.diffW[1]:cuda() 
		self.vlm.diffW[2] = self.vlm.diffW[2]:cuda() 
		idx = idx:cuda()
	end	
	local rep1 = self.vlm.rep:forward(idx)
	local rep2 = self.vlm.resize:forward(rep1)

	local sub= {}
	for  i = 1,2 do
		sub[i] = self.vlm.diff[i]:forward({self.vlm.diffW[i],rep2[i]})
	end
	local j1 = {}
	local size=  idx:size()[1]
	j1[1] = self.vlm.join1[1]:forward({sub[2][1]:resize(1,self.cfg.mem),sub[1]})
	j1[2] = self.vlm.join1[2]:forward({sub[2],sub[1][idx:size()[1]-1]:resize(1,self.cfg.mem)})
	local j2 = self.vlm.join2:forward(j1)	--各个状态之间的差
	local weight = torch.Tensor(size):zero()
	for i =1, size do	
		weight[i] = j2[i]:norm()
	end
	local sum = weight:sum()
	if self.cfg.gpu then 
		weight = weight:cuda()
	end	
	--local result = torch.Tensor(1,self.cfg.mem*2):zero():cuda()
	for i =1,size do 
		weight[i] = weight[i]/sum*size
	end
	for i =1,size do
		rep1[i] = rep1[i]*weight[i]
--		result = result + rep1[i]
	end	
	local result = self.LM.pip:forward(rep1):cuda()
	
	--print(result)
	--print(weight)
	return result
end

function Sat:demo(sent)
	vlm = {}
	vlm.rep = nn.Sequential()
	vlm.rep:add(self.LM.emd[1])
	vlm.rep:add(nn.SplitTable(1))
	vlm.rep:add(self.LM.bilstm[1])
	vlm.rep:add(nn.JoinTable(1))
	vlm.rep:add(nn.View(-1,2,self.cfg.mem))
	vlm.rep:add(nn.SplitTable(2))
	vlm.diff ={}
	vlm.join1 = {}
	vlm.join2 = {}
	vlm.diffW = {}
	for i = 1,2 do
		vlm.diff[i] =  nn.MM(false,false)
		vlm.diffW[i] = nil
		vlm.join1[i] = nn.JoinTable(1)
		vlm.join2 = nn.JoinTable(2)
		vlm.join3 = nn.JoinTable(2)
		vlm.cos = nn.CosineDistance()
	end
	if self.cfg.gpu then 
		vlm.rep:cuda() 
		vlm.join2:cuda()
		vlm.join3:cuda()
		vlm.cos:cuda()
		for i = 1,2 do 
			vlm.diff[i]:cuda()
			vlm.join1[i]:cuda()
		end
	end
	
--	sent= self.dataSet:getNextAnswer(true)	--从头开始计算answer的向量
	local sent= self.dataSet:getNextDev(true)
	while sent~=nil do
	local sample = {}
--	sample[1] = sent[2]
--	sample[2] =self.dataSet:getAnswer(sent[1][1])
	sample[1] = 'what be car insurance base on'
	sample[2] = 'what be health insurance base on'
	sample[3] = 'what be house insurance base on'
	for loop = 1,3 do
		local idx =  self:getIndex(sample[loop]):clone()
		self.LM.bilstm[1]:forget()
		local rep = vlm.rep:forward(idx)
		vlm.diffW[1] =  torch.Tensor(idx:size()[1]-1,idx:size()[1]):zero()
		vlm.diffW[2] =  torch.Tensor(idx:size()[1]-1,idx:size()[1]):zero()

		for i = 1,idx:size()[1]-1 do --构造
			vlm.diffW[1][i][i] = -1
			vlm.diffW[1][i][i+1] = 1	
			vlm.diffW[2][i][i] = 1
			vlm.diffW[2][i][i+1] = -1
		end
		if self.cfg.gpu then 
			idx = idx:cuda() 
			vlm.diffW[1] = vlm.diffW[1]:cuda() 
			vlm.diffW[2] = vlm.diffW[2]:cuda() 
		end
		local sub= {}
		for  i = 1,2 do
			sub[i] = vlm.diff[i]:forward({vlm.diffW[i],rep[i]})
		end
		local j1 = {}
		local size=  idx:size()[1]
		j1[1] = vlm.join1[1]:forward({sub[2][1]:resize(1,self.cfg.mem),sub[1]})
		j1[2] = vlm.join1[2]:forward({sub[2],sub[1][idx:size()[1]-1]:resize(1,self.cfg.mem)})
		
		local j2 = vlm.join2:forward(j1)
--		ht = vlm.join3:forward({rep[1][size]:resize(1,self.cfg.mem),rep[2][1]:resize(1,self.cfg.mem)})

		local pp =''
		local ps = ''
		local rank = {}
		local map  = {}
		local key = {}
		for  i =1,idx:size()[1] do
			local w = self.cfg.dict:token(idx[i])
			map[j2[i]:norm()] = i
			rank[i] = w
			table.insert(key,j2[i]:norm())
		end
		table.sort(key)
		for i=1,idx:size()[1] do
			local word = self.cfg.dict:token(idx[i])
			local score1 = string.format('%.2f',j2[i]:norm())
--			local score2 = string.format('%.2f',vlm.cos:forward({ht[1],j2[i]})[1])
			local tmp  = ' ' .. word ..'-'.. score1-- .. ' ' ..score2 
			pp = pp .. tmp
		end
		for i = #key,1,-1 do
			local score = key[i]
			local r = map[score]
			local w = rank[r]
			local s = string.format('%.2f',score)
			ps = ps.. ' ' .. w ..'-' .. s 
		end
			
		print(pp)
		print('\n')
		print(ps)
		print('\n')
	end
	break
	sent= self.dataSet:getNextDev(false)
	print('##############################################\n')
--	sent= self.dataSet:getNextAnswer(false)	--从头开始计算answer的向量
	end
end
