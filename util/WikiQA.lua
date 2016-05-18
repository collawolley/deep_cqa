--[[
	WikiQA的warper，用类的形式提供服务
	liangjz 2015-5-18
	调用规范：
	getNextSample 获取一个训练的三元组，包含打乱操作
	getNextTest  获取一组验证集或者测试集
--]]
--require('..')
local Wiki = torch.class('WikiQA')	--保险语料库的读取warper
function Wiki:__init(negative_size) --WikiQA 数据集本身给出的负样本是基于上下文的，所以也不需要考虑

	self.corpus=torch.load(deep_cqa.config.wiki.corp)
	self.train_set  = self.corpus['train']
	self.dev_set   = self.corpus['dev']
	self.test_set = self.corpus['test']

	self.indices = torch.randperm(self.train_set.size)	--对训练样本进行乱序处理

	self.current_train = 1	--当前样本的下标
	self.dev_index = 1	--遍历验证集时下标
	self.test_index = 1	--遍历测试集时的下标

end
function Wiki:getNextPair()	--遍历整个训练集
	if self.current_train > self.train_set.size then return nil end	--数据集已经遍历完毕
	local qst = self.train_set[self.indices[self.current_train]][1]	--一个问题
	local true_answer = self.train_set[self.indices[self.current_train]][2]	--正确答案
	local false_answer = self.train_set[self.indices[self.current_train]][3]	--错误答案
	self.current_train = self.current_train + 1
	return {qst,true_answer,false_answer}	--返回sample三元组
end

function Wiki:resetTrainset(negative_size)	--重新设定训练集
	self.indices = torch.randperm(self.train_set.size)	--对训练样本进行乱序处理
	self.current_train = 1	--当前样本的下标
end

function Wiki:getNextDev(marker)	--获取下一个验证集样本	【正确ids，问题，候选ids】
	local reset = (marker == true) or false	--是否从头开始
	if reset  then self.dev_index = 1 end
	if self.dev_index > self.dev_set.size then return nil end	--超出下标则返回空验证组
	self.dev_index = self.dev_index + 1
	return self.dev_set[self.dev_index -1]	--返回验证集的三元组

end

function Wiki:getNextTest(marker)	--获取下一个测试集样本
local reset = (marker == true) or false	--是否从头开始
	if reset then self.test_index = 1 end
	if self.test_index > self.test_set.size then return nil end	--超出下标则返回空答案
	self.test_index = self.test_index + 1 
	return self.test_set[self.test_index + 1]
end

