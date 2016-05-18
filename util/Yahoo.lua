--[[
	Yahoo manner 数据集的warper，用类的形式提供服务
	liangjz 2015-3-22
	训练集，通过get_sample获得一个三元组
	测试集
	问题数据，通过
--]]
--require('..')
local cjson = require('cjson')
local DataSet = torch.class('Yahoo')	--保险语料库的读取warper
function DataSet:__init(name,negative_size)
	local data_file =nil
	if name=='train' then
		data_file = 'data/yahoo/family.json'
	end
	if name=='dev' then
		data_file = 'data/yahoo/business.json'
	end
	if name=='test' then
		data_file = 'data/yahoo/health.json'
	end
	local tmp= cjson.decode(io.open(data_file,'r'):read())
	self.questions = tmp[1]
	self.answer_set = tmp[2]
	self.indices = torch.randperm(#self.questions)	--对训练样本进行乱序处理
	self.current_train = 1	--当前样本的下标	,一个样本只有一个正确答案
	self.current_negative = 1	--当前选取的负样本为第多少个负样本（针对这个训练样本来说）
	self.negative_size =negative_size or 1	--负样本采样个数,默认为1
	self.answer_vecs ={}	--存储答案的中间结果
	self.answer_index = 1	--遍历答案时的下标
	self.dev_index = 0	--遍历验证集时下标
	self.test_index = 0	--遍历测试集时的下标
end

function DataSet:getNextPair()	--生成下一对问题-正样本-负样本对
	if self.current_train > #self.questions then return nil end	--数据集已经遍历完毕
	local qst = self.questions[self.current_train][1]  ..' ' .. self.questions[self.current_train][2]	--一个问题
	local true_id = self.questions[self.current_train][3]	--一个正确答案
	local false_id = deep_cqa.ins_meth.random_negative_id({self.questions[self.current_train][3]},#self.questions-1,deep_cqa.config.random_seed)
	deep_cqa.config.random_seed = (deep_cqa.config.random_seed +5)%50000000	--随机种子
	true_id  = tostring(tonumber(true_id))
	false_id  = tostring(tonumber(false_id))
	local false_answer = self.answer_set[false_id]
	local true_answer = self.answer_set[true_id]
	self.current_negative = self.current_negative + 1
	if self.current_negative > self.negative_size then
		self.current_negative = 1
		self.current_train = self.current_train + 1
	end
	return {qst,true_answer,false_answer}	--返回sample三元组
end

function DataSet:resetTrainset(negative_size)	--重新设定训练集
	self.indices = torch.randperm(#self.questions)	--对训练样本进行乱序处理
	self.current_train = 1	--当前样本的下标
	self.current_negative = 1	--当前选取的负样本为第多少个负样本（针对这个训练样本来说）
	self.negative_size =negative_size or self.negative_size	--负样本采样个数
end

function DataSet:getAnswer(answer_id)	--根据编号返回答案的文本内容
	answer_id = tostring(tonumber(answer_id))
	return self.answer_set[answer_id]
end

function DataSet:getNextAnswer(marker)	--返回下一个编号的答案
	local reset = (marker == true) or false	--是否从头开始
	if reset then self.answer_index = 0 end
	if self.answer_index >= #self.questions then return nil end	--超出下标则返回空答案
	self.answer_index = self.answer_index + 1
	return {self.answer_index-1,self.answer_set[tostring(self.answer_index-1)]}	--返回答案下标和答案内容的二元组
end

function DataSet:saveAnswerVec(answer_id,answer_vec)	--保存答案的向量表达形式
	answer_id = tonumber(answer_id)
	self.answer_vecs[answer_id] = answer_vec	
end

function DataSet:getAnswerVec(answer_id)	--读取答案的向量表达形式
	answer_id = tonumber(answer_id)
	return self.answer_vecs[answer_id]
end

function DataSet:getNextTest(marker)	--获取下一个验证集样本	【正确ids，问题，候选ids】
	local reset = (marker == true) or false	--是否从头开始
	if reset then self.dev_index = 1 end
	if self.dev_index > #self.questions then return nil end	--超出下标则返回空验证组
	local tmp ={}
	tmp[1] = self.questions[self.dev_index][1] .. ' ' .. self.questions[self.dev_index][2]
	tmp[2] = self.questions[self.dev_index][3]
	tmp[3] = self.questions[self.dev_index][4]
	self.dev_index = self.dev_index + 1
	return tmp	--返回验证集的三元组

end
--[[
function DataSet:getNextTest(marker)	--获取下一个测试集样本
	print('getNextTest method is empty')
	local reset = (marker == true) or false	--是否从头开始
	if reset then self.test_index = 1 end
	if self.test_index > self.test_set.size then return nil end	--超出下标则返回空答案
	self.test_index = self.test_index + 1 
	return self.test_set[self.test_index + 1]
end
--]]

