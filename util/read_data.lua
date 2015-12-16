--[[
	写一些读取数据的函数
--]]
--------------------------------------------------------
--输入文件中每行都是一个句子，本函数将该文件转换成为词向量索引的矩阵格式
function deep_cqa.read_sentences(path,vocab)
	local sentences = {}
	local file = io.open(path,'r')
	local line
	while true do 
		line = file:read()
		if line == nil then 
			break 
		end

		local tokens = stringx.split(line)
		local len = #tokens
		local sent = torch.IntTensor(len)
		for i =1,len do
			local token = tokens[i]
			sent[i] = vocab:index(token)
		end
		sentences[#sentences+1] =sent
	end

	file:close()
	return sentences
end
--------------------------------------------------------
--将一个句子转换成为字典索引编码的格式，每个句子一个向量
function deep_cqa.read_one_sentece(sent,vocab)
	local tokens = stringx.split(sent)
	local vecs = torch.IntTensor(#tokens)
	for i = 1 , #tokens do
		local token = tokens[i]
		vecs[i] = vocab:index(token)
	end
	return vecs	--转换完毕返回转换后的向量
end
---------------------------------------------------------
--针对保险数据QA数据集设计的读取函数
function deep_cqa.insurance.load_train()
	local ip = deep_cqa.config.insurance.train
	local file = io.open(ip,'r')
	local line
	local train_set = {}
	while true do
		line = file:read()
		if line == nil then
			break
		end
		local list = line:split('\t')
		table.insert(train_set,{list[1],list[2]:split(' ')})
	end
	file:close()
	deep_cqa.insurance['train'] = train_set
end
-----------------------------------
function deep_cqa.insurance.load_answer()
	local ip = deep_cqa.config.insurance.answer
	local file = io.open(ip,'r')
	local line
	local answer_set = {}
	while true do
		line = file:read()
		if line == nil then
			break
		end
		local list = line:split('\t')
		answer_set[list[1]] = list[2]
	end
	deep_cqa.insurance['answer'] = answer_set
	file:close()
end
---------------------------------------
function deep_cqa.insurance.load_test(name)
	local ip = nil
	if name == 'dev' then
		ip = deep_cqa.config.insurance.dev
	end
	if name == 'test1' then
		ip = deep_cqa.config.insurance.test1
	end
	if name == 'test2' then
		ip = deep_cqa.config.insurance.test2
	end
	if ip == nil then 
		return nil
	end

	local file = io.open(ip,'r')
	local line
	local test_set = {}
	while true do
		line = file:read()
		if line == nil then
			break
		end
		local list = line:split('\t')
		table.insert(test_set,{list[1]:split(' '),list[2],list[3]:split(' ')})
	end
	deep_cqa.insurance[name] = test_set
	file:close()
end
---------------------------------------
--载入保险数据的整体执行函数
function deep_cqa.load_insurance_txt_dataset()
	deep_cqa.insurance.load_train()
	deep_cqa.insurance.load_answer()
	deep_cqa.insurance.load_test('dev')
	deep_cqa.insurance.load_test('test1')
	deep_cqa.insurance.load_test('test2')
end
function deep_cqa.save_insurance_binary()
	if deep_cqa.insurance == nil then
		return nil
	end
	local op = deep_cqa.config.insurance.binary
	torch.save(op,deep_cqa.insurance,'binary')
end
function deep_cqa.load_insurance_binary()
	local ip = deep_cqa.config.insurance.binary
	deep_cqa.insurance = torch.load(ip)
end
----------------------------------------
--保险数据集，构建合适的训练数据和测试数据
function deep_cqa.insurance.generate_train_set()
	local nsize = deep_cqa.config.insurance.negative_size
	local train = {}
	local dataset = deep_cqa.insurance
	for num,item in pairs(dataset['train']) do
		local qst =item[1]
		for i = 1,#item[2] do
			local ta = dataset['answer'][item[2][i]]
			
		end
	end
end
---------------
function deep_cqa.insurance.generate_test_set()
end
