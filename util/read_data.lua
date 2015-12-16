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
function deep_cqa.ins_meth.load_train()
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
function deep_cqa.ins_meth.load_answer()
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
	print(#answer_set)
	file:close()
end
---------------------------------------
function deep_cqa.ins_meth.load_test(name)
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
function deep_cqa.ins_meth.load_txt_dataset()
	deep_cqa.ins_meth.load_train()
	deep_cqa.ins_meth.load_answer()
	deep_cqa.ins_meth.load_test('dev')
	deep_cqa.ins_meth.load_test('test1')
	deep_cqa.ins_meth.load_test('test2')
end
----------
function deep_cqa.ins_meth.save_binary()
	if deep_cqa.insurance == nil then
		return nil
	end
	local op = deep_cqa.config.insurance.binary
	torch.save(op,deep_cqa.insurance,'binary')
end
----------
function deep_cqa.ins_meth.load_binary()
	local ip = deep_cqa.config.insurance.binary
	deep_cqa.insurance = torch.load(ip)
end
----------------------------------------
--保险数据集，构建合适的训练数据和测试数据
function deep_cqa.ins_meth.generate_train_set()
	local nsize = deep_cqa.config.insurance.negative_size
	local train = {}
	local dataset = deep_cqa.insurance
	local answer = dataset['answer']
	local answer_size = deep_cqa.get_size(answer)
	local seed =1
	for num,item in pairs(dataset['train']) do
		local qst =item[1]
		for i = 1,#item[2] do
			for j =1, nsize do
				local aid = item[2][i]
				local aid = tostring(tonumber(aid))
				local ta = answer[aid]
				seed = seed + 5
				local fa = deep_cqa.ins_meth.random_negative_id(item[2],answer_size,seed)
				fa = answer[fa]
				table.insert(train,{qst,ta,fa})
			end
		end
	end
	--table.sort(train,fuzzy_cmp)	--这条语句不能用
	--print(train)
	torch.save(deep_cqa.ins_meth.train,train)
	return train
end
-------------------------------
function fuzzy_cmp(a,b)	--比较函数的返回值不稳定，在lua中无法执行
	--随机获取一个answer id，该id不在传入的列表当中
	math.randomseed(tonumber(tostring(os.time()):reverse():sub(1, 7))+20000*deep_cqa.config.random_seed)
	deep_cqa.config.random_seed =(deep_cqa.config.random_seed +3)%5000
	if math.random()-0.5 > 0 then
		return true
	end
	return false
end
------------------------
function deep_cqa.get_size(tab)
	local count =0
	local i=nil
	local v=nil
	for i,v in pairs(tab) do
		count = count + 1
	end
	return count
end
-------------------------
function deep_cqa.ins_meth.random_negative_id(list,size,seed)
	--随机获取一个answer id，该id不在传入的列表当中
	math.randomseed(tonumber(tostring(os.time()):reverse():sub(1, 7))+20000*seed)
	local id =nil
	while true do
		local mark = nil
		id = math.random(size)
		seed =seed + 1
		for i = 1, #list do
			if id == tonumber(list[i]) then
				mark = 1
			end
		end
		if mark == nil then
			break
		end
	end
	return tostring(id)
end
---------------
function deep_cqa.ins_meth.generate_test_set(name)
	

end
