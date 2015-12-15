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
--读入一个语料库，并将其转换成为合适处理的格式
---------------------------------------------------------
