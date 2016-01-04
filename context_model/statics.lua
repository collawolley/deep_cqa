--[[
    统计语料中的一些文本共现信息
    包括：截止词、词共现矩阵
    liangjz 2016-1-4
--]]
require('..')
function cmp(i,v)
	return i[2]>v[2]
end
function get_stopwords()
	deep_cqa.ins_meth.load_binary()	--载入语料库
	wc = {}
	for i,item in pairs(deep_cqa.insurance['train']) do
		item[1] = string.gsub(item[1],'\r','')
		local qst = item[1]:split(' ')
		--print(qst)
		for j,word in pairs(qst) do
			if wc[word] == nil then
				wc[word] = 0
			end
			wc[word] = wc[word]+1
		end
	end
	for i,item in pairs(deep_cqa.insurance['answer']) do
		item = string.gsub(item,'\r','')
		local qst = item:split(' ')
		--print(qst)
		for j,word in pairs(qst) do
			if wc[word] == nil then
				wc[word] = 0
			end
			wc[word] = wc[word]+1
		end
	end
	for i,item in pairs(deep_cqa.insurance['dev']) do
		item[2] = string.gsub(item[2],'\r','')
		local qst = item[2]:split(' ')
		--print(qst)
		for j,word in pairs(qst) do
			if wc[word] == nil then
				wc[word] = 0
			end
			wc[word] = wc[word]+1
		end
	end	
	for i,item in pairs(deep_cqa.insurance['test1']) do
		item[2] = string.gsub(item[2],'\r','')
		local qst = item[2]:split(' ')
		--print(qst)
		for j,word in pairs(qst) do
			if wc[word] == nil then
				wc[word] = 0
			end
			wc[word] = wc[word]+1
		end
	end
	for i,item in pairs(deep_cqa.insurance['test2']) do
		item[2] = string.gsub(item[2],'\r','')
		local qst = item[2]:split(' ')
		--print(qst)
		for j,word in pairs(qst) do
			if wc[word] == nil then
				wc[word] = 0
			end
			wc[word] = wc[word]+1
		end
	end
	local wca = {}
	for i,v in pairs(wc) do
		table.insert(wca,{i,v})
	end
	table.sort(wca,cmp)
	sw = {}
	for i,v in pairs(wca) do
		if i<= 50 then
			sw[v[1]] = i
		end
	end
	torch.save(deep_cqa.config.stop_words,sw)
--	print(sw)
end
-------------------------------------------------------
function co_matrix()
	deep_cqa.ins_meth.load_binary()	--载入语料库
	co = {}
	sw = torch.load(deep_cqa.config.stop_words)
	answer = deep_cqa.insurance['answer']
	for i,item in pairs(deep_cqa.insurance['train']) do
		--if i >100 then break end
		item[1] = string.gsub(item[1],'\r','')
		local qst = item[1]:split(' ')
		for j,word in pairs(qst) do
			if sw[word] ==nil then	--非截止词
				if co[word] == nil then --每个词一个列表，类似于稀疏矩阵的存储方式
					co[word] ={}
				end
				for k,id in pairs(item[2]) do
					id = tostring(tonumber(id))
					a = answer[id]:split(' ')
					for l,w in pairs(a) do
						if sw[w] == nil then
							if co[word][w] == nil then
								co[word][w] = 0
							end
							co[word][w] = co[word][w]+1
						end
					end
				end
				
			end
		end
	end
--	print(co)
	torch.save(deep_cqa.config.co_matrix,co,'binary')
end
--get_stopwords()
--co_matrix()
