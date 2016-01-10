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
	local wc = {}
	for i,item in pairs(deep_cqa.insurance['train']) do
		item[1] = string.gsub(item[1],'\r','')
		local qst = item[1]:split(' ')
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
end

function get_idf()
	deep_cqa.ins_meth.load_binary()	--载入语料库
	local qwc = {}	--问题部分的词频统计，包括两个部分，绝对词频和出现该词的文档数量
	local awc = {}	--同上，是针对答案部分的记录
	local cache ={}
	local qst_wc = 0	--问题中（train）单词个数
	local qst_ac = 0	--问题中（train）句子个数
	local ans_wc = 0	--答案中（train）单词个数
	local ans_ac = 0	--答案中（train）句子个数

	for i,item in pairs(deep_cqa.insurance['train']) do
		qst_ac = qst_ac + 1
		item[1] = string.gsub(item[1],'\r','')
		local qst = item[1]:split(' ')
		for j,word in pairs(qst) do
			qst_wc = qst_wc + 1
			if cache[word] == nil then
				cache[word] = 0
			end
			cache[word] = cache[word]+1
		end
		----------------------
		for word,count in pairs(cache) do
			if qwc[word] ==nil then 
				qwc[word] ={0,0}
			end
			qwc[word][1] = qwc[word][1]+1
			qwc[word][2] = qwc[word][2]+count
		end
		cache ={}
		----------------------
	end
	for i,item in pairs(deep_cqa.insurance['answer']) do
		ans_ac = ans_ac + 1
		item = string.gsub(item,'\r','')
		local qst = item:split(' ')
		for j,word in pairs(qst) do
			ans_wc = ans_wc + 1
			if cache[word] == nil then
				cache[word] = 0
			end
			cache[word] = cache[word]+1
		end
		----------------------
		for word,count in pairs(cache) do
			if awc[word] ==nil then 
				awc[word] ={0,0}
			end
			awc[word][1] = awc[word][1]+1
			awc[word][2] = awc[word][2]+count
		end
		cache ={}
	end

	------------------------- 合并两个table awc,qwc
	local max ={0,0}
	local aqwc ={}
	for i,v in pairs(awc) do
		aqwc[i] = v
		if v[1]> max[1] then max[1]= v[1]end
		if v[2]> max[2] then max[2]= v[2]end
	end
	for i,v in pairs(qwc) do
		if aqwc[i]== nil then
			aqwc[i] ={0,0}
		end

		aqwc[i][1] = (aqwc[i][1] + v[1])
		aqwc[i][2] = (aqwc[i][2] + v[2])
		if aqwc[i][1] > max[1] then max[1] = aqwc[i][1] end
		if aqwc[i][2] > max[2] then max[2] = aqwc[i][2] end
	end
	minidf={}
	maxidf={}
	minidf[1] = (qst_ac+ans_ac)/max[1] --math.log( (qst_ac+ans_ac)/max[1])
	minidf[2] =(qst_wc+ans_wc)/max[2] --math.log( (qst_wc+ans_wc)/max[2])
	maxidf[1] = qst_ac+ans_ac    -- math.log(qst_wc+ans_wc)
	maxidf[2] = qst_wc+ans_wc    -- math.log(qst_wc+ans_wc)
	for i,v in pairs(aqwc) do
	--	aqwc[i][1] =math.log((0.0+qst_ac+ans_ac)/aqwc[i][1])
	--	aqwc[i][2] =math.log((0.0+qst_wc+ans_wc)/aqwc[i][2])
		--aqwc[i][1] = (math.log((0.0+qst_ac+ans_ac)/aqwc[i][1])-minidf[1])/(maxidf[1]-minidf[1])
		--aqwc[i][2] =(math.log((0.0+qst_wc+ans_wc)/aqwc[i][2])-minidf[2])/(maxidf[2]-minidf[2])
		aqwc[i][1] = (((0.0+qst_ac+ans_ac)/aqwc[i][1])-minidf[1])/(maxidf[1]-minidf[1])
		aqwc[i][2] =(((0.0+qst_wc+ans_wc)/aqwc[i][2])-minidf[2])/(maxidf[2]-minidf[2])


	end
	-------------------------
	torch.save(deep_cqa.config.word_count,aqwc)
	print(aqwc)
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
get_idf()
--co_matrix()
