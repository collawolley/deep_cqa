--[[
    统计语料中的一些文本共现信息
    包括：截止词、词共现矩阵
    liangjz 2016-1-4
--]]
require('..')
function cmp(i,v)
	return i[2]>v[2]
end
dataSet = InsSet()
dataSet:resetTrainset(1)
function get_stopwords()
	print('getting stop words')
	local wc = {}
	sample = dataSet:getNextPair()
	print(sample)
	while sample ~=nil do
		for i =1,2 do
			local sent = string.gsub(sample[i],'\r','')
			sent = sent:split(' ')
			for j,w  in pairs(sent) do
				if wc[w]==nil then
					wc[w]=0
				end
				wc[w] = wc[w] + 1
			end
		end
		print(sample[1])
		sample = dataSet:getNextPair()
	end
	print(wc)
end

function get_pmi()
	local qwc = {}
	local awc = {}
	local all_count = 0
	local cm = {}
	dataSet:resetTrainset(1)
	sample = dataSet:getNextPair()
	print(sample)
	while sample ~=nil do
		local qst = string.gsub(sample[1],'\r','')
		qst = qst:split(' ')
		local tas = string.gsub(sample[2],'\r','')
		tas = tas:split(' ')
		for i,v in pairs(qst) do
			for j,x in pairs(tas) do
				if qwc[v]==nil then
					qwc[v]=0
					cm[v]={}
				end
				if awc[x]==nil then
					awc[x]=0
				end
				if cm[v][x]==nil then
					cm[v][x]=0
				end
				qwc[v] = qwc[v]+1
				awc[x] = awc[x]+1
				cm[v][x] = cm[v][x]+1
				all_count = all_count +1
			end
		end
		sample = dataSet:getNextPair()
	end
	for i,v in pairs(cm) do
		for j,x in pairs(v) do
			pq = qwc[i]/all_count
			pa = awc[j]/all_count
		--	print('before',i,j,x,pq,pa,all_count)
			x = x/all_count
			x= x/pa/pq
		--	print(x)
		end
	end
	torch.save('insurance_pmi.bin',cm)
end
-------------------------------------------------------
--get_stopwords()
get_pmi()
