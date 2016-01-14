--[[
	读取词向量的相关操作
--]]
----------------------------------------------------------
function deep_cqa.read_embedding(vocab_path, emb_path)
	--从文件读取词向量和字典
	local vocab = deep_cqa.Vocab(vocab_path)
	local embedding = torch.load(emb_path)
	return vocab, embedding
end
------------------------------------------------------------
function deep_cqa.get_sub_embedding()
	local dict	= deep_cqa.Vocab(deep_cqa.config.insurance.dict)	--字典子集
	local emd_vec	=	deep_cqa.config.emd_vec		--完整的词向量
	local emd_dict	=	deep_cqa.config.emd_dict	--完整的字典
	local emd_dim	=	deep_cqa.config.emd_dim		--词向量的维度
	emd_dict,emd_vec	=	deep_cqa.read_embedding(emd_dict,emd_vec)
	local vecs = torch.Tensor(dict.size,emd_dim)
	local unseen_count = 0
	for i =1, dict.size do
		local w = string.gsub(dict:token(i),'\\','')
		if emd_dict:contains(w) then
			vecs[i] = emd_vec[emd_dict:index(w)]
			--vecs[i]:uniform(-0.05,0.05)
		else
			unseen_count =unseen_count + 1
			vecs[i]:uniform(-0.05,0.05)
		end
	end
	print('unseen words count:',unseen_count,dict.size)
	emd_vec = nil
	emd_dict = nil
	return dict,vecs
end
-------------------------------------------------------------
