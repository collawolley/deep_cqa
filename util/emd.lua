--[[
	读取词向量的相关操作
--]]
----------------------------------------------------------
function deep_cqa.read_embedding(vocab_path, emb_path)
  local vocab = deep_cqa.Vocab(vocab_path)
  local embedding = torch.load(emb_path)
  return vocab, embedding
end
------------------------------------------------------------
function deep_cqa.get_sub_embedding()
	local dict = deep_cqa.Vocab(deep_cqa.config.dict)
	local emd_vec = config.emd_vec
	local emd_dict = config.emd_dict
	local emd_dim =  config.emd_dim
	emd_dict,emd_vec = read_embedding(emd_dict,emd_vec)
	local vecs = torch.Tensor(dict.size,emd_dim)
	local unseen_count = 0
	for i =1, dict.size do
	local w =  string.gsub(dict:token(i),'\\','')
	print(w)
		if emd_dict:contains(w) then
			vecs[i] = emd_vec[emd_dict:index(w)]
		else
			unseen_count =unseen_count + 1
			vecs[i]	= uniform(-0.05,0.05)
		end
	end
	print('unseen words count:',unseen_count)
	emd_vec = nil
	emd_dict = nil
	return dict,vecs
end
-------------------------------------------------------------
deep_cqa.get_sub_embedding()
