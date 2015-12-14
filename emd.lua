--[[
	word vector reading layer
--]]
require('.')
emd ={}
local dict = deep_cqa.Vocab(deep_cqa.config.dict)
local emd_vec = deep_cqa.config.emd_vec
local emd_dict = deep_cqa.config.emb_dict
emd_dict,emd_vec = emd.read_embedding(emd_dict,emd_vec)
print(emd_dict)

-----------------------------------------------------------
function emd.read_embedding(vocab_path, emb_path)
  local vocab = treelstm.Vocab(vocab_path)
  local embedding = torch.load(emb_path)
  return vocab, embedding
end
------------------------------------------------------------