--[[
	word vector reading layer
--]]
require('.')
local dict = deep_cqa.Vocab(deep_cqa.config.dict)
local emb_vec = deep_cqa.config.emd_vec
local emb_dict = deep_cqa.config.emb_dict
print(emb_vec)
