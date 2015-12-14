--[[
	word vector reading layer
--]]
request('.')
request('config.lua')
local EMD,parent  = torch.class('deep_cqa.EMD','nn.Module')
printf('good')