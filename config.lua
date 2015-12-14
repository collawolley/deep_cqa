--[[
	项目的运行配置参数，用一个Config类来专门存储各项配置
--]]
local Config = torch.class('deep_cqa.Config')

function Config:__init( ... )
	self.test = 'every thing is good'
end