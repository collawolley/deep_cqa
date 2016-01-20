require('.')
--[[
data = torch.rand(2,5)
print('data',data)
scale = torch.rand(1,2)
l = nn.CMul(2,5)
print(l:forward(data))
print(l['weight'],l['bias'])
--]]
local tmp = Trans3(true)
tmp:train(1)
tmp:evaluate('dev')
