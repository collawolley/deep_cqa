require('.')--[[
a =nn.Linear(5,5)
a['weight']:zero()
a['bias']:zero()
for i =1,5 do
	a['weight'][i][i] =1
end
print(a['weight'])
b = torch.Tensor(5):fill(3)
print(b)
print(a:forward(b))
--]
a = torch.Tensor({3,3,3,3,3,3})
b = nn.View(-1,2)
c= b:forward(a)
print(c)
e = c*0.1
print(e)
d = b:backward(a,e)
print(d)

--]]

--[
local tmp = Sat1(true)
--tmp:train(1)
tmp:evaluate('dev')
--tmp:testLM()
--[[
tmp:train(1)
tmp:evaluate('dev')
--]]

